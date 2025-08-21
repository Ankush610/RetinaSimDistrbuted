import os # for os operations
import gc # for garbage collection
import copy # for deep copy
import lpips # for perceptual loss
import torch 
import wandb # for logging
from glob import glob # for file globbing (ex *.jpg, *.png)
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed 
from PIL import Image # for image processing
from torchvision import transforms
from tqdm.auto import tqdm 
from transformers import AutoTokenizer, CLIPTextModel 
from diffusers.optimization import get_scheduler # for learning rate scheduler
from peft.utils import get_peft_model_state_dict # for getting the state dict of the model
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance # for FID calculation
import vision_aided_loss # for vision aided loss
from model import make_1step_sched # for making the scheduler
from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.training_utils import UnpairedDataset, build_transform, parse_args_unpaired_training
from my_utils.dino_struct import DinoStructureLoss 
import time


def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps, 
        log_with=args.report_to
    )
    set_seed(args.seed)
    
    # ADDED: Training stopping criterion - choose between 'epochs' or 'steps'
    stopping_criterion = getattr(args, 'stopping_criterion', 'steps')  # default to 'steps' for backward compatibility
    if stopping_criterion not in ['epochs', 'steps']:
        raise ValueError("stopping_criterion must be either 'epochs' or 'steps'")
    
    if accelerator.is_main_process:
        print(f"Training stopping criterion: {stopping_criterion}")
        if stopping_criterion == 'epochs':
            print(f"Will stop after {args.max_train_epochs} epochs")
        else:
            print(f"Will stop after {args.max_train_steps} steps")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        "stabilityai/sd-turbo", 
        subfolder="tokenizer", 
        revision=args.revision, 
        use_fast=False,
    )
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained(
        "stabilityai/sd-turbo", 
        subfolder="text_encoder"
    ).cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(
        args.lora_rank_unet, 
        return_lora_module_names=True
    )
    vae_a2b, vae_lora_target_modules = initialize_vae(
        args.lora_rank_vae, 
        return_lora_module_names=True
    )

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(
            cv_type='clip', 
            loss_type=args.gan_loss_type, 
            device="cuda"
        )
        net_disc_a.cv_ensemble.requires_grad_(False)
        net_disc_b = vision_aided_loss.Discriminator(
            cv_type='clip', 
            loss_type=args.gan_loss_type, 
            device="cuda"
        )
        net_disc_b.cv_ensemble.requires_grad_(False)

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    optimizer_gen = torch.optim.AdamW(
        params_gen, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon,
    )

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(
        params_disc, 
        lr=args.learning_rate, 
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, 
        eps=args.adam_epsilon,
    )

    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder, 
        image_prep=args.train_img_prep, 
        split="train", 
        tokenizer=tokenizer
    )
    
    # FIXED: Proper dataloader with optimized settings
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.train_batch_size, 
        shuffle=True, 
        num_workers=args.dataloader_num_workers,
        pin_memory=True,  # Add pin memory for faster data transfer
        drop_last=True,   # Drop last incomplete batch
        persistent_workers=args.dataloader_num_workers > 0  # Keep workers alive
    )
    
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # OPTIMIZATION: Only compute FID reference on main process and cache it
    if accelerator.is_main_process:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        
        # FID reference statistics for A -> B translation
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        
        # Check if reference already computed
        ref_stats_file = os.path.join(output_dir_ref, "ref_stats.npz")
        if os.path.exists(ref_stats_file):
            ref_stats = np.load(ref_stats_file)
            a2b_ref_mu, a2b_ref_sigma = ref_stats['mu'], ref_stats['sigma']
            print("Loaded cached A2B reference statistics")
        else:
            # Compute reference statistics
            for _path in tqdm(l_images_tgt_test, desc="Processing A2B reference images"):
                _img = T_val(Image.open(_path).convert("RGB"))
                outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
                if not os.path.exists(outf):
                    _img.save(outf)
            
            ref_features = get_folder_features(
                output_dir_ref, 
                model=feat_model, 
                num_workers=0, 
                num=None,
                shuffle=False, 
                seed=0, 
                batch_size=8, 
                device=torch.device("cuda"),
                mode="clean", 
                custom_fn_resize=None, 
                description="", 
                verbose=True,
                custom_image_tranform=None
            )
            a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
            np.savez(ref_stats_file, mu=a2b_ref_mu, sigma=a2b_ref_sigma)
            print("Computed and cached A2B reference statistics")

        # FID reference statistics for B -> A translation
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        
        ref_stats_file = os.path.join(output_dir_ref, "ref_stats.npz")
        if os.path.exists(ref_stats_file):
            ref_stats = np.load(ref_stats_file)
            b2a_ref_mu, b2a_ref_sigma = ref_stats['mu'], ref_stats['sigma']
            print("Loaded cached B2A reference statistics")
        else:
            for _path in tqdm(l_images_src_test, desc="Processing B2A reference images"):
                _img = T_val(Image.open(_path).convert("RGB"))
                outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
                if not os.path.exists(outf):
                    _img.save(outf)
            
            ref_features = get_folder_features(
                output_dir_ref, 
                model=feat_model, 
                num_workers=0, 
                num=None,
                shuffle=False, 
                seed=0, 
                batch_size=8, 
                device=torch.device("cuda"),
                mode="clean", 
                custom_fn_resize=None, 
                description="", 
                verbose=True,
                custom_image_tranform=None
            )
            b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
            np.savez(ref_stats_file, mu=b2a_ref_mu, sigma=b2a_ref_sigma)
            print("Computed and cached B2A reference statistics")

    lr_scheduler_gen = get_scheduler(
        args.lr_scheduler, 
        optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, 
        power=args.lr_power
    )
    lr_scheduler_disc = get_scheduler(
        args.lr_scheduler, 
        optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, 
        power=args.lr_power
    )

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_a2b_tokens = tokenizer(
        fixed_caption_tgt, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    
    fixed_b2a_tokens = tokenizer(
        fixed_caption_src, 
        max_length=tokenizer.model_max_length, 
        padding="max_length", 
        truncation=True, 
        return_tensors="pt"
    ).input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(
        unet, vae_enc, vae_dec, net_disc_a, net_disc_b
    )
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = 0
    
    # FIXED: Calculate total steps properly
    num_batches_per_epoch = len(train_dataloader)
    total_epochs = args.max_train_epochs if hasattr(args, 'max_train_epochs') else (args.max_train_steps // num_batches_per_epoch) + 1
    
    # IMPROVED: Better progress tracking with flexible stopping
    if stopping_criterion == 'epochs':
        # When stopping by epochs, progress bar tracks epochs
        progress_bar = tqdm(
            range(0, args.max_train_epochs), 
            initial=first_epoch, 
            desc="Training Epochs",
            disable=not accelerator.is_local_main_process,
        )
        max_steps_for_progress = float('inf')  # No step limit
    else:
        # When stopping by steps, progress bar tracks steps
        progress_bar = tqdm(
            range(0, args.max_train_steps), 
            initial=global_step, 
            desc="Training Steps",
            disable=not accelerator.is_local_main_process,
        )
        max_steps_for_progress = args.max_train_steps
    
    # OPTIMIZATION: Turn off efficient attention for discriminators
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    # IMPROVED: Add timing and throughput tracking
    epoch_start_time = time.time()
    step_times = []
    
    print(f"Starting training:")
    print(f"  Dataset size: {len(dataset_train)}")
    print(f"  Batch size: {args.train_batch_size}")
    print(f"  Batches per epoch: {num_batches_per_epoch}")
    print(f"  Max epochs: {total_epochs}")
    print(f"  Max steps: {args.max_train_steps}")
    print(f"  Number of processes: {accelerator.num_processes}")
    print(f"  Stopping criterion: {stopping_criterion}")

    for epoch in range(first_epoch, total_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        if accelerator.is_main_process:
            print(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")
        
        for step, batch in enumerate(train_dataloader):
            # IMPROVED: Check stopping condition early in the loop
            if stopping_criterion == 'steps' and global_step >= args.max_train_steps:
                if accelerator.is_main_process:
                    print(f"\nReached maximum steps ({args.max_train_steps}). Stopping training.")
                break
            
            step_start_time = time.time()
            
            # FIXED: Proper gradient accumulation context
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor(
                    [noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, 
                    device=img_a.device
                ).long()

                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(
                    img_a, "a2b", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_a2b_emb
                )
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(
                    cyc_fake_b, "b2a", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_b2a_emb
                )
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                
                # B -> fake A -> rec B  
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(
                    img_b, "b2a", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_b2a_emb
                )
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(
                    cyc_fake_a, "a2b", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_a2b_emb
                )
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                
                total_cycle_loss = loss_cycle_a + loss_cycle_b
                accelerator.backward(total_cycle_loss, retain_graph=False)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator GAN Objective - Generate fresh fake images for GAN loss
                """
                with torch.no_grad():
                    # Store the fake images for discriminator training (detached)
                    fake_b_for_disc = CycleGAN_Turbo.forward_with_networks(
                        img_a, "a2b", vae_enc, unet, vae_dec, 
                        noise_scheduler_1step, timesteps, fixed_a2b_emb
                    ).detach()
                    fake_a_for_disc = CycleGAN_Turbo.forward_with_networks(
                        img_b, "b2a", vae_enc, unet, vae_dec, 
                        noise_scheduler_1step, timesteps, fixed_b2a_emb
                    ).detach()
                
                # Generate fresh fake images for generator loss (with gradients)
                gan_fake_b = CycleGAN_Turbo.forward_with_networks(
                    img_a, "a2b", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_a2b_emb
                )
                gan_fake_a = CycleGAN_Turbo.forward_with_networks(
                    img_b, "b2a", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_b2a_emb
                )
                
                loss_gan_a = net_disc_a(gan_fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(gan_fake_a, for_G=True).mean() * args.lambda_gan
                total_gan_loss = loss_gan_a + loss_gan_b
                
                accelerator.backward(total_gan_loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Identity Objective
                """
                idt_a = CycleGAN_Turbo.forward_with_networks(
                    img_b, "a2b", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_a2b_emb
                )
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                
                idt_b = CycleGAN_Turbo.forward_with_networks(
                    img_a, "b2a", vae_enc, unet, vae_dec, 
                    noise_scheduler_1step, timesteps, fixed_b2a_emb
                )
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator Training - Use detached fake images
                """
                # Fake discriminator losses (use pre-computed detached fakes)
                loss_D_A_fake = net_disc_a(fake_b_for_disc, for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a_for_disc, for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5

                # Real discriminator losses  
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5

                # Combined discriminator loss
                total_disc_loss = loss_D_fake + loss_D_real
                accelerator.backward(total_disc_loss, retain_graph=False)
                
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            # IMPROVED: Better logging with timing
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append(step_time)
            
            logs = {
                "epoch": epoch + 1,
                "step_in_epoch": step + 1,
                "cycle_a": loss_cycle_a.detach().item(),
                "cycle_b": loss_cycle_b.detach().item(),
                "gan_a": loss_gan_a.detach().item(),
                "gan_b": loss_gan_b.detach().item(),
                "disc_a": loss_D_A_fake.detach().item() + loss_D_A_real.detach().item(),
                "disc_b": loss_D_B_fake.detach().item() + loss_D_B_real.detach().item(),
                "idt_a": loss_idt_a.detach().item(),
                "idt_b": loss_idt_b.detach().item(),
                "step_time": step_time,
                "lr_gen": lr_scheduler_gen.get_last_lr()[0],
                "lr_disc": lr_scheduler_disc.get_last_lr()[0],
            }
            
            epoch_loss += (total_cycle_loss + total_gan_loss + loss_g_idt).detach().item()

            if accelerator.sync_gradients:
                if stopping_criterion == 'epochs':
                    # For epoch-based training, just update step counter
                    global_step += 1
                else:
                    # For step-based training, update progress bar
                    progress_bar.update(1)
                    global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    
                    # OPTIMIZED: Less frequent visualization 
                    if global_step % args.viz_freq == 1:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                    "train/rec_a": [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                    "train/rec_b": [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                    "train/fake_b": [wandb.Image(gan_fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                    "train/fake_a": [wandb.Image(gan_fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(min(bsz, 4))],
                                }
                                tracker.log(log_dict)

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {
                            "l_target_modules_encoder": l_modules_unet_encoder,
                            "l_target_modules_decoder": l_modules_unet_decoder,
                            "l_modules_others": l_modules_unet_others,
                            "rank_unet": args.lora_rank_unet,
                            "sd_encoder": get_peft_model_state_dict(eval_unet, adapter_name="default_encoder"),
                            "sd_decoder": get_peft_model_state_dict(eval_unet, adapter_name="default_decoder"),
                            "sd_other": get_peft_model_state_dict(eval_unet, adapter_name="default_others"),
                            "rank_vae": args.lora_rank_vae,
                            "vae_lora_target_modules": vae_lora_target_modules,
                            "sd_vae_enc": eval_vae_enc.state_dict(),
                            "sd_vae_dec": eval_vae_dec.state_dict(),
                        }
                        torch.save(sd, outf)

                    # OPTIMIZED: Less frequent validation with better memory management
                    if global_step % args.validation_steps == 1 and accelerator.is_main_process:
                        print(f"\n--- Running validation at step {global_step} ---")
                        _timesteps = torch.tensor(
                            [noise_scheduler_1step.config.num_train_timesteps - 1] * 1, 
                            device="cuda"
                        ).long()
                        
                        # Use context manager for DINO to ensure cleanup
                        with torch.cuda.device(accelerator.device):
                            net_dino = DinoStructureLoss()
                            
                            try:
                                # Evaluate A->B
                                fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                                os.makedirs(fid_output_dir, exist_ok=True)
                                l_dino_scores_a2b = []
                                
                                val_limit = min(args.validation_num_images, len(l_images_src_test)) if args.validation_num_images > 0 else len(l_images_src_test)
                                
                                for idx, input_img_path in enumerate(tqdm(l_images_src_test[:val_limit], desc="Validating A2B")):
                                    outf = os.path.join(fid_output_dir, f"{idx}.png")
                                    
                                    with torch.no_grad():
                                        input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                        img_a = transforms.ToTensor()(input_img)
                                        img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                        
                                        eval_fake_b = CycleGAN_Turbo.forward_with_networks(
                                            img_a, "a2b", eval_vae_enc, eval_unet,
                                            eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1]
                                        )
                                        eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                        eval_fake_b_pil.save(outf)
                                        
                                        a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                        b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                        dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                        l_dino_scores_a2b.append(dino_ssim)
                                
                                dino_score_a2b = np.mean(l_dino_scores_a2b)
                                gen_features = get_folder_features(
                                    fid_output_dir, model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_fn_resize=None, description="", verbose=True,
                                    custom_image_tranform=None
                                )
                                ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                                score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                                
                                # Evaluate B->A
                                fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_b2a")
                                os.makedirs(fid_output_dir, exist_ok=True)
                                l_dino_scores_b2a = []
                                
                                val_limit = min(args.validation_num_images, len(l_images_tgt_test)) if args.validation_num_images > 0 else len(l_images_tgt_test)
                                
                                for idx, input_img_path in enumerate(tqdm(l_images_tgt_test[:val_limit], desc="Validating B2A")):
                                    outf = os.path.join(fid_output_dir, f"{idx}.png")
                                    
                                    with torch.no_grad():
                                        input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                        img_b = transforms.ToTensor()(input_img)
                                        img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
                                        
                                        eval_fake_a = CycleGAN_Turbo.forward_with_networks(
                                            img_b, "b2a", eval_vae_enc, eval_unet,
                                            eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1]
                                        )
                                        eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                        eval_fake_a_pil.save(outf)
                                        
                                        a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                        b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                        dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                        l_dino_scores_b2a.append(dino_ssim)
                                
                                dino_score_b2a = np.mean(l_dino_scores_b2a)
                                gen_features = get_folder_features(
                                    fid_output_dir, model=feat_model, num_workers=0, num=None,
                                    shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                                    mode="clean", custom_fn_resize=None, description="", verbose=True,
                                    custom_image_tranform=None
                                )
                                ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                                score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                                
                                print(f"step={global_step}, fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}")
                                print(f"step={global_step}, fid(b2a)={score_fid_b2a:.2f}, dino(b2a)={dino_score_b2a:.3f}")
                                
                                logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                                logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                                
                            except Exception as e:
                                print(f"Validation error: {e}")
                            finally:
                                del net_dino  # Ensure cleanup
                                torch.cuda.empty_cache()

            # IMPROVED: Better progress display with throughput metrics
            if len(step_times) >= 10:
                avg_step_time = np.mean(step_times[-10:])
                throughput = args.train_batch_size / avg_step_time
                logs["avg_step_time"] = avg_step_time
                logs["images_per_sec"] = throughput
            
            progress_bar.set_postfix(**{k: f"{v:.4f}" if isinstance(v, float) else v for k, v in logs.items()})
            accelerator.log(logs, step=global_step)
            
            # IMPROVED: Flexible early stopping condition
            if stopping_criterion == 'steps' and global_step >= args.max_train_steps:
                break
        
        # IMPROVED: Update epoch progress bar if using epoch-based stopping
        if stopping_criterion == 'epochs':
            progress_bar.update(1)
        
        # IMPROVED: End-of-epoch logging
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start
        avg_loss = epoch_loss / len(train_dataloader)
        
        if accelerator.is_main_process:
            print(f"Epoch {epoch + 1} completed:")
            print(f"  Duration: {epoch_duration:.2f}s")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Steps in epoch: {len(train_dataloader)}")
            print(f"  Global step: {global_step}")
            if len(step_times) > 0:
                print(f"  Average step time: {np.mean(step_times):.3f}s")
                print(f"  Throughput: {args.train_batch_size / np.mean(step_times):.2f} images/sec")
        
        # IMPROVED: Flexible break condition
        if stopping_criterion == 'steps' and global_step >= args.max_train_steps:
            if accelerator.is_main_process:
                print(f"\nReached maximum steps ({args.max_train_steps}). Stopping training.")
            break
        elif stopping_criterion == 'epochs' and epoch + 1 >= args.max_train_epochs:
            if accelerator.is_main_process:
                print(f"\nCompleted {args.max_train_epochs} epochs. Stopping training.")
            break
    
    # IMPROVED: Final training summary
    if accelerator.is_main_process:
        total_time = time.time() - epoch_start_time
        print(f"\n=== Training Completed ===")
        print(f"Total training time: {total_time:.2f}s")
        print(f"Total steps: {global_step}")
        print(f"Average step time: {np.mean(step_times):.3f}s")
        print(f"Final throughput: {args.train_batch_size / np.mean(step_times):.2f} images/sec")


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)