import os
import torch
import torchvision.models as models
from huggingface_hub import hf_hub_download
from torchvision.models import VGG16_Weights

def download_clip_model():
    try:
        import clip
        cache_dir = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)
        model, _ = clip.load("ViT-B/32", device="cpu")
        print(f"[✔] CLIP ViT-B/32 model downloaded to: {cache_dir}")
    except Exception as e:
        print(f"[✖] Failed to download CLIP model: {e}")

def download_vgg16_model():
    try:
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        print(f"[✔] VGG16 model successfully downloaded to: {cache_dir}")
    except Exception as e:
        print(f"[✖] Failed to download VGG16 model: {e}")

def download_dino_model():
    try:
        cache_dir = os.path.expanduser("~/.cache/torch/hub/checkpoints")
        os.makedirs(cache_dir, exist_ok=True)
        model_url = "https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        destination_path = os.path.join(cache_dir, "dino_vitbase8_pretrain.pth")
        torch.hub.download_url_to_file(model_url, destination_path)
        print(f"[✔] DINO ViT-Base model downloaded to: {destination_path}")
    except Exception as e:
        print(f"[✖] Failed to download DINO model: {e}")

def download_sd_turbo_model():
    try:
        # Download the main model file from the repo
        hf_hub_download(repo_id="stabilityai/sd-turbo", filename="model_index.json")
        print(f"[✔] Stable Diffusion Turbo model metadata downloaded to HuggingFace cache.")
    except Exception as e:
        print(f"[✖] Failed to download Stable Diffusion Turbo model: {e}")

def main():
    print("Starting model downloads...\n")
    download_clip_model()
    download_vgg16_model()
    download_dino_model()
    download_sd_turbo_model()
    print("\nAll downloads attempted.")

if __name__ == "__main__":
    main()
