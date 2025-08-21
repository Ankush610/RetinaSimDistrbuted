from huggingface_hub import hf_hub_download

def download_sd_turbo_model():
    try:
        # Download the main model file from the repo
        hf_hub_download(repo_id="runwayml/stable-diffusion-v1-5", filename="model_index.json")
        print(f"[✔] Stable Diffusion Turbo model metadata downloaded to HuggingFace cache.")
    except Exception as e:
        print(f"[✖] Failed to download Stable Diffusion Turbo model: {e}")

download_sd_turbo_model()
