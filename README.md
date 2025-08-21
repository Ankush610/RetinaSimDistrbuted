## 📖 Project Overview: RetinaSim — Retinal Vein Map Generation Using CycleGAN

**RetinaSim** is a deep learning project aimed at generating retinal vein maps (annotations) from realistic retinal images. We utilize a **CycleGAN architecture** to perform image-to-image translation between real retina images and their corresponding vein annotations.

For scalable and efficient training in a **clustered environment**, we leverage the **Hugging Face `accelerate` library** in combination with **SLURM** workload manager. This enables us to distribute the training workload across multiple nodes and processors for faster experimentation.

---

## 📦 Environment Setup

You can set up the environment using either **Conda** or a **Python virtual environment**:

### 🔹 Using Conda:

```bash
conda env create -f environment.yaml
conda activate img2img-turbo
```

### 🔹 Using Virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Configuration

* Modify the `accelerate_temp.yaml` file to configure your cluster environment settings (such as number of processes, nodes, etc.).
* Update the `job.sh` SLURM script according to your job submission preferences.
* Before running the script run `python model_download.py` it will download all the models.
* Place your image data and corresponding annotation maps in the `data/` directory, structured appropriately for your use case.

---

## 🚀 Training

After setup:

1. Configure your `accelerate_temp.yaml`.
2. Submit the job using the `job.sh` script in your SLURM-managed environment.


