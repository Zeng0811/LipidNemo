# LipidNemo: Predicting In Vivo Organ Tropism of Lipid Nanoparticles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-brightgreen)](#)
## 📌 Overview
**LipidNemo** is an advanced machine learning framework designed to accurately predict the *in vivo* organ tropism (Liver, Lung, Spleen, or None) of Lipid Nanoparticles (LNPs). 

By leveraging a fine-tuned Large Language Model (LLM) to extract high-dimensional molecular representations from LNP formulations (SMILES and molar ratios), and integrating a powerful downstream classifier (TabPFN / 2D Transformer), LipidNemo provides a high-throughput, AI-assisted tool for the rational design and optimization of LNPs for targeted drug delivery.

### ✨ Key Features
- **SMILES-to-Embedding Extraction:** Utilizes a fine-tuned LLM to convert LNP formulation structures into robust numerical embeddings.
- **Transductive Learning Optimization:** Incorporates transductive PCA to allow the model to perceive target data distributions, significantly enhancing generalization capabilities on unseen LNPs.
- **High Accuracy & Interpretability:** Supports diverse downstream classifiers with comprehensive evaluations (Accuracy, ROC-AUC, and Confusion Matrices).

---

## 🖼️ Architecture
<img width="1256" height="538" alt="image" src="https://github.com/user-attachments/assets/86a2e9f6-5a07-4cb1-baf1-2f74e834794c" />

> Figure 1: Schematic overview of the LipidNemo pipeline.

---

## ⚙️ Installation

We recommend using [Conda](https://docs.conda.io/en/latest/) to manage your environment.

```bash
# 1. Clone the repository
git clone [https://github.com/](https://github.com/)[your-username]/LipidNemo.git
cd LipidNemo

# 2. Create a virtual environment
conda create -n lipidnemo python=3.10 -y
conda activate lipidnemo

# 3. Install required packages
pip install -r requirements.txt
```

## Fine-tuning the LLM
The core molecular representation model of LipidNemo is inherited from MegaMolBART within BioNeMo by NVIDIA. To ensure a reproducible environment and avoid complex dependency issues, we highly recommend utilizing the official BioNeMo Docker container for LLM fine-tuning and embedding extraction.

### 1. Launch the Docker Container
Ensure you have Docker and the NVIDIA Container Toolkit installed. You will also need your personal NGC API Key to pull the framework. 

Run the following command in your terminal from the root directory of this repository to mount the necessary data, configs, and scripts into the container:

```bash
docker run --gpus all -it --name LipidNemo_LLM --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --shm-size=1g \
  -v $(pwd)/models/checkpoints:/workspace/bionemo/models/checkpoints \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/workspace/bionemo/results \
  -v $(pwd)/scripts/config/my_pretrain.py:/workspace/bionemo/examples/molecule/megamolbart/my_pretrain.py \
  -v $(pwd)/scripts/config/my_testing.py:/workspace/bionemo/examples/molecule/megamolbart/my_testing.py \
  -v $(pwd)/scripts/config/my_pretrain_base.yaml:/workspace/bionemo/examples/molecule/megamolbart/conf/my_pretrain_base.yaml \
  -e NGC_CLI_API_KEY="YOUR_NGC_API_KEY" \
  -p 8888:8888 \
  nvcr.io/nvidia/clara/bionemo-framework:1.10 \
  bash
```

Replace "YOUR_NGC_API_KEY" with your actual NVIDIA NGC API key. You can create your own Key in [NGC](https://catalog.ngc.nvidia.com/) and download the MegaMolBART base model, following the instruction from [BioNeMo](https://docs.nvidia.com/bionemo-framework/latest/main/getting-started/access-startup/).


### 2. Excecute fine-tuning
```bash
python /workspace/bionemo/examples/molecule/megamolbart/my_pretrain.py \
    --config-path=/workspace/bionemo/examples/molecule/megamolbart/conf \
    --config-name=my_pretrain_base \
    do_training=True \
    trainer.devices=1 \
    trainer.num_nodes=1 \
    trainer.accelerator='gpu' \
    restore_from_path=/workspace/bionemo/models/MegaMolBART.nemo
```
This process utilizes [SwissLipids](https://www.swisslipids.org/#/), a public lipid library for fine-tuning, as illustrated in my_pretrain_base.yaml:
```
  data:
    dataset_path: /data # parent directory for data, contains train / val / test folders. Needs to be writeable for index creation.
    dataset:
      train: lipids_smile_train_filtered
      test: lipids_smile_test_filtered
      val: lipids_smile_val_filtered
```
Users are required to download SwissLipids manually and split the data into 98% train, 1% validation, and 1% test.

### 3. Extract LipidNemo Embeddings
To embed and concatenate the LNP formulation,run the noteboke Embedding-LipidNemo.ipynb in the container.

## 2D transformer-based classifier for organ selectivity prediction
To map the high-dimensional latent embeddings extracted by the LLM to macroscopic biological behaviors, LipidNemo utilizes a 2D Transformer-based classifier. Unlike traditional 1D models that struggle with the complex, non-linear combinatorial effects of multi-component nanoparticles, 2D architecture is specifically designed to understand the formulation as an integrated delivery system. Herein, TabPFN is used to predict the organ selectivity of LNPs.
```
conda activate lipidnemo
python python PCA_transductive.py && python classifier.py
```
