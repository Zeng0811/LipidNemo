# LipidNemo: Predicting In Vivo Organ Tropism of Lipid Nanoparticles

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-brightgreen)](#) ## 📌 Overview
**LipidNemo** is an advanced machine learning framework designed to accurately predict the *in vivo* organ tropism (Liver, Lung, Spleen, or None) of Lipid Nanoparticles (LNPs). 

By leveraging a fine-tuned Large Language Model (LLM) to extract high-dimensional molecular representations from LNP formulations (SMILES and molar ratios), and integrating a powerful downstream classifier (TabPFN / 2D Transformer), LipidNemo provides a high-throughput, AI-assisted tool for the rational design and optimization of LNPs for targeted drug delivery.

### ✨ Key Features
- **SMILES-to-Embedding Extraction:** Utilizes a fine-tuned LLM to convert LNP formulation structures into robust numerical embeddings.
- **Transductive Learning Optimization:** Incorporates transductive PCA to allow the model to perceive target data distributions, significantly enhancing generalization capabilities on unseen LNPs.
- **High Accuracy & Interpretability:** Supports diverse downstream classifiers with comprehensive evaluations (Accuracy, ROC-AUC, and Confusion Matrices).

---

## 🖼️ Architecture
![LipidNemo Workflow](./assets/figure1a.png) 
> *Figure 1: Schematic overview of the LipidNemo pipeline. Formulations are processed by a fine-tuned LLM, followed by transductive PCA and a downstream classifier with feature/sample attention mechanisms.*
*(Note: Create an `assets` folder and put your Figure 1a image there, then update this path)*

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
