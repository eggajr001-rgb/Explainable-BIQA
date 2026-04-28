# Explainable Blind Image Quality Assessment with Closed-Loop Semantic Guidance and Distortion Diagnosis


---

## 📖 Introduction

This repository provides the official inference code for the paper:

> **Explainable Blind Image Quality Assessment with Closed-Loop Semantic Guidance and Distortion Diagnosis**

The proposed framework aims to provide interpretable perception assessment for visual systems. It supports:

- **Quality Prediction:** Predicts perceptual quality scores using a dual-stream architecture that integrates semantic priors from a frozen CLIP model into a Vision Transformer (ViT) via zero-initialized FiLM.
- **Distortion Diagnosis:** Identifies dominant distortion types through a dedicated diagnosis branch, providing interpretable feedback for downstream decision-making.

---

## 🚀 Performance

The model achieves strong consistency with human subjective judgments and reliable distortion diagnosis:

| Dataset    | SRCC   | PLCC   | Diagnostic Accuracy |
|------------|--------|--------|---------------------|
| TID2013    | 0.9509 | 0.9552 | 96.17%              |
| KADID-10k  | 0.9408 | 0.9435 | 91.48%              |

*Results are averaged over five independent trials.*

---

## 📂 Project Structure

    ├── predict_one_image.py   # Entry point for single-image inference
    ├── config.py              # Configuration settings
    ├── models/                # Network architecture (ViT + semantic modules)
    ├── utils/                 # Preprocessing and utilities
    ├── checkpoints/           # Pretrained weights (to be downloaded)
    └── requirements.txt       # Dependencies


---

## 🛠️ Environment Setup

Install dependencies:

    pip install -r requirements.txt

**Core Dependencies:**
- PyTorch >= 1.10.0
- timm >= 0.4.12
- open_clip_torch >= 2.0.0
- einops

---

## 📥 Pre-trained Model Weights

Due to file size (~1GB), pretrained weights are hosted externally:

* **Download Link:** https://pan.baidu.com/s/1eIrfHgxcisLDusNDqQ0iuw?pwd=1234
* **Extraction Code:** `1234`

After downloading, place the `.pth` file into: `./checkpoints/`

---

## 💻 Usage

Run inference on a single image:

    python predict_one_image.py --img_path ./path_to_your_image.jpg

**Output:**
- Predicted quality score
- Predicted distortion type

---

## 🔗 Citation

If you find this work useful, please cite:

    @article{Song2026Explainable,
      title={Explainable Blind Image Quality Assessment with Closed-Loop Semantic Guidance and Distortion Diagnosis},
      author={Song, Chenye and Yuan, Fujiang and Zhang, Zhiwang},
      journal={Scientific Reports},
      year={2026}
    }

---

## 📄 License

This project is licensed under the MIT License.
