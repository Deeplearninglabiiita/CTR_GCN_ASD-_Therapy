# ASD Behavioral Analysis using Multi-Task CTR-GCN

## Overview

This project presents a multi-task graph convolutional framework for Autism Spectrum Disorder (ASD) behavioral analysis using privacy-preserving 3D skeletal data. The proposed framework extends Channel-wise Topology Refinement Graph Convolutional Networks (CTR-GCN) with temporal attention, joint attention, supervised contrastive learning, and auxiliary clinical prediction heads for ASD severity and Restricted and Repetitive Behavior (RRB) estimation.

The system is trained and evaluated on the MMASD dataset and further deployed on real-world videos using MediaPipe-based skeleton extraction and sliding-window temporal inference. In addition to activity recognition, the framework provides interpretable behavioral analysis through attention visualization and wrist-distance motion reasoning.

---

## Features

* 11-class ASD intervention activity recognition
* ASD severity estimation using ADOS-based labels
* Restricted and Repetitive Behavior (RRB) prediction
* CTR-GCN backbone with adaptive graph learning
* Temporal attention and joint attention mechanisms
* Supervised contrastive learning
* Weighted cross-entropy for class imbalance handling
* Sliding-window real-time video inference
* MediaPipe-based skeleton extraction
* Wrist-distance behavioral motion analysis
* Attention heatmap visualization
* t-SNE embedding analysis

---

## Dataset

This project uses the MMASD (Multimodal Autism Spectrum Disorder) dataset, a privacy-preserving dataset collected during ASD intervention sessions. The framework utilizes ROMP-generated 3D skeletal representations consisting of 24 body joints.

The dataset includes:

* Robotic intervention activities
* Rhythm-based activities
* Yoga-based therapeutic activities
* Clinical severity annotations
* Restricted and Repetitive Behavior (RRB) labels

---

## Proposed Framework

The proposed pipeline consists of:

1. 3D skeleton preprocessing
2. CTR-GCN spatio-temporal feature extraction
3. Temporal attention refinement
4. Joint attention refinement
5. Shared embedding learning
6. Multi-task prediction heads:

   * Activity recognition
   * Severity prediction
   * RRB prediction
7. Real-world behavioral inference using sliding-window analysis

---

## Project Structure

```text
CTR-GCN-ASD/
│
├── model/
├── graph/
├── feeder/
├── config/
├── Test/
├── plotsgraphs/
├── processed_data/
├── results/
├── paper_figures/
├── main.py
├── extract_attention_maps.py
├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/your-username/CTR-GCN-ASD-Analysis.git

cd CTR-GCN-ASD-Analysis
```

### Create Virtual Environment

```bash
python -m venv venv

venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train the proposed framework using:

```bash
python main.py --config config/mmasd/train.yaml
```

---

## Testing

Evaluate trained models using:

```bash
python main.py --config config/mmasd/train.yaml --phase test --weights ./work_dir/mmasd/ctrgcn_motion/runs-30-4440.pt
```

---

## Real-World Video Inference

The framework supports deployment on real-world videos using MediaPipe-based skeleton extraction.

Run inference using:

```bash
python video_inference.py
```

The deployment pipeline includes:

* MediaPipe skeleton extraction
* Skeleton normalization
* Sliding-window temporal inference
* Motion feature detection
* Behavioral summary generation

---

## Visualization and Analysis

The project supports multiple interpretability and analysis modules:

* Confusion matrix visualization
* Joint attention heatmaps
* Temporal activity dynamics
* Wrist-distance motion graphs
* t-SNE embedding visualization
* Behavioral summary tables

---

## Results

The proposed framework demonstrates strong performance on the MMASD dataset for ASD intervention activity recognition and behavioral analysis.

Key observations include:

* High classification accuracy across multiple intervention classes
* Strong embedding separability using supervised contrastive learning
* Interpretable joint attention focused on upper-body motion
* Real-world deployment capability using monocular RGB videos

---

## Acknowledgment

This work is built upon the official CTR-GCN implementation proposed by Chen et al. for skeleton-based action recognition.

Original repository:
https://github.com/Uason-Chen/CTR-GCN

We thank the original authors for their valuable contribution to the skeleton-based graph learning community.

---

## Citation

### Original CTR-GCN Paper

```bibtex
@inproceedings{chen2021channel,
  title={Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition},
  author={Chen, Yuxin and Zhang, Ziqi and Yuan, Chunfeng and Li, Bing and Deng, Ying and Hu, Weiming},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13359--13368},
  year={2021}
}
```

---

## Future Work

Future research directions include:

* Multimodal fusion with audio and facial behavior analysis
* Transformer-based graph architectures
* Larger clinically diverse ASD cohorts
* Real-time clinical deployment systems
* Explainable AI for therapeutic behavioral assessment.
