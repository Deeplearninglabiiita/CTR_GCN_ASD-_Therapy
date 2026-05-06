# ASD Behavioral Analysis using CTR-GCN

## Overview
This project implements a multi-task CTR-GCN framework for:
- Activity recognition
- ASD severity estimation
- RRB prediction

## Dataset
- MMASD dataset (3D skeletons)

## Features
- Temporal + Joint Attention
- Supervised Contrastive Learning
- Real-time inference using MediaPipe

## Results
- Accuracy: ~92%
- Interpretable attention maps
- Real-world deployment pipeline

## How to Run
```bash
python main.py --config config/mmasd/train.yaml