# CoDPL: Counterfactual cross-session preference inference for session-based recommendation

## Requirements

python >= 3.8
pytorch >= 1.10
scikit-learn
numpy
tqdm
scipy
pickle


## Overview
CoDPL models session preference from global preference and local intent perspectives. Counterfactual preference inference is adopted to enrich prediction results. We adopt K-Means clustering to accelerate similar session retrieval. 

Datasets follow the standard SR-GNN preprocessing pipeline.

## Quick Start
```bash
python main.py --dataset tmall
