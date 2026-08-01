# CoDPL：Counterfactual dual-view preference learning for session-based recommendation
Requirements
plaintext
python >= 3.8
pytorch >= 1.10
scikit-learn
numpy
tqdm
scipy
pickle
Overview
CoDPL models session preference from global preference and local intent perspectives. Counterfactual preference inference is adopted to enrich prediction results. We adopt K-Means clustering to accelerate similar session retrieval. The candidate pool consists of fixed historical training sessions and concurrent sessions within each mini-batch.

Data preprocessing follows SR-GNN standard pipeline.
Quick Start
bash
python main.py --dataset tmall

