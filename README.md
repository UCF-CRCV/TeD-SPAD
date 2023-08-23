# TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection [ICCV 2023]
[Joseph Fioresi](joefioresi718.github.io), [Ishan Dave](joefioresi718.github.io), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://joefioresi718.github.io/TeD-SPAD_webpage/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.06947)

Official PyTorch implementation and pre-trained models for TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection.

> **Abstract:**
> Video anomaly detection (VAD) without human monitoring is a difficult computer vision task that can have a positive impact on society if implemented successfully. While recent advances have made significant progress in solving this task, most existing approaches overlook a critical real-world concern: privacy. With the increasing popularity of artificial intelligence technologies, it becomes crucial to implement proper AI ethics into their development. Privacy leakage in VAD allows models to pick up and amplify unnecessary biases related to people’s personal information, which may lead to undesirable decision making.
> In this paper, we propose TeD-SPAD, a privacy-aware video anomaly detection framework that destroys visual private information in a self-supervised manner. In particular, we explore the impact of temporally-distinctive video representations for VAD, finding that temporal distinctiveness pairs well with current anomaly feature representation learning methods. We achieve a positive trade-off between privacy protection and utility anomaly detection performance on three popular weakly supervised VAD datasets: UCF-Crime, XD-Violence, and ShanghaiTech. Our proposed anonymization model reduces private attribute prediction by 32.25% while only reducing frame-level ROC AUC on the UCF-Crime anomaly detection dataset by 3.69%.

## Usage

### Environment Setup
```bash
# Clone the repo.
git clone https://github.com/joefioresi718/TeD-SPAD.git && cd TeD-SPAD

# Pip install to existing environment.
pip install -r pip_requirements.txt

# Anaconda install options.
conda create --name ted_spad --file conda_requirements.txt
conda env create -f ted_spad.yml

# Update aux_code/config.py paths to match dataset directories.
```

### Anonymization Training
```bash
cd anonymization_training
python train_anonymizer.py
```

### Privacy Evaluation
```bash
cd privacy_training
python train_privacy.py
```

### Feature Extraction
```bash
cd feature_extraction
python dali_extraction.py
```

### Anomaly Detection Evaluation
Code taken from [MGFN](https://github.com/carolchenyx/MGFN) repo with minor changes to support our workflow.
```bash
cd anomaly_detection_mgfn
python main.py
```
