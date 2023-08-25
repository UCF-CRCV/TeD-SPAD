# TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection [ICCV 2023]
[Joseph Fioresi](https://joefioresi718.github.io/), [Ishan Dave](https://scholar.google.com/citations?hl=en&user=fWu6sFgAAAAJ), [Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en&oi=ao)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://joefioresi718.github.io/TeD-SPAD_webpage/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.11072)

Official PyTorch implementation and pre-trained models for TeD-SPAD: Temporal Distinctiveness for Self-supervised Privacy-preservation for video Anomaly Detection.

> **Abstract:**
> Video anomaly detection (VAD) without human monitoring is a complex computer vision task that can have a positive impact on society if implemented successfully. While recent advances have made significant progress in solving this task, most existing approaches overlook a critical real-world concern: privacy. With the increasing popularity of artificial intelligence technologies, it becomes crucial to implement proper AI ethics into their development. Privacy leakage in VAD allows models to pick up and amplify unnecessary biases related to people’s personal information, which may lead to undesirable decision making.
> In this paper, we propose TeD-SPAD, a privacy-aware video anomaly detection framework that destroys visual private information in a self-supervised manner. In particular, we propose the use of a temporally-distinct triplet loss to promote temporally discriminative features, which complements current weakly-supervised VAD methods. Using TeD-SPAD, we achieve a positive trade-off between privacy protection and utility anomaly detection performance on three popular weakly supervised VAD datasets: UCF-Crime, XD-Violence, and ShanghaiTech. Our proposed anonymization model reduces private attribute prediction by 32.25% while only reducing frame-level ROC AUC on the UCF-Crime anomaly detection dataset by 3.69%.

## Usage

### Dataset Setup
[UCF-Crime Page](https://www.crcv.ucf.edu/projects/real-world/)

[VISPR Page](https://tribhuvanesh.github.io/vpa/)
```bash
# Download necessary datasets.
# UCF101
mkdir datasets && cd datasets
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
rm -rf UCF101.rar
wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip --no-check-certificate
unzip UCF101TrainTestSplits-RecognitionTask.zip
rm -rf UCF101TrainTestSplits-RecognitionTask.zip

cd aux_code
# Edit config.py to point paths to correct directory.
nano config.py
```
<sub> ** Move *action_classes.json* into UCF101/ucfTrainTestlist/ </sub>


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

Extracted features/model weights: [OneDrive](https://1drv.ms/f/s!Ah-hee3NbVf7ge97m4_amfmsrmHKig?e=etleeR)

### Usage Instructions
1. Navigate to desired directory
2. Modify parameters python file if necessary
3. Run the main python file

Example:
```bash
# Anonymization Training.
cd anonymization_training
python train_anonymizer.py

# Privacy Evaluation.
cd privacy_training
python train_privacy.py

# Action Recognition Training.
cd action_training
python train_action.py

# Feature Extraction.
cd feature_extraction
# ShanghaiTech
python st_feature_extraction.py
# UCF_Crime/XD-Violence
python dali_extraction.py

# Anonymization Visualization.
cd visualization
python visualize_anonymization.py
```


### Anomaly Detection Evaluation
Code taken from [MGFN](https://github.com/carolchenyx/MGFN) repo with minor changes to support our workflow.
- Extract/download features, place in 'anomaly_detection_mgfn/data/'
```bash
cd anomaly_detection_mgfn
python main.py
```
