# 🐋 whale-spectrogram-dataset-tools

A PyTorch-based toolkit for loading, processing, visualizing, and evaluating Humpback Whale song spectrograms. This repository serves as supplementary material to the Humpback Whales Spectrogram Dataset.

## 📂 Project Structure

- `src/whales_dataset.py`: Custom PyTorch `Dataset` classes (`LineLevelDataset`, `PageLevelDataset`) that handle complex JSON annotations.
- `src/transforms.py`: Specialized operations on the data, including patches creation (`RandomSpectrogramLinePatcher`) and format converters (`UnitIntervalsToYOLO`).
- `src/evaluate.py`: Evaluation script to calculate Precision, Recall, and mAP for object detection.
- `src/visualization.py`: Script for visualizing bounding boxes/polygons.
- `src/utils.py`: General helper functions (e.g. `is_valid_file`) and parsers (e.g. `parse_line_level_data`).

## 🚀 Installation
### Clone the repository:
   ```bash
   git clone https://github.com/g-matidis/whale-spectrogram-dataset-tools.git
   cd whale-spectrogram-dataset-tools
   ```

### Set up this project using the modern `uv` manager (recommended) or traditional `pip`.
* ### Option A: Using `uv` (recommended).
1. Install `uv' from https://astral.sh/uv.
2. Install dependencies:
   ```bash 
   uv sync
   ```

* ### Option 2: Using `pip`.
1. Create and activate your virtual environment:
   ```bash
   python -m venv .your_env_name
   source .venv/bin/activate
   ```

## 📥 Download the Dataset
1. Download the Whales Spectrogram Dataset [here](https://iptademokritosgr-my.sharepoint.com/:f:/g/personal/gmatidis_iit_demokritos_gr/IgDgYsfVn2V7T5eZEJzxkUrOAXc8Enf_BOgRjALbIt9vM00?e=ruXzIP).
2. Copy the dataset directory in [./data](./data).

## Visualize the data
TO BE WRITTEN!!!
