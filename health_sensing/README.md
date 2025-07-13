# DeepMedico™ Health Sensing Project

A comprehensive machine learning pipeline for detecting breathing irregularities during sleep using physiological signals.

## 📋 Project Overview

This project implements a complete pipeline for analyzing overnight sleep data to detect breathing irregularities (apnea, hypopnea) using:
- **Nasal Airflow** (32 Hz)
- **Thoracic Movement** (32 Hz) 
- **SpO₂ (Oxygen Saturation)** (4 Hz)

## 🚀 Quick Start

### Prerequisites
- Python 3.x
- Virtual environment already configured in `tfenv/`

### Run the Complete Pipeline
```bash
# Run everything (recommended for first time)
python run_project.py

# Or run specific steps:
python run_project.py --skip-vis              # Skip visualizations
python run_project.py --skip-dataset          # Skip dataset creation
python run_project.py --skip-training         # Skip model training
```

### Manual Execution

If you prefer to run each step manually:

```bash
# 1. Generate visualizations (for each participant)
python scripts/vis.py -name "Data/AP01"
python scripts/vis.py -name "Data/AP02"
python scripts/vis.py -name "Data/AP03"
python scripts/vis.py -name "Data/AP04"
python scripts/vis.py -name "Data/AP05"

# 2. Create dataset
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"

# 3. Train and evaluate models
python scripts/train_model.py
```

## 📁 Project Structure

```
health_sensing/
├── Data/                          # Raw participant data
│   ├── AP01/, AP02/, AP03/, AP04/, AP05/
│   │   ├── Flow - *.txt          # Nasal airflow signals
│   │   ├── Thorac - *.txt        # Thoracic movement signals
│   │   ├── SPO2 - *.txt          # Oxygen saturation signals
│   │   ├── Flow Events - *.txt   # Breathing event annotations
│   │   └── Sleep profile - *.txt # Sleep stage annotations
├── Visualizations/                # Generated PDF visualizations
├── Dataset/                       # Processed dataset
├── models/                        # Neural network architectures
│   ├── cnn_model.py              # 1D CNN implementation
│   └── conv_lstm_model.py        # Conv-LSTM implementation
├── scripts/                       # Data processing and training scripts
│   ├── vis.py                    # Visualization generation
│   ├── create_dataset.py         # Dataset creation with filtering
│   └── train_model.py            # Model training and evaluation
├── tfenv/                         # Python virtual environment
├── requirements.txt               # Package dependencies
└── run_project.py                # Complete pipeline runner
```

## 🔬 Technical Implementation

### 1. Visualization (`vis.py`)
- **✅ REQUIREMENT**: Plot all three physiological signals over 8-hour duration
- **✅ REQUIREMENT**: Overlay breathing event annotations
- **✅ REQUIREMENT**: Export as PDF format
- **✅ REQUIREMENT**: Handle different sampling rates using timestamps

**Features:**
- Time-synchronized plotting of multi-rate signals
- Event overlay with visual highlighting
- 5-minute windowed visualization for readability
- Automatic file pattern detection for different naming conventions

### 2. Data Cleaning & Dataset Creation (`create_dataset.py`)
- **✅ REQUIREMENT**: Apply bandpass filtering (0.17-0.4 Hz) for breathing frequency range
- **✅ REQUIREMENT**: Create 30-second windows with 50% overlap
- **✅ REQUIREMENT**: Label windows based on >50% overlap with events
- **✅ REQUIREMENT**: Focus on Hypopnea, Obstructive Apnea, and Normal labels

**Features:**
- Butterworth bandpass filter for noise reduction
- Robust file pattern matching across different participants
- Time-based labeling with overlap calculation
- CSV export for dataset portability

### 3. Machine Learning Models

#### 1D CNN (`cnn_model.py`)
- **✅ REQUIREMENT**: 1D Convolutional Neural Network architecture
- Multiple conv layers with pooling
- Fully connected classification head

#### Conv-LSTM (`conv_lstm_model.py`) 
- **✅ REQUIREMENT**: Combined 1D Conv + LSTM architecture
- Convolutional feature extraction
- LSTM temporal modeling
- Dropout for regularization

### 4. Evaluation (`train_model.py`)
- **✅ REQUIREMENT**: Leave-One-Participant-Out Cross-Validation
- **✅ REQUIREMENT**: Comprehensive metrics per class:
  - Accuracy, Precision, Recall
  - Sensitivity, Specificity
  - Confusion Matrix
- **✅ REQUIREMENT**: Aggregated results with mean ± std across folds

**Why Leave-One-Participant-Out?**
Prevents data leakage by ensuring no participant's data appears in both training and testing sets, which is crucial for physiological data where individual characteristics could bias results.

## 📊 Expected Output

### Visualizations
- `Visualizations/AP01_visualization.pdf` through `AP05_visualization.pdf`
- Multi-page PDFs showing 5-minute segments with overlaid breathing events

### Dataset
- `Dataset/breathing_dataset.csv`: Processed 30-second windows with features and labels

### Model Results
Console output showing:
- Per-fold results for each participant
- Aggregated metrics across all folds
- Confusion matrices for each model

## 🎯 Assignment Requirements Checklist

### Understanding the Data and Visualization [3 Marks] ✅
- [x] Plot Nasal Airflow, Thoracic Movement, SpO₂ over 8-hour duration
- [x] Overlay annotated flow events on signal plots
- [x] Export visualizations in PDF format
- [x] Handle different sampling rates using timestamps
- [x] Script accepts folder path as input: `python vis.py -name "Data/AP20"`

### Data Cleaning [4 Marks] ✅
- [x] Apply digital filtering for high-frequency noise removal
- [x] Focus on breathing frequency range (0.17 Hz to 0.4 Hz)
- [x] Use appropriate filtering libraries (SciPy)

### Dataset Creation [8 Marks] ✅
- [x] Split signals into 30-second windows with 50% overlap
- [x] Label windows based on >50% overlap with events
- [x] Focus on Hypopnea, Obstructive Apnea, Normal labels
- [x] Script usage: `python create_dataset.py -in_dir "Data" -out_dir "Dataset"`
- [x] Thoughtful file format choice (CSV for portability and inspection)

### Modeling [10 Marks] ✅
- [x] Implement 1D CNN architecture
- [x] Implement Conv-LSTM architecture
- [x] Use Leave-One-Participant-Out Cross-Validation
- [x] Report comprehensive metrics per class:
  - [x] Accuracy, Precision, Recall
  - [x] Sensitivity, Specificity  
  - [x] Confusion Matrix
- [x] Compute mean and standard deviation across folds
- [x] Explain why subject-wise validation is preferred over random split

## 🏆 Bonus Opportunity: Sleep Stage Classification [5 Marks]

The project structure supports extending to sleep stage classification using the sleep profile files. The framework can be adapted to:
- Replace breathing event labels with sleep stage labels (Wake, REM, N1, N2, N3)
- Use the same 30-second window framework
- Apply the same model architectures

## 🛠️ Troubleshooting

### Common Issues:
1. **File not found errors**: The script automatically handles different file naming patterns across participants
2. **Memory issues**: Reduce batch size in `train_model.py` if needed
3. **Missing packages**: All required packages should be installed in the virtual environment

### Data Format Notes:
- All signals include timestamps for proper alignment
- Different participants may have slightly different file naming conventions
- Event files contain start time, duration, and event type
- Sleep profile files contain 30-second sleep stage annotations

## 📈 Performance Notes

The models are designed for proof-of-concept demonstration. For production use, consider:
- Hyperparameter tuning
- More sophisticated architectures
- Data augmentation techniques
- Cross-participant normalization

---

**Author**: Data Science Team, DeepMedico™  
**Date**: Health Sensing Project Implementation
