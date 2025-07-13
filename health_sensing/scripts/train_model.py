import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader
from models.cnn_model import CNN1D
from models.conv_lstm_model import ConvLSTMModel
import ast


class BreathingDataset(Dataset):
    def __init__(self, df):
        self.data = df
        self.labels = {'Normal': 0, 'Hypopnea': 1, 'Obstructive Apnea': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #parse string representations of lists back to actual lists
        nasal = ast.literal_eval(self.data.iloc[idx]['nasal']) if isinstance(self.data.iloc[idx]['nasal'], str) else self.data.iloc[idx]['nasal']
        thoracic = ast.literal_eval(self.data.iloc[idx]['thoracic']) if isinstance(self.data.iloc[idx]['thoracic'], str) else self.data.iloc[idx]['thoracic']
        spo2 = ast.literal_eval(self.data.iloc[idx]['spo2']) if isinstance(self.data.iloc[idx]['spo2'], str) else self.data.iloc[idx]['spo2']
        
        #pad SPO2 to match the length of other signals
        target_len = len(nasal)
        if len(spo2) < target_len:
            spo2 = spo2 + [spo2[-1]] * (target_len - len(spo2))
        
        x = torch.tensor([nasal, thoracic, spo2[:target_len]], dtype=torch.float)
        y = self.labels[self.data.iloc[idx]['label']]
        return x, y


def calculate_metrics(y_true, y_pred, class_names):
    """Calculate comprehensive metrics for each class"""
    results = {}
    
    #Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    results['accuracy'] = accuracy
    
    #Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    #calculate sensitivity and specificity for class
    cm = confusion_matrix(y_true, y_pred)
    
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results[f'{class_name}_precision'] = precision[i]
        results[f'{class_name}_recall'] = recall[i]
        results[f'{class_name}_sensitivity'] = sensitivity
        results[f'{class_name}_specificity'] = specificity
        results[f'{class_name}_f1'] = f1[i]
    
    results['confusion_matrix'] = cm
    return results


def train_and_evaluate(model_class, model_name):
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    df = pd.read_csv("Dataset/breathing_dataset.csv")
    participants = df['participant_id'].unique()
    class_names = ['Normal', 'Hypopnea', 'Obstructive Apnea']
    
    all_fold_results = []
    
    for fold, pid in enumerate(participants):
        print(f"\nFold {fold + 1}: Testing on {pid}")
        print("-" * 30)
        
        train_df = df[df['participant_id'] != pid]
        test_df = df[df['participant_id'] == pid]
        
        print(f"Training samples: {len(train_df)}, Test samples: {len(test_df)}")
        
        train_ds = BreathingDataset(train_df)
        test_ds = BreathingDataset(test_df)

        model = model_class()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        # Training
        model.train()
        for epoch in range(10):  # Increased epochs
            total_loss = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}")

        # Evaluation
        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb)
                all_preds.extend(torch.argmax(preds, dim=1).tolist())
                all_true.extend(yb.tolist())

        # Calculate metrics
        fold_results = calculate_metrics(all_true, all_preds, class_names)
        all_fold_results.append(fold_results)
        
        print(f"Fold {fold + 1} Results:")
        print(f"Accuracy: {fold_results['accuracy']:.4f}")
        print("\nPer-class metrics:")
        for class_name in class_names:
            print(f"{class_name}:")
            print(f"  Precision: {fold_results[f'{class_name}_precision']:.4f}")
            print(f"  Recall: {fold_results[f'{class_name}_recall']:.4f}")
            print(f"  Sensitivity: {fold_results[f'{class_name}_sensitivity']:.4f}")
            print(f"  Specificity: {fold_results[f'{class_name}_specificity']:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(fold_results['confusion_matrix'])
    
    # Aggregate results
    print(f"\n{'='*50}")
    print(f"AGGREGATED RESULTS FOR {model_name}")
    print(f"{'='*50}")
    
    # Calculate mean and std across folds
    accuracies = [result['accuracy'] for result in all_fold_results]
    print(f"Overall Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    
    print("\nPer-class Aggregated Metrics:")
    for class_name in class_names:
        precisions = [result[f'{class_name}_precision'] for result in all_fold_results]
        recalls = [result[f'{class_name}_recall'] for result in all_fold_results]
        sensitivities = [result[f'{class_name}_sensitivity'] for result in all_fold_results]
        specificities = [result[f'{class_name}_specificity'] for result in all_fold_results]
        
        print(f"\n{class_name}:")
        print(f"  Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"  Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"  Sensitivity: {np.mean(sensitivities):.4f} ± {np.std(sensitivities):.4f}")
        print(f"  Specificity: {np.mean(specificities):.4f} ± {np.std(specificities):.4f}")


if __name__ == '__main__':
    print("Training 1D CNN model")
    train_and_evaluate(CNN1D, "1D CNN")
    
    print("\n" + "="*80)
    
    print("Training Conv-LSTM model")
    train_and_evaluate(ConvLSTMModel, "Conv-LSTM")
