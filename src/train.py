import os
# Prevent thread oversubscription in OpenCV and Albumentations (Crucial for PyTorch DataLoaders)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(0)

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.dataset import FreuidDataset, fxn_get_transforms
from src.models import fxn_get_model
from src.metrics import fxn_compute_metrics


def fxn_train_one_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            running_loss += loss.item() * images.size(0)
            
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    
    # Use: Compute FREUID metrics
    metrics = fxn_compute_metrics(all_labels, all_preds)
    metrics['val_loss'] = val_loss
    return metrics

def train_fold(fold, train_idx, val_idx, df, img_dir, config):
    print(f"\n========== Training Fold No. {fold} ==========")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Use: Temporary CSV paths for dataset folds
    train_csv = f"Data/train_fold_{fold}.csv"
    val_csv = f"Data/val_fold_{fold}.csv"
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    train_dataset = FreuidDataset(
        csv_file=train_csv,
        img_dir=img_dir,
        transform=fxn_get_transforms(img_size=config['img_size'], is_train=True)
    )
    val_dataset = FreuidDataset(
        csv_file=val_csv,
        img_dir=img_dir,
        transform=fxn_get_transforms(img_size=config['img_size'], is_train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Model
    model = fxn_get_model(config['model_name'], pretrained=True)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_freuid = 1.0
    best_metrics = {}
    
    os.makedirs(config['save_dir'], exist_ok=True)
    model_save_path = os.path.join(config['save_dir'], f"best_model_fold_{fold}.pth")
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss = fxn_train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        scheduler.step()
        
        val_metrics = validate(model, val_loader, criterion, device)
        val_loss = val_metrics['val_loss']
        freuid_score = val_metrics['FREUID']
        
        #Using AI made print statements for tracking run performance
        print(f"Fold {fold} - Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  AuDET: {val_metrics['AuDET']:.4f} | APCER@1%BPCER: {val_metrics['APCER_at_1pct_BPCER']:.4f}")
        print(f"  FREUID Score: {freuid_score:.4f}")
        
        if freuid_score < best_freuid:
            best_freuid = freuid_score
            best_metrics = val_metrics
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, model_save_path)
            print(f"  --> Saved new best model with FREUID: {best_freuid:.4f} to {model_save_path}")
            
    try:
        os.remove(train_csv)
        os.remove(val_csv)
    except:
        pass
        
    return best_metrics

def main(kaggle_dir = '/kaggle/working/freuid-dataset'):
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    config = {
        'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'img_size': 384,
        'batch_size': 64,  
        'lr': 1e-4,  # Lowered learning rate for stability
        'weight_decay': 1e-4,
        'epochs': 5,        
        'num_workers': 1,  
        'n_splits': 5,
        'save_dir': 'weights',
    }
    
    if os.path.exists(kaggle_dir):
        print(f"Detected Kaggle environment. Using data from {kaggle_dir}")
        img_dir = kaggle_dir
        train_csv_path = os.path.join(kaggle_dir, 'train_labels.csv')
    else:
        print("Using local data directory 'Data/'")
        img_dir = 'Data'
        train_csv_path = 'Data/train_labels.csv'
    
    df = pd.read_csv(train_csv_path)
    print(f"Loaded train metadata: {len(df)} samples")
    
    # Use: Creates a stratify column 
    df['stratify_col'] = df['type'] + "_" + df['label'].astype(str)
    
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=42)
    
    # Note: Just trains on fold 0 first for quick verification of the whole pipeline
    fold_metrics = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['stratify_col'])):
        # Train all 5 folds
        metrics = train_fold(fold, train_idx, val_idx, df, img_dir, config)
        fold_metrics.append(metrics)
        
    print("\nTraining completed.")
    print("Fold 0 Metrics:")
    for k, v in fold_metrics[0].items():
        print(f"  {k}: {v:.6f}")

if __name__ == '__main__':
    main()
