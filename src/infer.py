import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FreuidDataset, fxn_get_transforms
from src.models import fxn_get_model

def main(kaggle_dir = '/kaggle/working/freuid-dataset'):
    config = {
        'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'img_size': 384,
        'batch_size': 256,  
        'num_workers': 8, 
        'n_splits': 5,
        'weights_dir': 'weights',
        'output_path': 'submission.csv'
    }

    if os.path.exists(kaggle_dir):
        print(f"Detected Kaggle environment. Using data from {kaggle_dir}")
        config['img_dir'] = kaggle_dir
        config['sample_sub_path'] = os.path.join(kaggle_dir, 'sample_submission.csv')
    else:
        config['img_dir'] = 'Data'
        config['sample_sub_path'] = 'Data/sample_submission.csv'

    # Use AI for sucessive print statements for tracking run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading test data from {config['sample_sub_path']}...")
    test_dataset = FreuidDataset(
        csv_file=config['sample_sub_path'],
        img_dir=config['img_dir'],
        transform=fxn_get_transforms(img_size=config['img_size'], is_train=False),
        is_test=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    print(f"Creating base model {config['model_name']}...")
    base_model = fxn_get_model(config['model_name'], pretrained=False)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel for Inference!")
        base_model = torch.nn.DataParallel(base_model)
    base_model.to(device)
    
    # Pre-allocate array for ensembled predictions
    all_preds = np.zeros(len(test_dataset))
    all_ids = []
    
    loaded_folds = 0
    for fold in range(config['n_splits']):
        weights_path = os.path.join(config['weights_dir'], f"best_model_fold_{fold}.pth")
        print(f"\n--- Loading Fold {fold} from {weights_path} ---")
        if not os.path.exists(weights_path):
            print(f"Warning: {weights_path} not found. Skipping fold {fold}.")
            continue
            
        loaded_folds += 1
        state_dict = torch.load(weights_path, map_location=device)
        if isinstance(base_model, torch.nn.DataParallel):
            base_model.module.load_state_dict(state_dict)
        else:
            base_model.load_state_dict(state_dict)
            
        base_model.eval()
        
        fold_preds = []
        fold_ids = []
        
        with torch.no_grad():
            for images, batch_ids in tqdm(test_loader, desc=f"Inference Fold {fold}"):
                images = images.to(device)
                with torch.amp.autocast('cuda'):
                    # Standard pass
                    outputs1 = base_model(images)
                    probs1 = torch.sigmoid(outputs1)
                    
                    # TTA pass (Horizontal Flip)
                    outputs2 = base_model(torch.flip(images, dims=[3]))
                    probs2 = torch.sigmoid(outputs2)
                    
                # Average the probabilities from both passes
                probs = ((probs1 + probs2) / 2.0).cpu().numpy().flatten()
                
                fold_preds.extend(probs)
                if loaded_folds == 1:
                    fold_ids.extend(batch_ids)
                    
        all_preds += np.array(fold_preds)
        if loaded_folds == 1:
            all_ids = fold_ids

    if loaded_folds > 0:
        all_preds = all_preds / loaded_folds
    else:
        print("Error: No models were loaded! Submission will be all zeros.")

    print("Generating submission.csv...")
    sub_df = pd.DataFrame({
        'id': all_ids,
        'label': all_preds
    })
    

    
    sub_df.to_csv(config['output_path'], index=False)
    print(f"Submission saved to {config['output_path']}")

if __name__ == '__main__':
    main()
