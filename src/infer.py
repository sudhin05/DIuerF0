import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import FreuidDataset, fxn_get_transforms
from src.models import fxn_get_model

def main():
    config = {
        'model_name': 'tf_efficientnetv2_s.in21k_ft_in1k',
        'img_size': 384,
        'batch_size': 64,  
        'num_workers': 0, 
        'weights_path': 'weights/best_model_fold_0.pth',
        'sample_sub_path': 'Data/sample_submission.csv',
        'img_dir': 'Data',
        'output_path': 'submission.csv'
    }

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

    print(f"Loading model {config['model_name']}...")
    model = fxn_get_model(config['model_name'], pretrained=False)
    
    if not os.path.exists(config['weights_path']):
        raise FileNotFoundError(f"Could not find weights at {config['weights_path']}")
        
    model.load_state_dict(torch.load(config['weights_path'], map_location=device))
    model.to(device)
    model.eval()

    print("Starting inference...")
    ids = []
    preds = []
    
    with torch.no_grad():
        for images, batch_ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            
            ids.extend(batch_ids)
            preds.extend(probs)

    print("Generating submission.csv...")
    sub_df = pd.DataFrame({
        'id': ids,
        'label': preds
    })
    

    
    sub_df.to_csv(config['output_path'], index=False)
    print(f"Submission saved to {config['output_path']}")

if __name__ == '__main__':
    main()
