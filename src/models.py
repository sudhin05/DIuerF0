import torch
import torch.nn as nn
import timm
import argparse

def fxn_get_model(model_name='convnext_tiny', pretrained=True, num_classes=1):
    print(f"Creating model: {model_name} (pretrained={pretrained})")
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes
    )
    return model

if __name__ == '__main__':
    # Test model creation
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",type=str,default="efficientnet_b0")
    args = parser.parse_args()
    model = fxn_get_model(args.model_name, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("Output shape:", out.shape)
