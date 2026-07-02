import zipfile
import os
import sys
import argparse
from pathlib import Path


def fxn_extract_kaggle(zip_path,extract_dir):
    if not os.path.exists(zip_path):
        print(f"Error Trig 1")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        all_files = zip_ref.namelist()
        total_files = len(all_files)
        print(f"File ctr: {total_files}")
        
        for idx, file in enumerate(all_files):
            zip_ref.extract(file, extract_dir)
    print("Success")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path",type = str, default = "test.zip")
    parser.add_argument("--extract_dir",type = str, default="test")
    args = parser.parse_args()
    zip_path = Path(args.zip_path)
    extract_dir = Path(args.extract_dir)
    fxn_extract_kaggle(zip_path,extract_dir)
