"""
Script to process all images from the EVICAN datasets and store them in the database.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.database import ImageDatabase

def main():
    # Initialize database
    db = ImageDatabase()
    
    # Process each dataset
    datasets = {
        #"dev": "data/raw/dev",
        "train": "data/raw/EVICAN_train2019",
        "val": "data/raw/EVICAN_val2019",
        "test": "data/raw/EVICAN_eval2019"
    }
    
    for dataset_name, dataset_path in datasets.items():
        print(f"\nProcessing {dataset_name} dataset...")
        if os.path.exists(dataset_path):
            db.process_and_add_directory(dataset_path)
        else:
            print(f"Warning: Dataset path {dataset_path} does not exist")
            
    print("\nFinished processing all datasets!")

if __name__ == "__main__":
    main() 