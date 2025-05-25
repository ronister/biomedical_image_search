"""
Utility functions for image processing and handling.
"""

import numpy as np
from typing import List, Union
from pathlib import Path
import os
from PIL import Image
from urllib.request import urlopen
from io import BytesIO
import torch
from open_clip import create_model_from_pretrained

def prepare_model():
    """Load the BiomedCLIP model and prepare it for inference."""
    model, preprocess = create_model_from_pretrained(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, preprocess, device

def load_image(image_path: Union[str, Path]) -> Image:
    """
    Load and preprocess an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL.Image: image in RGB mode
    """
    try:
        # Check if path is a URL
        if str(image_path).startswith(('http://', 'https://')):
            with urlopen(str(image_path)) as response:
                image = Image.open(BytesIO(response.read()))
        else:
            # Load from local file
            image = Image.open(str(image_path))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        print(f"Error loading image from {image_path}: {str(e)}")
        return None

def validate_image(image_path: Union[str, Path]) -> bool:
    """
    Validate if a file is a valid image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        bool: True if valid image, False otherwise
    """
    try:
        # Skip background images
        if "Background" in str(image_path):
            return False
            
        # Check if file exists
        if not os.path.exists(str(image_path)):
            return False
            
        # Try to open the image
        img = Image.open(str(image_path))
        img.verify()  # Verify it's a valid image
        return True
    except Exception:
        return False

def get_image_metadata(image_path: Union[str, Path]) -> dict:
    """
    Extract metadata from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Image metadata
    """
    path = str(image_path)
    metadata = {
        "file_path": path,
        "dataset": "unknown"
    }
    
    # Determine dataset from path
    if "dev" in path:
        metadata["dataset"] = "dev"
    elif "EVICAN_train2019" in path:
        metadata["dataset"] = "train"
    elif "EVICAN_val2019" in path:
        metadata["dataset"] = "val"
    elif "EVICAN_eval2019" in path:
        metadata["dataset"] = "test"
        
    return metadata

def find_cell_images(directory_path: Union[str, Path], recursive: bool = False) -> List[str]:
    """Find all cell image files in a directory."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    image_paths = []
    
    if recursive:
        for root, _, files in os.walk(str(directory_path)):
            for file in files:
                if os.path.splitext(file.lower())[1] in image_extensions:
                    full_path = os.path.join(root, file)
                    if validate_image(full_path):  # Skip invalid images
                        image_paths.append(full_path)
    else:
        image_paths = [
            os.path.join(str(directory_path), f) for f in os.listdir(str(directory_path))
            if os.path.isfile(os.path.join(str(directory_path), f)) and
            os.path.splitext(f.lower())[1] in image_extensions and
            validate_image(os.path.join(str(directory_path), f))
        ]
    
    return sorted(image_paths)