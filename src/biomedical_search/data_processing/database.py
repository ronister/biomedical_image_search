"""
Database module for managing image embeddings and similarity search using ChromaDB.
"""

import chromadb
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import torch
from ..utils.image_utils import load_image, prepare_model

class ImageDatabase:
    def __init__(self, persist_directory: str = "data/embeddings"):
        """
        Initialize the image database with ChromaDB.
        
        Args:
            persist_directory: Directory to persist the database
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="biomedical_images",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize the model
        self.model, self.preprocess, self.device = prepare_model()
        
    def add_images(self, embeddings: np.ndarray, image_paths: List[str], 
                  metadata: List[Dict[str, Any]] = None):
        """
        Add image embeddings to the database.
        
        Args:
            embeddings: Feature vectors for the images
            image_paths: List of image file paths
            metadata: Optional metadata for each image
        """
        if metadata is None:
            metadata = [{"file_path": path} for path in image_paths]
            
        # Convert embeddings to list format for ChromaDB
        embeddings_list = embeddings.tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings_list,
            documents=image_paths,  # Using paths as document IDs
            metadatas=metadata,
            ids=[f"img_{i}" for i in range(len(image_paths))]
        )
        
    def search_similar(self, query_embedding: np.ndarray, n_results: int = 5) -> List[Dict]:
        """
        Search for similar images using a query embedding.
        
        Args:
            query_embedding: Feature vector of the query image
            n_results: Number of similar images to return
            
        Returns:
            List of dictionaries containing similar images and their metadata
        """
        # Convert query embedding to list format
        query_embedding_list = query_embedding.tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding_list],
            n_results=n_results
        )
        
        # Format results
        similar_images = []
        for i in range(len(results['documents'][0])):
            similar_images.append({
                'path': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
            
        return similar_images
        
    def process_and_add_directory(self, directory_path: str, batch_size: int = 16):
        """ 
        Process all images in a directory and add them to the vector database C
        
        Args:
            directory_path: Path to directory containing images
            batch_size: Batch size for processing images
        """
        from ..utils.image_utils import find_cell_images, get_image_metadata
        
        # Find all valid images
        image_paths = find_cell_images(directory_path, recursive=False)
        if not image_paths:
            print(f"No valid images found in {directory_path}")
            return
            
        # Process images in batches
        all_embeddings = []
        all_metadata = []
        valid_paths = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing images"):
            batch_paths = image_paths[i:i+batch_size]
            batch_images = []
            batch_valid_indices = []
            
            # Load and preprocess each image
            for j, path in enumerate(batch_paths):
                image = load_image(path)
                if image is not None:
                    batch_images.append(self.preprocess(image))
                    batch_valid_indices.append(j)
                    valid_paths.append(path)
                    all_metadata.append(get_image_metadata(path))
            
            if not batch_images:
                continue
                
            # Stack batch and move to device
            batch_tensors = torch.stack(batch_images).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensors)
                # Normalize the features
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
            
            all_embeddings.append(batch_features.cpu().numpy())
        
        if not all_embeddings:
            print("No valid embeddings were generated")
            return
            
        # Combine all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        # Add to database
        self.add_images(all_embeddings, valid_paths, all_metadata)
        print(f"Added {len(valid_paths)} images to the ChromaDB vector database")