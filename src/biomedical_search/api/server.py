"""
API server module for the biomedical image search system.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import os
import tempfile
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_processing.database import ImageDatabase
from utils.image_utils import load_image, prepare_model

# Initialize database and model
db = ImageDatabase()
model, preprocess, device = prepare_model()

app = FastAPI(
    title="Biomedical Image Search API",
    description="API for searching similar biomedical images using multimodal models",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search")
async def search_similar_images(
    query_image: UploadFile = File(...),
    n_results: int = 5
) -> List[dict]:
    """
    Search for similar images using a query image.
    
    Args:
        query_image: The query image file
        n_results: Number of similar images to return
        
    Returns:
        List of similar images with their metadata
    """
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(query_image.filename)[1]) as temp_file:
            temp_file_path = temp_file.name
            content = await query_image.read()
            temp_file.write(content)
        
        # Load and convert the image to RGB mode
        image = load_image(temp_file_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Preprocess image for the model (resize, normalize, etc.)
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        
        # Extract Embeddings
        with torch.no_grad():
            features = model.encode_image(img_tensor, normalize=True) # Normalize the features
            # We could Normalize the features like this:
            # features = features / features.norm(dim=-1, keepdim=True)
        
        # Convert to numpy array
        query_embedding = features.cpu().numpy().squeeze()
        
        # Search for similar images
        similar_images = db.search_similar(query_embedding, n_results=n_results)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        return similar_images
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)