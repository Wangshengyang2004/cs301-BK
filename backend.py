from fastapi import FastAPI, File, UploadFile
import numpy as np
import uvicorn

from feature_extract import FeatureExtract
import faiss  # Ensure FAISS is properly installed and imported

app = FastAPI()
feature_extractor = FeatureExtract()
index = faiss.IndexFlatL2(512)  # Assuming the output feature dimension is 512

@app.post("/extract-features")
async def extract_features(file: UploadFile = File(...)):
    image_data = await file.read()
    features = feature_extractor.extract_features(image_data)
    return {"features": features.tolist()}

@app.post("/index-image")
async def index_image(features: list):
    feature_np = np.array(features, dtype='float32').reshape(1, -1)
    index.add(feature_np)
    return {"message": "Image indexed"}

@app.post("/search-similar")
async def search_similar(features: list):
    feature_np = np.array(features, dtype='float32').reshape(1, -1)
    distances, indices = index.search(feature_np, 5)  # Top 5 similar items
    return {"distances": distances.tolist(), "indices": indices.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
