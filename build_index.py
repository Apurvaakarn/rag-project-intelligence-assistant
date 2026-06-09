import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load slides
with open("slides.json", "r", encoding="utf-8") as f:
    slides = json.load(f)

texts = [slide["text"] for slide in slides]

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(texts)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings, dtype=np.float32))

faiss.write_index(index, "slides.index")

print("Index created successfully!")
print("Total documents indexed:", len(texts))