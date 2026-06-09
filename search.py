import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load data
with open("slides.json", "r", encoding="utf-8") as f:
    slides = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index("slides.index")

query = input("Ask a question: ")

# Convert query to embedding
query_embedding = model.encode([query])

# Search
distances, indices = index.search(
    np.array(query_embedding, dtype=np.float32),
    5
)

print("\nTop Results:\n")

for i in indices[0]:
    print("=" * 50)
    print("Slide:", slides[i]["slide"])
    print(slides[i]["text"][:500])
    print()