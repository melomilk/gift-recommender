
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["She loves hiking and cozy evenings"])

print("Shape:", embeddings.shape)
print("First 5 numbers:", embeddings[0][:5])