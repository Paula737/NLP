"""
ingest_humaneval.py
Loads the OpenAI HumanEval dataset and stores it into a persistent
Chroma vector database, to be used later by the RAG retrieval module.

Run this ONCE to build your knowledge base:
    python ingest_humaneval.py
"""

import chromadb
from chromadb.utils import embedding_functions
from datasets import load_dataset

# ---------------------------------------------------------
# 1. Config
# ---------------------------------------------------------
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "coding_knowledge"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # local, free, no API key needed

# ---------------------------------------------------------
# 2. Load the HumanEval dataset
# ---------------------------------------------------------
print("Loading HumanEval dataset from Hugging Face...")
dataset = load_dataset("openai/openai_humaneval", split="test")
print(f"Loaded {len(dataset)} problems.")

# ---------------------------------------------------------
# 3. Set up Chroma persistent client + embedding function
# ---------------------------------------------------------
client = chromadb.PersistentClient(path=CHROMA_PATH)

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
    metadata={"description": "HumanEval programming problems + solutions"}
)

# ---------------------------------------------------------
# 4. Build documents, metadata, ids
# ---------------------------------------------------------
documents = []
metadatas = []
ids = []

for row in dataset:
    task_id = row["task_id"]              # e.g. "HumanEval/0"
    prompt = row["prompt"]                 # function signature + docstring
    solution = row["canonical_solution"]   # reference implementation
    entry_point = row["entry_point"]       # function name
    test_code = row["test"]                

    doc_text = (
        f"Problem:\n{prompt}\n\n"
        f"Reference Solution:\n{prompt}{solution}"
    )

    documents.append(doc_text)
    metadatas.append({
        "task_id": task_id,
        "entry_point": entry_point,
        "source": "openai_humaneval",
        "test": test_code                    
    })
    ids.append(task_id.replace("/", "_"))

# ---------------------------------------------------------
# 5. Insert into Chroma (batched)
# ---------------------------------------------------------
BATCH_SIZE = 50
for i in range(0, len(documents), BATCH_SIZE):
    collection.add(
        documents=documents[i:i + BATCH_SIZE],
        metadatas=metadatas[i:i + BATCH_SIZE],
        ids=ids[i:i + BATCH_SIZE],
    )
    print(f"Inserted {min(i + BATCH_SIZE, len(documents))}/{len(documents)}")

print(f"\n✅ Done. Collection '{COLLECTION_NAME}' now has {collection.count()} documents.")