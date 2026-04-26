# Lab 3: Vector Stores

## Introduction

This lab covers vector stores, the third critical stage of the RAG pipeline. After splitting text into chunks, you must convert those chunks into vector embeddings and store them in a database optimized for similarity search. Vector stores enable fast, semantic retrieval of relevant documents based on query similarity.

This lab teaches you to work with embedding models, create and query vector stores, and perform filtered searches using metadata.

**Prerequisites:** Completion of Lab 1 (Document Loaders) and Lab 2 (Text Splitting), or equivalent knowledge of loading and splitting documents.

## Learning Objectives

By the end of this lab, you will be able to:

1. Explain what vector embeddings are and how they enable semantic search
2. Initialize embedding models from various providers (Ollama, OpenAI)
3. Create vector stores using Chroma and FAISS
4. Add, update, and delete documents in vector stores
5. Perform similarity search with and without scores
6. Implement metadata filtering for precise retrieval
7. Persist and reload vector stores for production use

## Prologue: The Challenge

Your document loading and text splitting pipelines are working well. You have thousands of document chunks ready, but you need a way to search them efficiently. Traditional keyword search has limitations:

1. **Synonym mismatch**: A query for "heart disease" misses documents mentioning "cardiac illness"
2. **No semantic understanding**: The system cannot understand that "doctor" and "physician" are related
3. **Scalability issues**: Searching through thousands of chunks becomes slow without indexing

The research team needs sub-second response times for their medical Q&A system while understanding complex medical terminology. You need to implement vector-based semantic search that understands meaning, not just keywords.

Success criteria:
- Convert text chunks into vector embeddings
- Store embeddings in a searchable vector database
- Retrieve relevant documents using similarity search
- Filter search results using metadata
- Persist the vector store for production use

## Environment Setup

Install required packages:

```bash
pip install langchain-chroma langchain-community langchain-ollama faiss-cpu ollama
```

Start Ollama and pull models:

```bash
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## Chapter 1: Understanding Vector Embeddings

Before implementing vector stores, you must understand what embeddings are and why they enable semantic search.

### 1.1 What You Will Build

Vector embeddings transform text into numerical vectors (lists of numbers) that capture semantic meaning. Documents with similar meaning produce similar vectors, enabling semantic similarity search.

### 1.2 Think First: How Does Semantic Search Work?

**Question:** If I search for "programming language", why does a document about "Python development" appear in the results, even though it doesn't contain the exact words "programming language"?

<details>
<summary>Click to review</summary>

Vector embeddings capture semantic meaning:

1. **Training on relationships**: Embedding models learn from massive text corpora that "Python" relates to "programming", "language", "development", etc.

2. **Spatial proximity**: Semantically similar concepts end up close together in the high-dimensional vector space.

3. **Cosine similarity**: When you search, your query becomes a vector. The system finds the nearest vectors (highest cosine similarity) to your query vector.

4. **No keyword matching**: This is fundamentally different from keyword search. The system understands meaning, not just exact matches.

</details>

### 1.3 Implementation: Creating Embeddings

Create `embedding_demo.py`:

```python
# embedding_demo.py
from langchain_ollama import OllamaEmbeddings

# Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create embeddings for different texts
texts = [
    "Machine learning is a subset of artificial intelligence",
    "Python is a programming language used for data science",
    "Deep learning uses neural networks with multiple layers",
    "The weather today is sunny and warm"
]

# Generate embeddings
embedded = embeddings.embed_documents(texts)

print(f"Number of texts: {len(texts)}")
print(f"Embedding dimension: {len(embedded[0])}")
print(f"\nFirst embedding (first 10 values):")
print(embedded[0][:10])

# Embed a query
query = "What is artificial intelligence?"
query_embedding = embeddings.embed_query(query)

print(f"\nQuery embedding dimension: {len(query_embedding)}")
print(f"Query: '{query}'")
```

### 1.4 Implementation: Calculating Similarity

Create `similarity_demo.py`:

```python
# similarity_demo.py
import numpy as np
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Related and unrelated texts
texts = [
    "The heart pumps blood throughout the body",  # Related to query
    "Python is a popular programming language",    # Unrelated
    "Cardiovascular health is important for longevity",  # Related
    "I love eating pizza for dinner",              # Unrelated
    "Heart disease is a leading cause of death globally"  # Related
]

query = "Tell me about heart health and cardiac conditions"

# Generate embeddings
text_embeddings = embeddings.embed_documents(texts)
query_embedding = embeddings.embed_query(query)

# Calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

print("Similarity scores with query: 'Tell me about heart health and cardiac conditions'\n")

similarities = []
for i, text in enumerate(texts):
    sim = cosine_similarity(query_embedding, text_embeddings[i])
    similarities.append((text, sim))
    print(f"  {sim:.4f} - {text}")

# Sort by similarity
print("\nRanked by relevance:")
for text, sim in sorted(similarities, key=lambda x: x[1], reverse=True):
    print(f"  {sim:.4f} - {text}")
```

**Observation:** Texts about heart health have higher similarity scores to the heart-related query, even though they use different words.

### 1.5 Checkpoint

**Self-Assessment:**
- [ ] You can create embeddings using OllamaEmbeddings
- [ ] You can embed both documents and queries
- [ ] You understand how cosine similarity measures semantic relevance
- [ ] You can explain why embeddings enable semantic search

---

## Chapter 2: Creating and Populating Vector Stores

Now that you understand embeddings, learn to store them in vector databases for efficient search.

### 2.1 What You Will Build

Vector stores persist embeddings and provide fast similarity search. This chapter covers creating stores and adding documents.

### 2.2 Implementation: Creating a Chroma Vector Store

Create `chroma_store.py`:

```python
# chroma_store.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pathlib import Path

# Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Sample documents about IPL cricket
documents = [
    Document(
        page_content="Virat Kohli is one of the most successful batsmen in IPL history. Known for his aggressive batting and fitness, he has led Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore", "role": "batsman"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings.",
        metadata={"team": "Mumbai Indians", "role": "batsman"}
    ),
    Document(
        page_content="MS Dhoni, known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills and wicketkeeping are legendary.",
        metadata={"team": "Chennai Super Kings", "role": "wicket-keeper"}
    ),
    Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians", "role": "bowler"}
    ),
    Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his fielding is exceptional.",
        metadata={"team": "Chennai Super Kings", "role": "all-rounder"}
    ),
]

# Create Chroma vector store (in memory)
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="ipl_players"
)

# Add documents to the store
vector_store.add_documents(documents)

print(f"Added {len(documents)} documents to the vector store")
print(f"Collection count: {vector_store._collection.count()}")
```

### 2.3 Implementation: Creating a FAISS Vector Store

Create `faiss_store.py`:

```python
# faiss_store.py
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Initialize embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Sample documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# Create FAISS vector store
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

print(f"Created FAISS index with {len(documents)} documents")

# Save the index to disk
vector_store.save_local("faiss_index")
print("Saved FAISS index to 'faiss_index' directory")

# Reload the index
loaded_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print(f"Loaded FAISS index with {loaded_store.index.ntotal} documents")
```

### 2.4 Checkpoint

**Self-Assessment:**
- [ ] You can create a Chroma vector store
- [ ] You can add documents with metadata to a vector store
- [ ] You can create a FAISS vector store
- [ ] You can persist and reload a FAISS vector store
- [ ] You understand when to use Chroma vs FAISS

---

## Chapter 3: Similarity Search

Now learn to query your vector store to retrieve relevant documents.

### 3.1 What You Will Build

Vector stores provide multiple search methods:
- Basic similarity search
- Search with similarity scores
- Maximum Marginal Relevance (MMR) for diverse results

### 3.2 Implementation: Basic Similarity Search

Create `similarity_search.py`:

```python
# similarity_search.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Setup (same as before)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document(
        page_content="Virat Kohli is one of the most successful batsmen in IPL history.",
        metadata={"team": "Royal Challengers Bangalore", "role": "batsman"}
    ),
    Document(
        page_content="Rohit Sharma is the most successful captain in IPL history.",
        metadata={"team": "Mumbai Indians", "role": "batsman"}
    ),
    Document(
        page_content="MS Dhoni is known for his finishing skills and wicketkeeping.",
        metadata={"team": "Chennai Super Kings", "role": "wicket-keeper"}
    ),
    Document(
        page_content="Jasprit Bumrah is known for his yorkers and death-over bowling.",
        metadata={"team": "Mumbai Indians", "role": "bowler"}
    ),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="ipl_search"
)

# Perform similarity search
query = "Who is the best bowler in T20 cricket?"

results = vector_store.similarity_search(query, k=2)

print(f"Query: '{query}'\n")
print("Top 2 results:")
for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

### 3.3 Implementation: Search with Similarity Scores

Create `similarity_with_scores.py`:

```python
# similarity_with_scores.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Setup (same documents)
documents = [
    Document(page_content="Python is a programming language for data science."),
    Document(page_content="Machine learning is a subset of artificial intelligence."),
    Document(page_content="The sun rises in the east every morning."),
    Document(page_content="Deep learning uses neural networks with multiple layers."),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="scored_search"
)

# Search with similarity scores
query = "What is Python used for?"

results = vector_store.similarity_search_with_score(query, k=3)

print(f"Query: '{query}'\n")
print("Results with scores (lower score = more similar):")
for doc, score in results:
    print(f"\n  Score: {score:.4f}")
    print(f"  Content: {doc.page_content}")
```

### 3.4 Implementation: Maximum Marginal Relevance (MMR)

MMR returns diverse results by balancing relevance with diversity:

```python
# mmr_search.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Create diverse document set
documents = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="mmr_demo"
)

query = "What is LangChain?"

# Standard similarity search
similarity_results = vector_store.similarity_search(query, k=3)
print("=== Standard Similarity Search ===")
for doc in similarity_results:
    print(f"  - {doc.page_content}")

# MMR search for diversity
mmr_results = vector_store.similarity_search(
    query,
    k=3,
    filter=None,
    fetch_k=20,  # Fetch more candidates
    lambda_mult=0.5  # 0 = max diversity, 1 = max relevance
)

print("\n=== MMR Search (diverse results) ===")
for doc in mmr_results:
    print(f"  - {doc.page_content}")
```

**Question:** What happens when you set lambda_mult to 0? What about 1?

<details>
<summary>Click to review</summary>

- `lambda_mult=0`: Maximum diversity—returns completely different topics
- `lambda_mult=1`: Maximum relevance—behaves like standard similarity search
- Values around 0.5 provide a balance between relevance and diversity

</details>

### 3.5 Checkpoint

**Self-Assessment:**
- [ ] You can perform basic similarity search
- [ ] You can retrieve results with similarity scores
- [ ] You can use MMR for diverse result retrieval
- [ ] You understand when to use each search type

---

## Chapter 4: Metadata Filtering

Vector stores support filtering by metadata to narrow down search results.

### 4.1 What You Will Build

Learn to combine semantic search with metadata filters for precise retrieval.

### 4.2 Implementation: Filtering by Metadata

Create `metadata_filtering.py`:

```python
# metadata_filtering.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Setup with rich metadata
documents = [
    Document(
        page_content="Virat Kohli - RCB batsman with highest IPL runs",
        metadata={"team": "RCB", "role": "batsman", "nationality": "Indian"}
    ),
    Document(
        page_content="Rohit Sharma - MI captain with 5 titles",
        metadata={"team": "MI", "role": "batsman", "nationality": "Indian"}
    ),
    Document(
        page_content="David Warner - Australian batsman for DC",
        metadata={"team": "DC", "role": "batsman", "nationality": "Australian"}
    ),
    Document(
        page_content="Jasprit Bumrah - MI bowler with best economy",
        metadata={"team": "MI", "role": "bowler", "nationality": "Indian"}
    ),
    Document(
        page_content="Ravindra Jadeja - CSK all-rounder",
        metadata={"team": "CSK", "role": "all-rounder", "nationality": "Indian"}
    ),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="filtered_search"
)

# Filter: Only Mumbai Indians (MI) players
print("=== Filter: Mumbai Indians players ===")
results = vector_store.similarity_search(
    query="Who are the key players?",
    k=5,
    filter={"team": "MI"}
)
for doc in results:
    print(f"  - {doc.page_content} ({doc.metadata})")

# Filter: Bowlers only
print("\n=== Filter: Bowlers ===")
results = vector_store.similarity_search(
    query="Who are the bowlers?",
    k=5,
    filter={"role": "bowler"}
)
for doc in results:
    print(f"  - {doc.page_content} ({doc.metadata})")

# Filter: Indian players (using $in operator)
print("\n=== Filter: Indian players ===")
results = vector_store.similarity_search(
    query="Indian players",
    k=5,
    filter={"nationality": {"$in": ["Indian"]}}
)
for doc in results:
    print(f"  - {doc.page_content} ({doc.metadata})")
```

### 4.3 Implementation: Combining Semantic Search with Filters

Create `hybrid_search.py`:

```python
# hybrid_search.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# Medical documents with categories
documents = [
    Document(
        page_content="Aspirin is used to reduce pain, fever, or inflammation.",
        metadata={"category": "medication", "department": "pharmacology"}
    ),
    Document(
        page_content="Diabetes patients need to monitor blood sugar levels regularly.",
        metadata={"category": "disease", "department": "endocrinology"}
    ),
    Document(
        page_content="Metformin is commonly prescribed for type 2 diabetes.",
        metadata={"category": "medication", "department": "endocrinology"}
    ),
    Document(
        page_content="Ibuprofen is another anti-inflammatory medication.",
        metadata={"category": "medication", "department": "pharmacology"}
    ),
    Document(
        page_content="Heart failure requires careful management of fluid intake.",
        metadata={"category": "disease", "department": "cardiology"}
    ),
]

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="medical_docs"
)

# Search with multiple filters
print("=== Medication category only ===")
results = vector_store.similarity_search_with_score(
    query="anti-inflammatory drugs",
    k=5,
    filter={"category": "medication"}
)

for doc, score in results:
    print(f"  {score:.4f} - {doc.page_content}")

print("\n=== Medication category + endocrinology ===")
results = vector_store.similarity_search_with_score(
    query="diabetes treatment",
    k=5,
    filter={"category": "medication", "department": "endocrinology"}
)

for doc, score in results:
    print(f"  {score:.4f} - {doc.page_content}")
```

### 4.4 Checkpoint

**Self-Assessment:**
- [ ] You can filter by exact metadata match
- [ ] You can combine semantic search with filters
- [ ] You can use $in operator for multiple values
- [ ] You understand when metadata filtering is useful

---

## Chapter 5: Managing Vector Stores

Learn to update, delete, and manage documents in your vector store.

### 5.1 Implementation: Updating and Deleting Documents

Create `manage_documents.py`:

```python
# manage_documents.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create vector store
documents = [
    Document(page_content="Virat Kohli is a famous cricketer.", metadata={"id": "1"}),
    Document(page_content="Rohit Sharma is the MI captain.", metadata={"id": "2"}),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="manage_demo"
)

print(f"Initial count: {vector_store._collection.count()}")

# Get document IDs
all_docs = vector_store.get()
print(f"Document IDs: {all_docs['ids']}")

# Update a document
updated_doc = Document(
    page_content="Virat Kohli (updated) is a world-renowned batsman who has scored over 7000 IPL runs.",
    metadata={"id": "1"}
)
vector_store.update_document(document_id="1", document=updated_doc)

print("\nAfter update:")
results = vector_store.get(include=["documents"])
for doc in results["documents"]:
    print(f"  - {doc[:50]}...")

# Delete a document
vector_store.delete(ids=["2"])
print(f"\nAfter delete: {vector_store._collection.count()} documents")
```

### 5.2 Implementation: Converting Vector Store to Retriever

Create `retriever_conversion.py`:

```python
# retriever_conversion.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document(page_content="LangChain helps build LLM apps.", metadata={"topic": "langchain"}),
    Document(page_content="Chroma is a vector database.", metadata={"topic": "database"}),
    Document(page_content="Embeddings represent text as vectors.", metadata={"topic": "embeddings"}),
]

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="retriever_demo"
)

# Convert to retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}
)

# Use retriever
query = "What is LangChain?"
results = retriever.invoke(query)

print(f"Query: {query}\n")
print("Retrieved documents:")
for doc in results:
    print(f"  - {doc.page_content} ({doc.metadata})")
```

### 5.3 Checkpoint

**Self-Assessment:**
- [ ] You can update documents in a vector store
- [ ] You can delete documents from a vector store
- [ ] You can convert a vector store to a retriever
- [ ] You can configure retriever search parameters

---

## Epilogue: The Complete System

Your vector store pipeline now supports:

| Operation | Method | Purpose |
|-----------|--------|---------|
| Create | Chroma.from_documents() | Initialize and populate store |
| Search | similarity_search() | Basic similarity retrieval |
| Search with scores | similarity_search_with_score() | Get relevance scores |
| Diverse results | similarity_search(mmr) | Balance relevance and diversity |
| Filter | filter parameter | Narrow by metadata |
| Update | update_document() | Modify existing documents |
| Delete | delete() | Remove documents |
| Persist | save_local() | Save to disk |

**Key parameters:**
- `k`: Number of results to return
- `fetch_k`: Candidates to consider before MMR selection
- `lambda_mult`: Relevance-diversity balance
- `filter`: Metadata filter conditions

**End-to-end verification:**

```python
# verify_vectorstore.py
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

embeddings = OllamaEmbeddings(model="nomic-embed-text")

documents = [
    Document(page_content="Sample document about RAG systems.", metadata={"topic": "AI"}),
]

store = Chroma.from_documents(documents, embeddings, collection_name="test")
results = store.similarity_search("RAG", k=1)

print(f"Documents in store: {store._collection.count()}")
print(f"Search works: {len(results) > 0}")
print("Vector store is working correctly!")
```

---

## The Principles

1. **Embeddings capture meaning**: Vector representations enable semantic search beyond keyword matching.

2. **Choose embeddings based on use case**: Local models (Ollama) offer privacy, cloud models (OpenAI) offer quality.

3. **Combine semantic search with filters**: Use metadata filters to narrow results after semantic retrieval.

4. **MMR balances relevance and diversity**: For recommendation systems, MMR prevents homogeneous results.

5. **Persist vector stores for production**: Always save your index to disk for production deployments.

---

## Troubleshooting

### Error: Could not connect to Ollama

**Cause:** Ollama server is not running.

**Solution:**
```bash
ollama serve
# In another terminal:
ollama pull nomic-embed-text
```

### No results returned from search

**Cause:** Embedding model not set correctly or query has no semantic match.

**Solution:**
```python
# Verify embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text")
test_embedding = embeddings.embed_query("test")
print(f"Embedding dimension: {len(test_embedding)}")
```

### Wrong results with filters

**Cause:** Filter syntax incorrect or metadata not stored properly.

**Solution:**
```python
# Check stored metadata
results = vector_store.get(include=["metadatas"])
print(results["metadatas"])
```

### Vector store too slow

**Cause:** Too many documents without proper indexing.

**Solution:**
- Use Chroma with HNSW persistence
- Reduce fetch_k parameter
- Implement pagination for large result sets

---

## Next Steps

After completing this lab:

1. Experiment with different embedding models (OpenAI, Cohere)
2. Implement hybrid search combining vector and keyword search
3. Progress to Lab 4: Retrieval to learn advanced retrieval patterns

---

## Additional Resources

- [LangChain Vector Stores Documentation](https://python.langchain.com/docs/modules/data_connection/vectorstores/)
- [Chroma Documentation](https://docs.trychroma.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Embedding Model Comparison](https://python.langchain.com/docs/integrations/text_embedding/)