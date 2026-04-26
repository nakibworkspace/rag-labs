# RAG Hands-On Labs

This directory contains comprehensive hands-on labs for learning Retrieval-Augmented Generation (RAG) using LangChain.

## Overview

The labs follow a sequential progression through the four main stages of the RAG pipeline:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Document       │     │    Text         │     │    Vector       │     │    Retrieval    │
│  Loading        │────▶│    Splitting    │────▶│    Store        │────▶│    + Generation │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Lab Structure

### Lab 1: Introduction to RAG and Document Loaders
**Duration:** ~2 hours

**Topics covered:**
- What is RAG and why it is used
- The four stages of the RAG pipeline
- Understanding LangChain Document objects
- Loading PDF documents with PyPDFLoader
- Loading CSV files with CSVLoader
- Loading text files with TextLoader
- Loading web pages with WebBaseLoader
- Loading multiple files with DirectoryLoader
- Lazy loading vs. eager loading for memory management

**Key concepts:** Document loaders, Document objects, metadata, lazy loading

---

### Lab 2: Text Splitting
**Duration:** ~2.5 hours

**Topics covered:**
- Why text splitting is necessary
- Understanding chunk_size and chunk_overlap
- CharacterTextSplitter for basic splitting
- RecursiveCharacterTextSplitter for better boundaries
- Language-specific splitters (Python, Markdown, HTML)
- Semantic chunking with embedding-based breakpoints
- Choosing the right splitting strategy

**Key concepts:** Character splitting, recursive splitting, semantic chunking, chunk boundaries

---

### Lab 3: Vector Stores
**Duration:** ~2.5 hours

**Topics covered:**
- Understanding vector embeddings
- Creating embeddings with Ollama and OpenAI
- Working with Chroma vector store
- Working with FAISS vector store
- Adding, updating, and deleting documents
- Similarity search with scores
- Maximum Marginal Relevance (MMR)
- Metadata filtering
- Converting vector store to retriever
- Persisting vector stores

**Key concepts:** Embeddings, similarity search, vector databases, metadata filtering, MMR

---

### Lab 4: Retrieval
**Duration:** ~3 hours

**Topics covered:**
- Retrieval fundamentals in RAG
- Building retrieval chains with LLMs
- MultiQueryRetriever for vocabulary mismatch
- Contextual compression for noise reduction
- Ensemble retrieval combining multiple strategies
- Self-query retrieval for automatic filtering
- Parent document retrieval for context preservation

**Key concepts:** Retrieval chains, multi-query, compression, ensemble, MMR

---

## Prerequisites

Each lab has specific prerequisites:

| Lab | Prerequisites |
|-----|---------------|
| Lab 1 | Basic Python knowledge |
| Lab 2 | Lab 1 completion or equivalent |
| Lab 3 | Lab 2 completion or equivalent |
| Lab 4 | Lab 3 completion or equivalent |

## Environment Setup

Common requirements for all labs:

```bash
# Core packages
pip install langchain langchain-community langchain-chroma langchain-ollama

# Optional for specific labs
pip install pypdf beautifulsoup4 python-dotenv faiss-cpu openai wikipedia

# For local embeddings and LLMs (optional)
pip install ollama
ollama pull llama3.2
ollama pull nomic-embed-text
```

## Learning Approach

These labs follow the Poridhi Labs methodology emphasizing:

1. **Active learning**: Fill-in-the-blank exercises, prediction tasks
2. **Hands-on implementation**: Build each component from scratch
3. **Progressive complexity**: Each lab builds on previous knowledge
4. **Real-world scenarios**: Context-based challenges that motivate learning
5. **Self-assessment**: Checkpoints after each chapter to verify understanding

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install langchain langchain-community langchain-chroma langchain-ollama ollama
   ```

2. **Start Ollama (if using local models):**
   ```bash
   ollama serve &
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Start with Lab 1:**
   ```bash
   cd docs
   # Read lab-01-introduction-to-rag-and-document-loaders.md
   ```

## Directory Structure

```
hands-on-labs/
├── docs/
│   ├── lab-01-introduction-to-rag-and-document-loaders.md
│   ├── lab-02-text-splitting.md
│   ├── lab-03-vector-stores.md
│   ├── lab-04-retrieval.md
│   └── README.md (this file)
└── diagrams/
    └── (architecture diagrams)
```

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangChain Academy](https://academy.langchain.com/)
- [RAG Survey Paper](https://arxiv.org/abs/2407.01254)

## License

These labs are developed for educational purposes as part of Poridhi's technical training program.