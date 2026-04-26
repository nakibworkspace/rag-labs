# Lab 1: Introduction to RAG and Document Loaders

## Introduction

This lab introduces Retrieval-Augmented Generation (RAG), a powerful architecture that combines large language models with external knowledge bases. You will learn the fundamental concepts of RAG and implement document loaders to ingest various data sources into your application.

This lab approaches RAG by building each component step-by-step, giving you a complete understanding of how data flows through the system. By the end, you will have a working document loading pipeline that can be extended into a full RAG application.

**Prerequisites:** Basic Python knowledge including functions, classes, and working with packages. Familiarity with JSON and file operations is helpful.

## Learning Objectives

By the end of this lab, you will be able to:

1. Explain what RAG is and why it is used in production AI applications
2. Identify the four main stages of the RAG pipeline
3. Use LangChain document loaders to import data from PDFs, CSV files, text files, web pages, and directories
4. Describe the structure of LangChain Document objects
5. Implement lazy loading for large datasets to manage memory efficiently

## Prologue: The Challenge

You join a data engineering team at a healthcare analytics company. The company has accumulated years of medical research papers, patient records, and clinical trial data across multiple file formats and locations. The research team wants to build a system that allows doctors and researchers to ask questions in natural language and get accurate answers drawn from these documents.

Currently, the data sits in silos: PDFs in one folder, CSVs in a database, text files in another location, and web articles scattered across bookmarks. The team needs a unified way to ingest all this data into a system that can later power a question-answering application.

Your task is to build the data ingestion layer—the first critical component of the RAG pipeline. This layer must handle multiple document formats, manage memory efficiently for large datasets, and produce a standardized output that subsequent pipeline stages can consume.

Success criteria:
- Load documents from at least 5 different sources (PDF, CSV, Text, Web, Directory)
- Handle large datasets without running out of memory
- Produce consistent Document objects that other pipeline components can process

## Environment Setup

Install the required packages:

```bash
pip install langchain langchain-community langchain-ollama ollama beautifulsoup4 python-dotenv pypdf
```

If you are using a local Ollama instance for embeddings or LLM access:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2
ollama pull nomic-embed-text
```

Create the project structure:

```bash
mkdir -p lab1_document_loaders
cd lab1_document_loaders
mkdir -p data books articles
```

---

## Chapter 1: Understanding RAG Architecture

Before implementing document loaders, you need to understand where they fit in the RAG pipeline and why each component exists.

### 1.1 What You Will Build

The RAG pipeline consists of four sequential stages:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Document       │     │    Text         │     │    Vector       │     │    Retrieval    │
│  Loading        │────▶│    Splitting    │────▶│    Store        │────▶│    + Generation │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

Each stage transforms data into a form the next stage can use efficiently.

### 1.2 Think First: Why Do We Need RAG?

**Question:** Large language models already contain vast amounts of knowledge. Why do we need to add external document retrieval instead of relying solely on the model's training data?

<details>
<summary>Click to review</summary>

There are three key reasons:

1. **Knowledge cutoff**: LLMs are trained on data up to a specific date. They cannot answer questions about events, research, or information that emerged after their training.

2. **Domain-specific knowledge**: General-purpose models may lack expertise in specialized fields like your company's internal documents, recent medical research, or proprietary codebases.

3. **Hallucination control**: When a model retrieves relevant context before answering, you can verify the sources and reduce fabricated information.

</details>

### 1.3 Implementation: The Document Object

LangChain standardizes all loaded content into Document objects. Understanding this structure is essential.

Create `document_structure.py`:

```python
# document_structure.py
from langchain_core.documents import Document

# A Document object contains two key fields:
# - page_content: The actual text content (string)
# - metadata: Additional information about the source (dict)

doc = Document(
    page_content="This is the text content of the document.",
    metadata={
        "source": "example.txt",
        "page": 1,
        "author": "John Doe"
    }
)

# Access the fields
print(f"Content: {doc.page_content}")
print(f"Metadata: {doc.metadata}")
print(f"Source file: {doc.metadata['source']}")
```

**Self-Assessment:**
- [ ] You can create a Document object with custom content and metadata
- [ ] You can access both page_content and metadata fields
- [ ] You understand why metadata is useful for tracking document sources

---

## Chapter 2: Loading PDF Documents

PDFs are a common format for academic papers, reports, and documentation. LangChain provides specialized loaders for different document types.

### 2.1 Think First: Why Not Just Read PDF Files Directly?

**Question:** Operating systems can open PDF files. Why do we need specialized document loaders instead of simply reading the file content?

<details>
<summary>Click to review</summary>

PDF loaders handle several complexities:

1. **Structure extraction**: PDFs store content as positioned text blocks, not paragraphs. Loaders reconstruct logical reading order.

2. **Metadata extraction**: Loaders extract document properties like author, creation date, and number of pages.

3. **Multi-page handling**: Loaders process each page and maintain page numbers for reference.

4. **Encoding issues**: Loaders handle various character encodings and special characters automatically.

</details>

### 2.2 Implementation: Loading a Single PDF

Create `pdf_loader.py`:

```python
# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader

# Initialize the loader with the PDF file path
loader = PyPDFLoader("data/medical_paper.pdf")

# Load all pages
docs = loader.load()

# Explore the loaded documents
print(f"Total pages loaded: {len(docs)}")
print(f"\nFirst page content (first 500 chars):")
print(docs[0].page_content[:500])
print(f"\nFirst page metadata:")
print(docs[0].metadata)
```

**Prediction Exercise:**

What do you expect to see in `docs[0].metadata`?

- A) The full text of the first page
- B) A dictionary with page number and source file
- C) A list of all pages in the document

<details>
<summary>Click to verify</summary>

The answer is B. The metadata contains at minimum:
- `source`: The file path
- `page`: The page number (0-indexed)

</details>

### 2.3 Implementation: Loading Multiple PDFs from a Directory

For larger datasets, you often need to load multiple files from a folder.

Create `directory_loader.py`:

```python
# directory_loader.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Method 1: Load all documents into memory (for small datasets)
loader = DirectoryLoader(
    path="books/",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(f"Loaded {len(docs)} documents")
for doc in docs[:3]:
    print(f"  - {doc.metadata.get('source', 'unknown')}")

# Method 2: Lazy loading (for large datasets)
# This does NOT load all documents at once
lazy_docs = loader.lazy_load()

print("\nUsing lazy loading:")
for i, doc in enumerate(lazy_docs):
    print(f"  Document {i}: {doc.metadata.get('source', 'unknown')}")
    if i >= 2:  # Stop after 3 for demonstration
        print("  ... (more documents available)")
        break
```

### 2.4 Checkpoint

**Self-Assessment:**
- [ ] You can load a single PDF using PyPDFLoader
- [ ] You can load multiple PDFs from a directory
- [ ] You understand when to use load() vs lazy_load()
- [ ] You can access page content and metadata from loaded documents

---

## Chapter 3: Loading CSV and Text Files

Structured data like CSVs and plain text files require different handling approaches.

### 3.1 Think First: Why Do Different File Types Need Different Loaders?

**Question:** All loaders return Document objects with page_content and metadata. Why do we need different loader classes for different file formats?

<details>
<summary>Click to review</summary>

Different loaders parse their specific format:

1. **CSV loaders** treat each row as a separate document, extracting column headers and values into meaningful text.

2. **Text loaders** handle encoding issues (UTF-8, ASCII, etc.) and preserve line breaks appropriately.

3. **PDF loaders** must reconstruct document structure from positioned text blocks.

Each format has unique parsing challenges that the specialized loaders handle internally.

</details>

### 3.2 Implementation: Loading CSV Files

Create `csv_loader.py`:

```python
# csv_loader.py
from langchain_community.document_loaders import CSVLoader

# Load a CSV file
loader = CSVLoader(file_path="data/survey_responses.csv")

docs = loader.load()

print(f"Total rows loaded: {len(docs)}")
print(f"\nFirst row content:")
print(docs[0].page_content)
print(f"\nFirst row metadata:")
print(docs[0].metadata)

# Access specific columns
print(f"\nExamining a few rows:")
for i, doc in enumerate(docs[:3]):
    print(f"\nRow {i+1}:")
    print(doc.page_content[:200])
```

### 3.3 Implementation: Loading Text Files

Create `text_loader.py`:

```python
# text_loader.py
from langchain_community.document_loaders import TextLoader

# Load a text file with UTF-8 encoding
loader = TextLoader("data/article.txt", encoding="utf-8")

docs = loader.load()

print(f"Content type: {type(docs[0].page_content)}")
print(f"\nDocument content:")
print(docs[0].page_content)
print(f"\nMetadata:")
print(docs[0].metadata)
```

### 3.4 Implementation: Web Page Loader

Create `web_loader.py`:

```python
# web_loader.py
from langchain_community.document_loaders import WebBaseLoader

# Load content from a web page
loader = WebBaseLoader("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")

docs = loader.load()

print(f"Content length: {len(docs[0].page_content)} characters")
print(f"\nFirst 1000 characters:")
print(docs[0].page_content[:1000])
print(f"\nMetadata:")
print(docs[0].metadata)
```

### 3.5 Checkpoint

**Self-Assessment:**
- [ ] You can load CSV files with CSVLoader
- [ ] You can load text files with TextLoader
- [ ] You can load web pages with WebBaseLoader
- [ ] You understand why each file type needs a specific loader

---

## Chapter 4: Complete Document Loading Pipeline

Now combine all the loaders into a unified pipeline that can handle multiple source types.

### 4.1 What You Will Build

Create a modular document loading system that:
- Accepts multiple file types
- Processes files from directories
- Handles errors gracefully
- Produces standardized Document objects

### 4.2 Implementation: Unified Document Loader

Create `unified_loader.py`:

```python
# unified_loader.py
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    WebBaseLoader,
    DirectoryLoader
)
from pathlib import Path
from typing import List
from langchain_core.documents import Document

def load_document(file_path: str) -> List[Document]:
    """Load a single document based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(file_path)
    elif suffix == ".csv":
        loader = CSVLoader(file_path=file_path)
    elif suffix == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()

def load_directory(dir_path: str, glob_pattern: str = "*.pdf") -> List[Document]:
    """Load all matching files from a directory."""
    loader = DirectoryLoader(
        path=dir_path,
        glob=glob_pattern,
        loader_cls=PyPDFLoader
    )
    return loader.load()

def load_web_page(url: str) -> List[Document]:
    """Load content from a web page."""
    loader = WebBaseLoader(url)
    return loader.load()

# Example usage
if __name__ == "__main__":
    # Load various document types
    print("Loading sample documents...")

    # Note: These files must exist in your data directory
    # Uncomment and modify paths as needed

    # pdf_docs = load_document("data/sample.pdf")
    # print(f"PDF: {len(pdf_docs)} pages")

    # csv_docs = load_document("data/survey.csv")
    # print(f"CSV: {len(csv_docs)} rows")

    # text_docs = load_document("data/notes.txt")
    # print(f"Text: {len(text_docs)} documents")

    web_docs = load_web_page("https://python.org")
    print(f"Web: {len(web_docs)} documents")
    print(f"Content preview: {web_docs[0].page_content[:200]}")
```

### 4.3 Experiment: Memory Management

Test the difference between loading methods:

```python
# experiment_memory.py
import sys

# First, check memory usage with load()
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(path="books/", glob="*.pdf", loader_cls=PyPDFLoader)

# This loads everything into memory at once
print("Using load():")
docs = loader.load()
print(f"  Memory for documents: {sys.getsizeof(docs)} bytes")
print(f"  Number of documents: {len(docs)}")

# This yields documents one at a time
print("\nUsing lazy_load():")
lazy_gen = loader.lazy_load()
first_doc = next(lazy_gen)
print(f"  First document loaded: {first_doc.metadata.get('source', 'unknown')}")
print("  Remaining documents can be iterated as needed")
```

**Question:** In a production system processing thousands of PDFs, which loading method would you choose and why?

<details>
<summary>Click to review</summary>

Use `lazy_load()` for large datasets because:
- It does not load all documents into memory at once
- It processes documents one at a time as you iterate
- It prevents out-of-memory errors with large document collections
- It allows for parallel processing if needed

</details>

### 4.4 Checkpoint

**Self-Assessment:**
- [ ] You can create a unified loader function that handles multiple file types
- [ ] You understand the difference between load() and lazy_load()
- [ ] You can load documents from directories
- [ ] You can load web pages as documents
- [ ] You can explain memory implications of different loading strategies

---

## Epilogue: The Complete System

Your document loading pipeline now handles:

| Source Type | Loader Class | Use Case |
|-------------|--------------|----------|
| PDF | PyPDFLoader | Academic papers, reports |
| CSV | CSVLoader | Structured tabular data |
| Text | TextLoader | Plain text documents |
| Web | WebBaseLoader | Online articles |
| Directory | DirectoryLoader | Batch loading multiple files |

All loaders produce standardized Document objects with:
- `page_content`: The extracted text
- `metadata`: Source information (file path, page number, etc.)

**End-to-end verification:**

```python
# verify_pipeline.py
from unified_loader import load_web_page

# Test the complete pipeline
docs = load_web_page("https://en.wikipedia.org/wiki/Large_language_model")

print(f"Documents loaded: {len(docs)}")
print(f"Content length: {len(docs[0].page_content)} characters")
print(f"Metadata: {docs[0].metadata}")
print("Document loading pipeline is working correctly!")
```

---

## The Principles

1. **Standardization through Document objects**: All data sources convert to a common format, enabling pipeline components to work interchangeably with any data source.

2. **Lazy loading for scale**: Always use lazy loading when dealing with potentially large datasets to prevent memory exhaustion.

3. **Metadata preservation**: Track the source of each document. This information becomes critical for citation and debugging in production systems.

4. **Modular loader design**: Build loader functions that can be easily extended to support new document types without modifying existing code.

---

## Troubleshooting

### Error: FileNotFoundError

**Cause:** The file path is incorrect or the file does not exist.

**Solution:**
```python
from pathlib import Path

# Verify file exists
file_path = "data/document.pdf"
if Path(file_path).exists():
    print("File found!")
else:
    print(f"File not found. Available files: {list(Path('data').glob('*'))}")
```

### Error: UnicodeDecodeError

**Cause:** Text file encoding does not match what the loader expects.

**Solution:**
```python
# Try different encodings
loader = TextLoader("data/file.txt", encoding="latin-1")
# or
loader = TextLoader("data/file.txt", encoding="utf-16")
```

### Error: ModuleNotFoundError

**Cause:** Required package is not installed.

**Solution:**
```bash
pip install langchain-community pypdf beautifulsoup4
```

### Error: PDF Password Protected

**Cause:** The PDF requires a password to open.

**Solution:**
```python
loader = PyPDFLoader("data/protected.pdf", password="your_password")
```

---

## Next Steps

After completing this lab:

1. Extend the unified loader to support additional formats (JSON, HTML, Markdown)
2. Add error handling and logging to the pipeline
3. Implement parallel loading for faster processing of large directories
4. Progress to Lab 2: Text Splitting to learn how to prepare loaded documents for vector storage

---

## Additional Resources

- [LangChain Document Loaders Documentation](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [PyPDFLoader API Reference](https://python.langchain.com/docs/integrations/document_loaders/pypdf/)
- [Wikipedia on Retrieval-Augmented Generation](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)