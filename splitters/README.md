# Lab 2: Text Splitting

## Introduction

This lab covers text splitting, the second critical stage of the RAG pipeline. After loading documents, you must divide them into smaller chunks that can be embedded and searched effectively. The way you split text directly impacts retrieval quality and downstream application performance.

![dia1](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/splitters/assets/rag2-1.drawio.svg)

This lab teaches multiple splitting strategies, from simple character-based approaches to sophisticated semantic chunking that respects document structure.

**Prerequisites:** Completion of Lab 1 (Document Loaders) or equivalent knowledge of loading documents with LangChain.

## Learning Objectives

By the end of this lab, you will be able to:

1. Explain why text splitting is necessary in RAG systems
2. Implement character-based text splitting with configurable parameters
3. Use recursive text splitting for better boundary detection
4. Apply language-specific splitters for code and markdown
5. Implement semantic chunking based on embedding similarity
6. Choose appropriate splitting strategies for different document types

## Prologue: The Challenge

Continuing from Lab 1, your document loading pipeline is working. However, when you attempt to embed and search the documents, you encounter problems:

1. **Context overflow**: Large documents exceed embedding model token limits
2. **Loss of specificity**: A query about "heart disease treatment" returns an entire 50-page medical paper instead of the relevant section
3. **Poor recall**: Key information gets buried in large text chunks, making retrieval unreliable
4. **Redundancy**: Similar content appears in multiple chunks, diluting search relevance

The research team reports that search results are either too broad (returning entire documents) or too narrow (missing relevant information). You need to implement intelligent text splitting that creates meaningful, searchable chunks while preserving context.

Success criteria:
- Create chunks of appropriate size for embedding models (typically 200-1000 tokens)
- Preserve meaningful boundaries (paragraphs, sections, code blocks)
- Maintain overlap between chunks to prevent context loss at boundaries
- Handle different document types (plain text, code, markdown) appropriately

## Environment Setup

Ensure you have the required packages:

```bash
pip install langchain-text-splitters langchain-experimental langchain-ollama ollama
```

Start Ollama and pull required models:

```bash
ollama serve &
ollama pull nomic-embed-text
```

---

## Chapter 1: Understanding Text Splitting

Before implementing splitters, you must understand why splitting matters and what makes a good chunk.

### 1.1 What You Will Build

Text splitting transforms raw document content into smaller, semantically coherent pieces that can be:
- Embedded into vector space efficiently
- Retrieved precisely based on query relevance
- Passed to language models as context

### 1.2 Think First: What Makes a Good Chunk?

**Question:** If you split a document about climate change at random positions (e.g., every 500 characters), what problems might occur?

<details>
<summary>Click to review</summary>

Random splitting causes several issues:

1. **Meaning disruption**: A sentence about "rising temperatures causing sea level rise" might be split between "rising temperatures" and "causing sea level rise", making both chunks semantically incomplete.

2. **Context loss**: A paragraph explaining a concept followed by an example supporting it—if split incorrectly—loses the connection between explanation and example.

3. **Search quality degradation**: Embeddings of incomplete phrases do not represent the original meaning, leading to poor similarity matching.

4. **Redundant embeddings**: Similar context appearing in multiple chunks dilutes the semantic signal.

</details>

### 1.3 The Chunking Parameters

Key parameters that control splitting behavior:

| Parameter | Purpose | Typical Range |
|-----------|---------|---------------|
| chunk_size | Maximum characters/tokens per chunk | 200-1000 |
| chunk_overlap | Characters shared between adjacent chunks | 0-200 |
| separator | Character(s) to split on | "\n\n", "\n", " " |


### 1.4 What is Chunk Overlap?
In RAG chunk overlap is a practice of maintaining shared text between consecutive document chunks (typically 10–20% of the chunk size) to ensure semantic context is preserved across boundaries. It prevents losing information split between segments, improving retrieval accuracy for context-dependent data, though too much overlap adds redundant noise.

![dia2](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/splitters/assets/rag2-2.drawio.svg)


### 1.5 Implementation: Basic Chunk Exploration

Create `chunk_explorer.py` to understand how splitting works:

```python
# chunk_explorer.py
from langchain_text_splitters import CharacterTextSplitter

# Sample text with clear paragraph structure
text = """Climate change refers to long-term shifts in temperatures and weather patterns.
Since the 1800s, human activities have been the main driver of climate change,
primarily due to burning fossil fuels like coal, oil and gas.

Burning fossil fuels generates greenhouse gas emissions that act like a blanket
around Earth, trapping the sun's heat and causing the planet to warm. This is
known as the greenhouse effect.

The main greenhouse gases are:
- Carbon dioxide (CO2)
- Methane (CH4)
- Nitrous oxide (N2O)

Countries agreed at the 2015 Paris Agreement to limit global temperature rise
to 1.5 degrees Celsius above pre-industrial levels."""

# Create a splitter with specific parameters
splitter = CharacterTextSplitter(
    chunk_size=200,      # Maximum chunk size in characters
    chunk_overlap=50,    # Overlap between chunks
    separator="\n\n"     # Split on paragraph boundaries
)

# Split the text
chunks = splitter.split_text(text)

print(f"Number of chunks created: {len(chunks)}\n")

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk)
    print()
```

**Predict:** How many chunks will be created with chunk_size=200 and chunk_overlap=50?

<details>
<summary>Click to verify</summary>

The exact number depends on the text, but with chunk_size=200 and the paragraph structure, you should get approximately 5-7 chunks. The overlap ensures that context is preserved across chunk boundaries.

</details>

### 1.6 Checkpoint

**Self-Assessment:**
- [ ] You understand why chunk_size and chunk_overlap matter
- [ ] You can create basic CharacterTextSplitter
- [ ] You can observe how text is divided into chunks

---

## Chapter 2: Recursive Text Splitting

Recursive splitting attempts multiple separators in sequence, falling back to smaller separators when needed.

![dia3](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/splitters/assets/rag2-3.drawio.svg)

### 2.1 Think First: Why Use Multiple Separators?

**Question:** Why not simply split on newlines ("\n") or spaces (" ")? What advantage does using multiple separators in sequence provide?

<details>
<summary>Click to review</summary>

Using multiple separators in sequence (from largest to smallest) provides:

1. **Priority to natural boundaries**: Paragraph breaks (double newlines) are more meaningful than single line breaks.

2. **Graceful degradation**: If a paragraph exceeds chunk_size, the splitter tries single newlines, then spaces, then individual characters.

3. **Preserved structure**: Longer separators maintain document structure when possible.

4. **Complete coverage**: Even very long words or concatenated content eventually gets split.

</details>

### 2.2 Implementation: Recursive Character Splitting

Create `recursive_splitter.py`:

```python
# recursive_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Sample text with various structures
text = """# Machine Learning Overview

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

There are three main types:

1. **Supervised Learning**: Learning from labeled data
2. **Unsupervised Learning**: Finding patterns in unlabeled data
3. **Reinforcement Learning**: Learning through trial and error

## Applications

Machine learning powers many modern applications including image recognition,
natural language processing, recommendation systems, and autonomous vehicles.

The field continues to evolve rapidly with new algorithms and techniques
being developed regularly."""

# Method 1: Default separator order
default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

# Method 2: Custom separator order
custom_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]
)

print("=== Default Separators ===")
default_chunks = default_splitter.split_text(text)
for i, chunk in enumerate(default_chunks[:3]):
    print(f"Chunk {i+1}: {chunk[:100]}...\n")

print("\n=== Custom Separators ===")
custom_chunks = custom_splitter.split_text(text)
for i, chunk in enumerate(custom_chunks[:3]):
    print(f"Chunk {i+1}: {chunk[:100]}...\n")
```

### 2.3 Implementation: Splitting Documents

Split document objects (not just raw text):

```python
# split_documents.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load a document
loader = TextLoader("data/article.txt", encoding="utf-8")
documents = loader.load()

# Create the splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

# Split documents (preserves metadata)
chunks = splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Resulting chunks: {len(chunks)}")
print(f"\nFirst chunk content:")
print(chunks[0].page_content)
print(f"\nFirst chunk metadata (preserved from original):")
print(chunks[0].metadata)
```

### 2.4 Checkpoint

**Self-Assessment:**
- [ ] You can explain why recursive splitting works better than single-separator splitting
- [ ] You can customize separator order for different document types
- [ ] You can split document objects while preserving metadata

---

## Chapter 3: Language-Specific Splitting

Different content types (code, markdown) have their own structural elements that should be respected during splitting.

### 3.1 Think First: Why Split Code Differently?

**Question:** Why would you use a Python-specific splitter instead of a general text splitter for Python code?

<details>
<summary>Click to review</summary>

Code requires special handling:

1. **Syntax preservation**: Splitting in the middle of a function definition or class would produce meaningless chunks.

2. **Indentation significance**: In Python, indentation defines code blocks. Losing indentation breaks the code.

3. **Comment grouping**: Keeping related code with its comments provides better context.

4. **Language-specific tokens**: Each programming language has specific delimiters (braces in C++, indent in Python, keywords).

</details>

### 3.2 Implementation: Python Code Splitting

Create `python_splitter.py`:

```python
# python_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

# Python code sample
python_code = """
class Student:
    def __init__(self, name, age, grade):
        self.name = name
        self.age = age
        self.grade = grade

    def get_details(self):
        return f"{self.name} is {self.age} years old"

    def is_passing(self):
        return self.grade >= 6.0


def calculate_average(grades):
    if not grades:
        return 0
    return sum(grades) / len(grades)


# Example usage
student1 = Student("Alice", 20, 8.5)
print(student1.get_details())
"""

# Create Python-specific splitter
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500,
    chunk_overlap=50
)

chunks = python_splitter.split_text(python_code)

print(f"Created {len(chunks)} chunks from Python code:\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk)
    print()
```

### 3.3 Implementation: Markdown Splitting

Create `markdown_splitter.py`:

```python
# markdown_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

markdown_text = """
# Project Documentation

## Overview

This project implements a REST API for managing user data.

## Installation

```bash
pip install flask
pip install sqlalchemy
```

## Usage

```python
from app import create_app
app = create_app()
app.run()
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /users | List all users |
| POST | /users | Create user |
| GET | /users/:id | Get user |

## Configuration

Set the following environment variables:
- DATABASE_URL
- SECRET_KEY
"""

# Create Markdown-specific splitter
markdown_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=500,
    chunk_overlap=50
)

chunks = markdown_splitter.split_text(markdown_text)

print(f"Created {len(chunks)} chunks from Markdown:\n")
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk[:200])
    print()
```

### 3.4 Experiment: Comparing Splitting Strategies

```python
# experiment_comparison.py
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    Language
)

text = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total

def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    return calculate_sum(numbers) / len(numbers)

# Test the functions
result = calculate_average([10, 20, 30, 40, 50])
print(f"Average: {result}")
"""

# Strategy 1: Simple character splitting
char_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)

# Strategy 2: Recursive text splitting
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

# Strategy 3: Language-specific splitting
code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=0
)

char_chunks = char_splitter.split_text(text)
recursive_chunks = recursive_splitter.split_text(text)
code_chunks = code_splitter.split_text(text)

print(f"Character splitting: {len(char_chunks)} chunks")
print(f"Recursive splitting: {len(recursive_chunks)} chunks")
print(f"Code splitting: {len(code_chunks)} chunks")

print("\n--- Character Split (loses code structure) ---")
print(char_chunks[0])

print("\n--- Code Split (preserves functions) ---")
print(code_chunks[0])
```

**Observe:** Character splitting may break code mid-function. Code-aware splitting keeps complete function definitions together.

### 3.5 Checkpoint

**Self-Assessment:**
- [ ] You can use Language enum to specify document type
- [ ] You can split Python code while preserving function definitions
- [ ] You can split Markdown while preserving headers and code blocks
- [ ] You can explain when to use language-specific vs. general splitters

---

## Chapter 4: Semantic Chunking

Semantic chunking uses embedding similarity to identify natural topic boundaries rather than relying on character counts.

![dia4](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/splitters/assets/rag2-4.drawio.svg)

### 4.1 What You Will Build

Unlike fixed-size splitting, semantic chunking:
- Identifies natural breakpoints in content
- Groups related concepts together
- Adapts chunk sizes based on content structure

### 4.2 Implementation: Semantic Chunking

Create `semantic_chunker.py`:

```python
# semantic_chunker.py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create semantic chunker
text_splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1.0
)

# Sample text with distinct topics
sample_text = """
Farmers were working hard in the fields, preparing the soil and planting seeds
for the next season. The sun was bright, and the air smelled of earth and fresh
grass. Agriculture has been the backbone of rural economies for thousands of years.

The Indian Premier League (IPL) is the biggest cricket league in the world.
People all over the world watch the matches and cheer for their favourite teams.
Cricket is particularly popular in South Asia, Australia, and England.

Terrorism is a big danger to peace and safety in modern societies. It causes
harm to innocent people and creates fear in cities and villages. Governments
around the world work together to combat this threat through intelligence
sharing and security measures.

Machine learning is transforming technology companies. Neural networks can now
recognize images, translate languages, and even generate creative content.
The field advances rapidly with new research published daily.
"""

# Create documents using semantic chunking
docs = text_splitter.create_documents([sample_text])

print(f"Total semantic chunks created: {len(docs)}\n")
for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
    print()
```

### 4.3 Understanding Breakpoint Threshold Types

Semantic chunker supports different threshold strategies:

```python
# threshold_types.py
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Different threshold strategies
strategies = [
    {"type": "percentile", "amount": 95},
    {"type": "standard_deviation", "amount": 1.0},
    {"type": "interquartile", "amount": 1.5},
    {"type": "gradient", "amount": 9.8},
]

sample_text = """
Machine learning is a field of study that gives computers the ability to learn
without being explicitly programmed. It focuses on developing algorithms that
can access data and use it to learn patterns.

The sun is a star at the center of our solar system. It provides light and heat
for Earth, making life possible. The sun consists mostly of hydrogen and helium.

Python is a high-level programming language known for its readability and
versatility. It is widely used in web development, data science, and AI.
"""

for strategy in strategies:
    splitter = SemanticChunker(
        embeddings,
        breakpoint_threshold_type=strategy["type"],
        breakpoint_threshold_amount=strategy["amount"]
    )
    chunks = splitter.split_text(sample_text)
    print(f"{strategy['type']}: {len(chunks)} chunks")
```

### 4.4 Checkpoint

**Self-Assessment:**
- [ ] You can create a SemanticChunker with embedding model
- [ ] You can explain what breakpoint_threshold controls
- [ ] You understand when semantic chunking is preferable to fixed-size chunking
- [ ] You can compare different threshold strategies

---

## Chapter 5: Choosing the Right Strategy

Different use cases require different splitting strategies.

### 5.1 Decision Framework

| Use Case | Recommended Splitter | Key Parameters |
|----------|---------------------|----------------|
| Plain text documents | RecursiveCharacterTextSplitter | chunk_size=500, separators=["\n\n", "\n", " "] |
| Python code | Language.PYTHON | chunk_size=500, chunk_overlap=50 |
| Markdown documentation | Language.MARKDOWN | chunk_size=500, chunk_overlap=50 |
| HTML content | Language.HTML | chunk_size=500 |
| Variable content with clear topics | SemanticChunker | threshold strategies |
| Maximum control | CharacterTextSplitter | custom separator |

### 5.2 Implementation: Adaptive Splitter

Create `adaptive_splitter.py` that chooses the strategy based on document type:

```python
# adaptive_splitter.py
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from pathlib import Path

def get_splitter(file_path: str, use_semantic: bool = False):
    """Choose appropriate splitter based on file extension."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    # Map file types to language splitters
    language_map = {
        ".py": Language.PYTHON,
        ".js": Language.JS,
        ".ts": Language.TS,
        ".md": Language.MARKDOWN,
        ".html": Language.HTML,
        ".css": Language.CSS,
    }

    if suffix in language_map:
        # Use language-specific splitter
        return RecursiveCharacterTextSplitter.from_language(
            language=language_map[suffix],
            chunk_size=500,
            chunk_overlap=50
        )
    elif use_semantic:
        # Use semantic chunking
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        return SemanticChunker(embeddings)
    else:
        # Default recursive splitter
        return RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

# Example usage
if __name__ == "__main__":
    # Test with different file types
    test_files = ["script.py", "readme.md", "article.txt"]

    for file in test_files:
        splitter = get_splitter(file)
        print(f"File: {file} -> Splitter: {type(splitter).__name__}")
```

### 5.3 Checkpoint

**Self-Assessment:**
- [ ] You can select appropriate splitting strategy based on document type
- [ ] You understand trade-offs between different splitting approaches
- [ ] You can implement an adaptive splitter that handles multiple document types

---

## Epilogue: The Complete System

Your text splitting pipeline now supports:

| Strategy | Use Case | Example |
|----------|----------|---------|
| CharacterTextSplitter | Simple splitting by character count | Basic text processing |
| RecursiveCharacterTextSplitter | General purpose with multiple separators | Most document types |
| Language-specific | Code, Markdown, HTML | Preserves syntax structure |
| SemanticChunker | Content-aware splitting | Variable-length topics |

**Key parameters to tune:**
- `chunk_size`: Smaller = more precise retrieval, larger = more context
- `chunk_overlap`: Prevents context loss at boundaries
- `separators`: Order matters—larger first, then smaller fallbacks

**End-to-end verification:**

```python
# verify_splitter.py
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Complete pipeline: Load -> Split
loader = TextLoader("data/document.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Documents loaded: {len(documents)}")
print(f"Chunks created: {len(chunks)}")
print(f"Average chunk size: {sum(len(c.page_content) for c in chunks) // len(chunks)} chars")
print("Text splitting pipeline is working correctly!")
```

---

## The Principles

1. **Preserve semantic boundaries**: Split at natural boundaries (paragraphs, sentences, code blocks) rather than arbitrary character positions.

2. **Overlap prevents context loss**: Always include overlap to ensure context flows between chunks. Typical overlap is 10-20% of chunk size.

3. **Match splitter to content type**: Use language-specific splitters for code, semantic splitters for variable-length topics, and recursive splitters for general text.

4. **Tune chunk_size for your use case**: Smaller chunks (200-300) provide precise retrieval but less context. Larger chunks (500-1000) provide more context but may reduce precision.

5. **Test with real queries**: The best splitting strategy depends on how users will query the system. Test with representative queries and adjust accordingly.

---

## Troubleshooting

### Chunks too large for embedding model

**Cause:** chunk_size exceeds the token limit of your embedding model.

**Solution:**
```python
# Most embedding models handle 512-2048 tokens
# Assuming ~4 characters per token, use appropriate chunk_size
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)  # ~250 tokens
```

### Important context gets split across chunks

**Cause:** chunk_overlap is too small or separators are inappropriate.

**Solution:**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,  # Increase overlap
    separators=["\n\n", "\n", ". ", "! "]  # Add sentence-ending separators first
)
```

### Code chunks are unusable

**Cause:** Using generic splitter instead of language-specific.

**Solution:**
```python
from langchain_text_splitters import Language
splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=500
)
```

### Semantic chunking produces too few chunks

**Cause:** Breakpoint threshold is too high.

**Solution:**
```python
splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80  # Lower = more chunks
)
```

---

## Next Steps

After completing this lab:

1. Experiment with different chunk sizes and test retrieval quality
2. Implement a hybrid approach: semantic chunking with size limits
3. Progress to Lab 3: Vector Store to learn how to embed and store your chunks

---

## Additional Resources

- [LangChain Text Splitters Documentation](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Semantic Chunking Guide](https://python.langchain.com/docs/extras/modules/data_connection/document_transformers/semantic_chunker)
- [Token Estimation](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)