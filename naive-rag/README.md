# Hands-On Lab: Building a Naive RAG API

## Introduction

This lab teaches the fundamentals of Retrieval-Augmented Generation (RAG) by building a working API from scratch. You will create a system that loads PDF documents, chunks them into embeddings, stores them in a vector database, and uses a large language model to answer questions based on retrieved context.

Unlike typical tutorials that provide complete code to copy, this lab uses scaffolded exercises where you complete each component. You will predict outputs, experiment with intentional failures, and verify your understanding at each checkpoint.

**Prerequisites:** Basic Python knowledge, familiarity with REST APIs, and understanding of vector embeddings concepts.

---

## Learning Objectives

By the end of this lab, you will be able to:

1. Load and process PDF documents using LangChain document loaders
2. Split documents into chunks using text splitters
3. Create vector embeddings and store them in Chroma database
4. Implement retrieval with similarity search
5. Build a RAG pipeline that combines retrieval with LLM generation
6. Configure HuggingFace endpoints for both embeddings and LLM inference
7. Test RAG output quality using evaluation frameworks

---

## Prologue: The Challenge

You join a data science team at a research organization. Your colleagues have compiled extensive research papers on Reinforcement Learning into a PDF (RL.pdf), but finding relevant information takes hours of manual reading.

Your task is to build a RAG API that:
- Loads the PDF document
- Chunks text into semantically meaningful pieces
- Creates embeddings for semantic search
- Retrieves relevant context for user queries
- Generates accurate answers using an LLM

The system should answer questions like "What is Reinforcement Learning?" by retrieving the most relevant sections and passing them to the language model.

---

## Environment Setup

### N.1 Verify Python and Install Dependencies

Check your Python version:

```bash
python3 --version
```

Expected output: Python 3.9 or higher.

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install required packages:

```bash
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain-openai langchain-text-splitters langchain-core
pip install langchain-huggingface
pip install chromadb
pip install python-dotenv
pip install pypdf
pip install huggingface-hub
pip install pytest
pip install deepeval
```

### N.2 Configure Environment Variables

Create the `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add your HuggingFace token:

```
HUGGINGFACEHUB_API_TOKEN=your-actual-token-here
```

### N.3 Obtain Sample Document

Place your `RL.pdf` file in the project directory. Verify it exists:

```bash
ls -la RL.pdf
```

---

## Chapter 1: Document Loading and Text Splitting

### 1.1 Opening Context

The RAG pipeline begins with loading source documents. Raw PDFs contain unstructured text that must be parsed and broken into manageable chunks. If chunks are too large, they may exceed context windows or include irrelevant information. If too small, they lose semantic coherence.

This chapter covers loading PDFs and splitting text into optimal chunks for embedding.

### 1.2 Think First: Chunk Size Trade-offs

**Question:** Consider chunk_size=500 vs chunk_size=2000. Which would you choose for a technical research document? Why?

<details>
<summary>Click to review</summary>

Larger chunks (2000) preserve more semantic context but may exceed token limits or dilute relevant information with noise. Smaller chunks (500) are precise but may lose inter-sentence context. For technical documents with well-defined sections, 1000 with 200 overlap provides balance—enough context to be meaningful while staying within typical embedding limits.

</details>

### 1.3 Implementation

Complete the document processor in `main.py`:

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "RL.pdf"

def process_pdf(path):
    """Load PDF and split into chunks."""
    print(f"Loading {path}")
    loader = ___  # Q1: Which PyPDFLoader parameter?
    documents = loader.___()  # Q2: What method loads the PDF?
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=___,  # Q3: What size balances context and precision?
        chunk_overlap=___  # Q4: What overlap preserves continuity?
    )
    return ___.___  # Q5: What method returns split documents?
```

**Hints:**
- Q1: The path to the PDF file
- Q2: Method name for loading (singular form)
- Q3: Recommended size for technical documents
- Q4: Recommended overlap for continuity
- Q5: Method that performs the splitting

<details>
<summary>Click to see solution</summary>

```python
def process_pdf(path):
    """Load PDF and split into chunks."""
    print(f"Loading {path}")
    loader = PyPDFLoader(path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)
```

</details>

### 1.4 Understanding the Code

**Question:** What happens if chunk_overlap=0? Why is overlap necessary?

<details>
<summary>Click to review</summary>

Without overlap, semantic boundaries could split related content across chunks, losing continuity. Overlap ensures some context repeats between chunks, preserving semantic coherence at boundaries.

</details>

### 1.5 Test and Verify

Test your implementation:

```bash
python3 -c "
from main import process_pdf
chunks = process_pdf('RL.pdf')
print(f'Loaded {len(chunks)} chunks')
print(f'First chunk preview: {chunks[0].page_content[:200]}...')
"
```

**Predict:** How many chunks will a 50-page PDF produce approximately?

<details>
<summary>Click to verify</summary>

Typically 80-150 chunks depending on content density. The exact number varies based on text density per page.

</details>

### 1.6 Checkpoint

**Self-Assessment:**
- [ ] PDF loads without errors
- [ ] Chunks are created (verify with print statement)
- [ ] You can explain why chunk_size=1000 was chosen
- [ ] You can predict chunk count for different PDF sizes

### 1.7 Experiment: Chunk Size Impact

1. Change chunk_size to 500 and rerun
2. Change chunk_size to 2000 and rerun

**Observe:** How does chunk count change?

**Question:** What happens to answer quality if chunks are too small vs too large?

<details>
<summary>Click to review</summary>

Too small: Lost context, incomplete answers. Too large: Irrelevant information included, potential token limits hit. Finding the right balance requires experimentation.

</details>

---

## Chapter 2: Vector Store Creation

### 2.1 Opening Context

To search documents by meaning rather than keywords, we convert text chunks into vector embeddings—numerical representations that capture semantic meaning. A vector database stores these embeddings and enables similarity search.

This chapter creates the embedding pipeline and vector store.

### 2.2 Think First: Embedding Models

**Question:** Why use "sentence-transformers/all-MiniLM-L6-v2" instead of a larger model?

<details>
<summary>Click to review</summary>

This model balances quality and speed. It's optimized for semantic search while being computationally efficient. Larger models provide marginal improvement for retrieval tasks but increase latency significantly.

</details>

### 2.3 Implementation

Complete the vector store function:

```python
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
import os

CHROMA_PATH = "chroma_db"

def get_vector_store(chunks):
    """Create vector store from document chunks."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found")
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=___,
        model=___  # Q1: Which model for semantic search?
    )
    
    return Chroma.from_documents(
        documents=___,
        embedding=___,
        persist_directory=___
    )
```

**Hints:**
- Q1: The model name for embeddings

<details>
<summary>Click to see solution</summary>

```python
def get_vector_store(chunks):
    """Create vector store from document chunks."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found")
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=hf_token,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
```

</details>

### 2.4 Understanding the Code

**Question:** What does persist_directory do? What happens if you change it?

<details>
<summary>Click to review</summary>

persist_directory specifies where Chroma stores the database. Changing it creates a new empty database. Keeping the same path reuses existing embeddings—useful during development to avoid recomputation.

</details>

### 2.5 Test and Verify

```bash
python3 -c "
from main import process_pdf, get_vector_store
chunks = process_pdf('RL.pdf')
vector_db = get_vector_store(chunks)
print(f'Vector store created with {vector_db._collection.count()} documents')
"
```

**Predict:** What method retrieves documents by similarity?

<details>
<summary>Click to verify</summary>

The `as_retriever()` method creates a retriever object for similarity search.

</details>

### 2.6 Checkpoint

**Self-Assessment:**
- [ ] Vector store creates without errors
- [ ] You can retrieve documents by query
- [ ] You can explain HuggingFaceEndpointEmbeddings role
- [ ] You can predict what happens with invalid tokens

---

## Chapter 3: Retrieval Configuration

### 3.1 Opening Context

Not all retrieved documents are equally relevant. The retriever configuration determines how many chunks to retrieve and which ones. Too few chunks provide insufficient context; too many dilute relevance with noise.

This chapter configures retrieval parameters.

### 3.2 Think First: Retrieval Count

**Question:** If top_k=5 vs top_k=1, which would produce more accurate answers?

<details>
<summary>Click to review</summary>

More context generally helps, but diminishing returns occur. k=3-5 works well for most questions. Too many chunks may include irrelevant information that confuses the LLM.

</details>

### 3.3 Implementation

Create the retriever with configuration:

```python
def setup_rag():
    """Initialize RAG components."""
    global _retriever
    
    chunks = process_pdf(PDF_PATH)
    vector_db = get_vector_store(chunks)
    
    _retriever = vector_db.as_retriever(
        search_kwargs={"k": ___}  # Q1: How many chunks to retrieve?
    )
    
    print(f"Retriever configured with k={___}")
```

**Hints:**
- Q1: Recommended value for balanced retrieval

<details>
<summary>Click to see solution</summary>

```python
def setup_rag():
    """Initialize RAG components."""
    global _retriever
    
    chunks = process_pdf(PDF_PATH)
    vector_db = get_vector_store(chunks)
    
    _retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    print(f"Retriever configured with k=3")
```

</details>

### 3.4 Test and Verify

```bash
python3 -c "
from main import setup_rag
setup_rag()
docs = main._retriever.invoke('What is reinforcement learning?')
print(f'Retrieved {len(docs)} documents')
for i, d in enumerate(docs):
    print(f'Doc {i}: {d.page_content[:100]}...')
"
```

### 3.5 Checkpoint

**Self-Assessment:**
- [ ] Retriever returns relevant documents
- [ ] You can configure k parameter
- [ ] You can explain why k=3 is optimal

---

## Chapter 4: LLM Integration

### 4.1 Opening Context

The retriever finds relevant context, but the language model generates the final answer. We use HuggingFace's inference API to access models without local deployment.

This chapter integrates the LLM with the RAG pipeline.

### 4.2 Think First: API vs Local Models

**Question:** Why use HuggingFace inference API instead of running Llama locally?

<details>
<summary>Click to review</summary>

Inference APIs provide access to large models without requiring GPU resources. The tradeoff is latency and API limits vs. hardware costs. For learning and development, APIs are more accessible.

</details>

### 4.3 Implementation

Complete the LLM loader:

```python
from langchain_openai import ChatOpenAI

LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def load_hf_api_llm():
    """Connect to HuggingFace inference API."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    return ChatOpenAI(
        model=___,
        openai_api_key=___,
        openai_api_base=___,
        max_tokens=___,
        temperature=___
    )
```

**Hints:**
- Q1: The model identifier
- Q2: API token from environment
- Q3: Base URL for HF inference
- Q4: Max tokens to generate
- Q5: Temperature for creativity vs precision

<details>
<summary>Click to see solution</summary>

```python
def load_hf_api_llm():
    """Connect to HuggingFace inference API."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    return ChatOpenAI(
        model=LLM_MODEL_ID,
        openai_api_key=hf_token,
        openai_api_base="https://router.huggingface.co/v1",
        max_tokens=512,
        temperature=0.1
    )
```

</details>

### 4.4 Understanding the Code

**Question:** Why temperature=0.1 instead of 0 or 1.0?

<details>
<summary>Click to review</summary>

Temperature controls randomness. 0 = deterministic (same prompt = same output). 1.0 = very creative/unpredictable. 0.1 provides focused answers while allowing slight variation—ideal for factual question answering.

</details>

### 4.5 Test and Verify

```bash
python3 -c "
from main import load_hf_api_llm
llm = load_hf_api_llm()
response = llm.invoke('What is 2+2?')
print(f'Response: {response.content}')
"
```

### 4.6 Checkpoint

**Self-Assessment:**
- [ ] LLM connects successfully
- [ ] You can explain temperature parameter
- [ ] You can predict output differences across temperatures

---

## Chapter 5: Building the RAG Pipeline

### 5.1 Opening Context

The complete RAG pipeline combines retrieval and generation. The retriever finds relevant context, formats it into a prompt, and passes both to the LLM. The prompt template structures this interaction.

This chapter builds the complete pipeline.

### 5.2 Think First: Prompt Design

**Question:** Why include both context and question in the prompt?

<details>
<summary>Click to review</summary>

Without context, the LLM has no source information to base answers on. Without the question, the LLM doesn't know what to answer. The prompt structure must clearly separate these elements.

</details>

### 5.3 Implementation

Complete the RAG chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

_rag_chain = None

def setup_rag():
    """Build complete RAG pipeline."""
    global _rag_chain
    
    # Initialize components
    chunks = process_pdf(PDF_PATH)
    vector_db = get_vector_store(chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = load_hf_api_llm()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """Context: {context}
        
Question: {input}

Answer:"""
    )
    
    # Build RAG chain
    _rag_chain = (
        {"context": ___ | ___ , "input": ___ }  # Q1: Complete the chain
        | prompt
        | llm
        | StrOutputParser()
    )
```

**Hints:**
- Use retriever with lambda to extract page_content
- Use RunnablePassthrough for input

<details>
<summary>Click to see solution</summary>

```python
def setup_rag():
    """Build complete RAG pipeline."""
    global _rag_chain
    
    # Initialize components
    chunks = process_pdf(PDF_PATH)
    vector_db = get_vector_store(chunks)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    llm = load_hf_api_llm()
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """Context: {context}
        
Question: {input}

Answer:"""
    )
    
    # Build RAG chain
    _rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
```

</details>

### 5.4 Understanding the Code

**Question:** What does the lambda function do?

<details>
<summary>Click to review</summary>

The retriever returns document objects. The lambda extracts the page_content from each document and joins them with newlines, creating a single context string for the prompt.

</details>

### 5.5 Test and Verify

```bash
python3 -c "
from main import setup_rag, run_rag_query
setup_rag()
result = run_rag_query('What is Reinforcement Learning?')
print(f'Answer: {result}')
"
```

### 5.6 Checkpoint

**Self-Assessment:**
- [ ] Pipeline executes end-to-end
- [ ] Answer incorporates retrieved context
- [ ] You can explain each chain component
- [ ] You can modify prompt template

---

## Chapter 6: Testing with DeepEval

### 6.1 Opening Context

RAG systems require quality assurance beyond basic testing. DeepEval provides LLM-based evaluation to measure answer relevance, groundedness, and accuracy.

This chapter adds evaluation to the pipeline.

### 6.2 Think First: Evaluation Metrics

**Question:** Why use an LLM to evaluate LLM outputs?

<details>
<summary>Click to review</summary>

Exact match metrics miss semantic correctness. An LLM judge can evaluate whether answers actually address the question and use relevant context—closer to human assessment.

</details>

### 6.3 Implementation

Complete the test file:

```python
import pytest
import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from main import run_rag_query

load_dotenv()

class HuggingFaceDeepEvalModel(DeepEvalBaseLLM):
    """Custom class to let DeepEval use Hugging Face for evaluation"""
    def __init__(self, model_name):
        self.model_name = model_name

    def load_model(self):
        return ChatOpenAI(
            model=self.model_name,
            openai_api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
            openai_api_base="https://router.huggingface.co/v1",
        )

    def generate(self, prompt: str) -> str:
        model = self.load_model()
        return model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        model = self.load_model()
        res = await model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name

def test_answer_relevancy():
    query = ___  # Q1: What question to test?
    
    # Run your actual RAG pipeline
    answer = run_rag_query(query)
    
    # Setup the HF model as the Evaluator
    hf_judge = HuggingFaceDeepEvalModel(
        model_name=___
    )
    
    # Define the Test Case
    test_case = LLMTestCase(
        input=___,
        actual_output=___
    )
    
    # Define the metric
    metric = AnswerRelevancyMetric(
        threshold=___,
        model=___
    )
    
    # Run evaluation
    assert_test(test_case, [metric])
```

**Hints:**
- Q1: A question about Reinforcement Learning
- Q2: The same model used in main.py
- Q3: Threshold for passing (0.0-1.0)

<details>
<summary>Click to see solution</summary>

```python
def test_answer_relevancy():
    query = "What is Reinforcement Learning?"
    
    # Run your actual RAG pipeline
    answer = run_rag_query(query)
    
    # Setup the HF model as the Evaluator
    hf_judge = HuggingFaceDeepEvalModel(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct"
    )
    
    # Define the Test Case
    test_case = LLMTestCase(
        input=query,
        actual_output=answer
    )
    
    # Define the metric
    metric = AnswerRelevancyMetric(
        threshold=0.5,
        model=hf_judge
    )
    
    # Run evaluation
    assert_test(test_case, [metric])
```

</details>

### 6.4 Test and Verify

Run the evaluation:

```bash
pytest test_rag.py -v -s
```

**Predict:** What does a threshold of 0.5 mean?

<details>
<summary>Click to verify</summary>

A score below 0.5 fails the test; 0.5+ passes. Higher thresholds require more relevant answers.

</details>

### 6.6 Checkpoint

**Self-Assessment:**
- [ ] Test runs without errors
- [ ] You can interpret the score
- [ ] You can explain why evaluation matters

---

## Epilogue: The Complete System

Your complete RAG API now provides:

| Component | Function |
|----------|----------|
| process_pdf() | Loads PDF and splits into chunks |
| get_vector_store() | Creates embeddings in Chroma |
| load_hf_api_llm() | Connects to HuggingFace LLM |
| setup_rag() | Builds the RAG pipeline |
| run_rag_query() | Executes end-to-end RAG |

### End-to-End Verification

```bash
python3 -c "
from main import run_rag_query

queries = [
    'What is Reinforcement Learning?',
    'How does Q-learning work?',
    'What are the applications of RL?'
]

for query in queries:
    print(f'\\nQuery: {query}')
    answer = run_rag_query(query)
    print(f'Answer: {answer[:200]}...')
"
```

---

## The Principles

1. **Chunk strategically** — Balance context preservation with semantic coherence
2. **Embed once, retrieve many** — Compute embeddings once, reuse for all queries
3. **Configure retrieval** — k=3 provides optimal context for most questions
4. **Structure prompts clearly** — Separate context from question in the prompt
5. **Evaluate with LLMs** — Use LLM-based metrics for semantic assessment
6. **Temperature affects precision** — Lower temperature for factual questions

---

## Troubleshooting

### Error: FileNotFoundError: RL.pdf not found

**Cause:** PDF file not in current directory.

**Solution:**
```bash
ls -la  # Verify RL.pdf exists
# Or use absolute path
```

### Error: ModuleNotFoundError

**Cause:** Packages not installed.

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Error: HUGGINGFACEHUB_API_TOKEN not found

**Cause:** Token not in .env file.

**Solution:**
```bash
cp .env.example .env
# Edit .env and add your token
```

### Error: Rate limit exceeded

**Cause:** Too many API requests.

**Solution:**
- Wait before retrying
- Consider caching retrievals
- Use smaller models

---

## Next Steps

1. **Add web scraping** — Integrate web document loaders for live data
2. **Implement hybrid search** — Combine vector and keyword search
3. **Add reranking** — Use cross-encoder for better relevance
4. **Build FastAPI service** — Expose as HTTP endpoint
5. **Add chat history** — Implement conversation context

---

## Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs)
- [Chroma Documentation](https://docs.trychroma.com)
- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- [DeepEval Documentation](https://docs.deepeval.com)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag)

---

## Pre-Publish Checklist

- [ ] All code tested and working
- [ ] Commands produce documented output
- [ ] Prerequisites accurately stated
- [ ] Every chapter has Think First section
- [ ] Fill-in-the-blank exercises in implementation
- [ ] Solutions in collapsible sections
- [ ] Self-assessment at each checkpoint
- [ ] No emojis or prohibited phrases
- [ ] Consistent terminology throughout
- [ ] Troubleshooting covers common errors