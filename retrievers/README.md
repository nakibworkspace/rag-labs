# Lab 4: Retrieval

## Introduction

This lab covers retrieval patterns, the final stage of the RAG pipeline. After storing document embeddings, you need sophisticated retrieval strategies to fetch the most relevant context for user queries. This lab teaches advanced retrieval techniques that improve both recall and precision in production RAG systems.

This lab covers multiple retrieval strategies, from basic similarity search to advanced techniques like contextual compression and multi-query retrieval.

![dia2](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/retrievers/assets/rag4-2.drawio.svg)

**Prerequisites:** Completion of Labs 1-3 (Document Loaders, Text Splitting, Vector Stores), or equivalent knowledge of loading, splitting, and storing documents.

## Learning Objectives

By the end of this lab, you will be able to:

1. Explain the difference between retrieval and generation in RAG systems
2. Implement basic and advanced retrieval patterns
3. Use MultiQueryRetriever to expand search coverage
4. Apply contextual compression to filter relevant content
5. Implement ensemble retrieval combining multiple methods
6. Build retrieval chains that connect to language models

## Prologue: The Challenge

Your RAG pipeline is complete: documents load, split into chunks, and store in a vector database. However, the research team reports persistent issues:

1. **Missed relevant content**: Similarity search sometimes fails to retrieve documents that use different terminology
2. **Too much irrelevant context**: Retrieved documents contain useful information but also much noise, wasting context window space
3. **No diverse results**: Multiple variations of the same answer appear, missing alternative perspectives
4. **Inconsistent performance**: Some queries work well, others return poor results

The team needs a more robust retrieval system that handles varied query formulations, filters out noise, and produces consistent results across different query types.

Success criteria:
- Implement at least 3 advanced retrieval strategies
- Achieve measurable improvement in retrieval quality
- Handle queries with different terminology than stored documents
- Filter retrieved content to relevant passages only

## Environment Setup

Install required packages:

```bash
pip install langchain langchain-community langchain-chroma faiss-cpu openai wikipedia
```

For this lab, you may need an OpenAI API key:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

---

## Chapter 1: Retrieval Fundamentals

Before implementing advanced patterns, establish the fundamentals of retrieval in RAG systems.

![dia1](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/retrievers/assets/rag4-1.drawio.svg)

### 1.1 What You Will Build

Retrieval in RAG works as follows:
1. User query is embedded into a vector
2. Vector store finds most similar document chunks
3. Retrieved chunks become context for the LLM
4. LLM generates answer based on retrieved context

### 1.2 Think First: Why Is Retrieval Hard?

**Question:** If semantic search finds "similar" vectors, why does retrieval still fail sometimes?

<details>
<summary>Click to review</summary>

Retrieval faces several challenges:

1. **Vocabulary mismatch**: Users use different words than documents ("heart" vs "cardiac", "car" vs "automobile")

2. **Semantic complexity**: Some concepts are hard to embed accurately (sarcasm, irony, nuanced opinions)

3. **Chunk boundary issues**: Relevant information spans chunk boundaries

4. **Context window limits**: Only top-k results can fit in context, potentially missing relevant information

5. **Embedding quality**: Embeddings trained on general text may not capture domain-specific relationships

</details>

### 1.3 Implementation: Basic Retrieval Pipeline

Create `basic_retrieval.py`:

```python
# basic_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create sample documents
documents = [
    Document(
        page_content="The heart is a muscular organ that pumps blood throughout the body. It beats around 100,000 times per day.",
        metadata={"source": "doc1", "topic": "anatomy"}
    ),
    Document(
        page_content="Cardiovascular diseases are the leading cause of death globally. Risk factors include high blood pressure and cholesterol.",
        metadata={"source": "doc2", "topic": "disease"}
    ),
    Document(
        page_content="Regular exercise strengthens the heart muscle and improves cardiovascular health. Walking is excellent for heart health.",
        metadata={"source": "doc3", "topic": "prevention"}
    ),
    Document(
        page_content="The brain controls most activities of the body. It processes information from the senses and controls movements.",
        metadata={"source": "doc4", "topic": "anatomy"}
    ),
]

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# Convert to retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Define prompt template
prompt = PromptTemplate(
    template="""Answer the question based only on the following context:

{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Create the RAG chain
llm = ChatOpenAI(model="gpt-3.5-turbo")
parser = StrOutputParser()

rag_chain = prompt | llm | parser

# Execute a query
question = "How does exercise affect heart health?"

# Step 1: Retrieve relevant documents
retrieved_docs = retriever.invoke(question)
print("=== Retrieved Documents ===")
for i, doc in enumerate(retrieved_docs):
    print(f"\n{i+1}. {doc.page_content}")
    print(f"   Metadata: {doc.metadata}")

# Step 2: Generate answer
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
answer = rag_chain.invoke({"context": context, "question": question})

print("\n=== Generated Answer ===")
print(answer)
```

### 1.4 Checkpoint

**Self-Assessment:**
- [ ] You can create a basic retrieval chain
- [ ] You can retrieve documents using a retriever
- [ ] You can pass retrieved context to an LLM
- [ ] You understand the full retrieval-to-generation flow

---

## Chapter 2: MultiQuery Retrieval

MultiQueryRetriever generates multiple versions of the query to improve recall.

![dia3](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/retrievers/assets/rag4-3.drawio.svg)

### 2.1 What You Will Build

MultiQueryRetriever uses an LLM to generate variations of the user query, retrieves documents for each variation, and combines unique results. This addresses vocabulary mismatch by covering different ways the same question might be phrased.

### 2.2 Think First: Why Generate Multiple Queries?

**Question:** If I search for "heart disease treatment", why would generating variations like "cardiac illness therapy" or "heart condition remedy" help?

<details>
<summary>Click to review</summary>

MultiQuery helps because:

1. **Synonym coverage**: Different terms map to different vectors in the embedding space

2. **Query expansion**: Generated queries explore different aspects of the topic

3. **Failure recovery**: If one query formulation fails to find relevant docs, others may succeed

4. **No manual synonym dictionaries**: The LLM automatically generates relevant variations

</details>

### 2.3 Implementation: MultiQueryRetriever

Create `multi_query.py`:

```python
# multi_query.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

# Create diverse document set
documents = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

# Create vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Create MultiQueryRetriever
llm = ChatOpenAI(model="gpt-3.5-turbo")
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    llm=llm
)

# Query
query = "How to improve energy levels and maintain balance?"

# Get results with standard retriever
standard_results = base_retriever.invoke(query)

# Get results with MultiQueryRetriever
multiquery_results = multi_query_retriever.invoke(query)

print("=== Standard Similarity Search ===")
for i, doc in enumerate(standard_results):
    print(f"\n{i+1}. {doc.page_content}")

print("\n" + "="*50)
print("=== MultiQuery Retriever ===")
for i, doc in enumerate(multiquery_results):
    print(f"\n{i+1}. {doc.page_content}")
```

### 2.4 Experiment: Comparing Recall

```python
# experiment_recall.py
# Run multiple queries and compare unique relevant documents

query = "ways to improve heart health"

# Standard search
standard_docs = base_retriever.invoke(query)

# MultiQuery
multiquery_docs = multi_query_retriever.invoke(query)

# Check unique documents
standard_sources = set([doc.metadata['source'] for doc in standard_docs])
multiquery_sources = set([doc.metadata['source'] for doc in multiquery_docs])

print(f"Standard retrieved: {len(standard_docs)} docs, sources: {standard_sources}")
print(f"MultiQuery retrieved: {len(multiquery_docs)} docs, sources: {multiquery_sources}")
print(f"Additional sources from MultiQuery: {multiquery_sources - standard_sources}")
```

**Observation:** MultiQuery typically retrieves more unique relevant documents by covering different query formulations.

### 2.5 Checkpoint

**Self-Assessment:**
- [ ] You can create a MultiQueryRetriever
- [ ] You can explain when MultiQuery improves results
- [ ] You can compare standard vs. MultiQuery retrieval
- [ ] You understand the trade-offs (more queries = more computation)

---

## Chapter 3: Contextual Compression

Sometimes retrieved documents contain relevant content mixed with noise. Contextual compression filters to keep only relevant passages.

![dia4](https://raw.githubusercontent.com/nakibworkspace/rag-labs/34d6cbad9a47f4dcaadc2a9d5f0e6692be06f76f/retrievers/assets/rag4-4.drawio.svg)

### 3.1 What You Will Build

Contextual compression uses an LLM to extract only the relevant content from each retrieved document, reducing noise and maximizing useful context.

### 3.2 Implementation: ContextualCompressionRetriever

Create `contextual_compression.py`:

```python
# contextual_compression.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# Documents with mixed relevant and irrelevant content
docs = [
    Document(page_content="""
        The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see the Grand Canyon every year.
        The rocks at the Grand Canyon date back millions of years.
    """, metadata={"source": "Doc1"}),

    Document(page_content="""
        In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal for protection.
        Siege weapons were often used to breach castle walls.
    """, metadata={"source": "Doc2"}),

    Document(page_content="""
        Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets.
        The NBA is now a global professional basketball league.
    """, metadata={"source": "Doc3"}),

    Document(page_content="""
        The history of cinema began in the late 1800s with silent films.
        Thomas Edison was among the pioneers of early cinema technology.
        Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sophisticated sound design.
    """, metadata={"source": "Doc4"})
]

# Create vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embedding_model)

# Create base retriever
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Setup compression
llm = ChatOpenAI(model="gpt-3.5-turbo")
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# Query about photosynthesis
query = "What is photosynthesis?"

print("=== Base Retrieval (includes irrelevant content) ===")
base_results = base_retriever.invoke(query)
for doc in base_results:
    print(f"\n{doc.page_content}")

print("\n" + "="*50)
print("=== Compressed Retrieval (only relevant content) ===")
compressed_results = compression_retriever.invoke(query)
for doc in compressed_results:
    print(f"\n{doc.page_content}")
```

**Observe:** The compression retriever extracts only the sentences relevant to photosynthesis, filtering out unrelated content about castles, basketball, and cinema.

### 3.3 Checkpoint

**Self-Assessment:**
- [ ] You can create a ContextualCompressionRetriever
- [ ] You can explain how compression filters relevant content
- [ ] You can compare base vs. compressed retrieval
- [ ] You understand when compression is beneficial

---

## Chapter 4: Ensemble Retrieval

Ensemble retrieval combines multiple retrieval strategies for better overall performance.

### 4.1 What You Will Build

Learn to combine different retrievers using LangChain's EnsembleRetriever to leverage the strengths of multiple approaches.

### 4.2 Implementation: Combining Retrievers

Create `ensemble_retrieval.py`:

```python
# ensemble_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_openai import ChatOpenAI

# Create document set
documents = [
    Document(page_content="Python is a high-level programming language known for simplicity.", metadata={"type": "code"}),
    Document(page_content="Snakes are reptiles that crawl on the ground.", metadata={"type": "animal"}),
    Document(page_content="Python pandas is a data analysis library.", metadata={"type": "code"}),
    Document(page_content="The python snake is a large non-venomous snake found in Asia.", metadata={"type": "animal"}),
    Document(page_content="Django is a Python web framework for building web applications.", metadata={"type": "code"}),
    Document(page_content="Web development involves creating websites and web applications.", metadata={"type": "web"}),
]

# Create vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

# Create multiple retrievers with different configurations
retriever1 = vectorstore.as_retriever(
    search_kwargs={"k": 3},
    search_type="similarity"
)

retriever2 = vectorstore.as_retriever(
    search_kwargs={"k": 3},
    search_type="mmr"
)

# Ensemble them together
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever1, retriever2],
    weights=[0.5, 0.5]  # Equal weight to similarity and MMR
)

# Test queries
queries = [
    "Tell me about the Python programming language",
    "Tell me about Python snakes"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    print("-" * 50)

    results = ensemble_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc.page_content} (type: {doc.metadata['type']})")
```

### 4.3 Implementation: Custom Weighted Ensemble

Create `weighted_ensemble.py`:

```python
# weighted_ensemble.py
from langchain_community.retrievers import ContextualCompressionRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document

# Create documents with different relevance to different query types
documents = [
    Document(page_content="Python was created by Guido van Rossum in 1991.", metadata={"topic": "history"}),
    Document(page_content="Python supports object-oriented, functional, and procedural programming.", metadata={"topic": "features"}),
    Document(page_content="JavaScript is primarily used for web development.", metadata={"topic": "javascript"}),
    Document(page_content="Python has simple syntax making it beginner-friendly.", metadata={"topic": "features"}),
    Document(page_content="Java is a compiled language that runs on the JVM.", metadata={"topic": "java"}),
]

# Note: In practice, you would create different vector stores for different aspects
# This shows the concept of weighted ensemble

print("Weighted ensemble combines scores from multiple retrievers:")
print("- Semantic retriever: Best for conceptual queries")
print("- Keyword retriever: Best for specific term matching")
print("- Ensemble: Optimal for mixed queries")
```

### 4.4 Checkpoint

- [ ] You can create an EnsembleRetriever
- [ ] You can weight different retrievers
- [ ] You understand when ensemble improves performance

---

## Chapter 5: Advanced Retrieval Patterns

Learn additional patterns for production systems.

### 5.1 Self-Query Retriever

The SelfQueryRetriever can generate its own filters from natural language queries:

```python
# self_query.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.retrievers.self_query import SelfQueryRetriever
from langchain_openai import ChatOpenAI

# Create documents with metadata
documents = [
    Document(page_content="Action movie with intense fight scenes", metadata={"genre": "action", "year": 2020}),
    Document(page_content="Romantic comedy about meeting someone special", metadata={"genre": "romance", "year": 2021}),
    Document(page_content="Sci-fi thriller in space", metadata={"genre": "sci-fi", "year": 2019}),
    Document(page_content="Comedy about a stand-up comedian", metadata={"genre": "comedy", "year": 2022}),
    Document(page_content="Horror movie set in an old mansion", metadata={"genre": "horror", "year": 2021}),
]

# Create vector store
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

# Create self-query retriever
llm = ChatOpenAI(model="gpt-3.5-turbo")
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description="Movie descriptions",
    metadata_field_info=[
        {"name": "genre", "description": "The genre of the movie", "type": "string"},
        {"name": "year", "description": "The year the movie was released", "type": "int"},
    ]
)

# Query with implicit filter
query = "What are some romantic movies from after 2020?"

results = retriever.invoke(query)
print(f"Query: '{query}'\n")
print("Results:")
for doc in results:
    print(f"  - {doc.page_content} ({doc.metadata})")
```

### 5.2 Parent Document Retriever

For large documents, retrieve parent chunks to maintain context:

```python
# parent_document.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever

# Large document
large_text = """
Machine Learning Overview

Machine learning is a subset of artificial intelligence that enables systems
to learn and improve from experience without being explicitly programmed.

Types of Machine Learning

There are three main types:

1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error

Applications

Machine learning powers many modern applications including image recognition,
natural language processing, recommendation systems, and autonomous vehicles.
""".strip()

# Split into small chunks for retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)

# Split into parent chunks (larger)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

# Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(
    [Document(page_content=large_text)],
    embeddings
)

# Create parent document retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=vectorstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    search_kwargs={"k": 2}
)

results = retriever.invoke("What is machine learning?")
print(f"Retrieved {len(results)} documents:")
for doc in results:
    print(f"  - {doc.page_content[:100]}...")
```

### 5.3 Checkpoint

- [ ] You can use SelfQueryRetriever for automatic filtering
- [ ] You understand Parent Document Retrieval for context preservation

---

## Epilogue: The Complete System

Your retrieval system now supports multiple strategies:

| Strategy | Use Case | Key Benefit |
|----------|----------|--------------|
| Similarity Search | General purpose | Simple, fast |
| MMR | Diverse results | Avoids redundancy |
| MultiQuery | Vocabulary mismatch | Covers synonyms |
| Contextual Compression | Noisy documents | Filters to relevant |
| Ensemble | Mixed queries | Combines strengths |
| Self-Query | Implicit filters | Auto-generates filters |
| Parent Document | Large documents | Maintains context |

**Choosing a Strategy:**

| If... | Use... |
|-------|--------|
| Basic needs | Similarity search |
| Need diverse results | MMR |
| Users use varied terminology | MultiQuery |
| Documents contain noise | Contextual Compression |
| Multiple query types | Ensemble |
| Natural language filters | Self-Query |
| Large documents | Parent Document |

**End-to-end verification:**

```python
# verify_retrieval.py
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

embeddings = OpenAIEmbeddings()
docs = [Document(page_content="RAG combines retrieval with generation.", metadata={"source": "test"})]
vs = FAISS.from_documents(docs, embeddings)

retriever = vs.as_retriever(search_kwargs={"k": 1})
results = retriever.invoke("What is RAG?")

print(f"Retrieved: {len(results)} documents")
print(f"Content: {results[0].page_content}")
print("Retrieval pipeline is working correctly!")
```

---

## The Principles

1. **No single best retriever**: Different strategies excel in different scenarios. Build systems that can swap strategies based on query type.

2. **Combine for robustness**: Ensemble methods typically outperform single retrievers by covering multiple aspects.

3. **Compression maximizes context**: In limited context windows, compressed relevant content beats full irrelevant documents.

4. **MultiQuery costs compute**: Generating multiple queries uses more LLM calls. Balance against retrieval quality improvements.

5. **Test with real queries**: Your retrieval strategy should be validated against actual user queries, not synthetic benchmarks.

---

## Troubleshooting

### MultiQuery returns duplicate documents

**Cause:** Different generated queries retrieve the same documents.

**Solution:** This is expected behavior. Use deduplication logic or accept duplicates for simplicity.

### Contextual compression returns empty results

**Cause:** Compressor LLM cannot extract relevant content.

**Solution:**
- Check if documents actually contain relevant information
- Try a more capable LLM model
- Use less aggressive compression

### Self-Query generates wrong filters

**Cause:** LLM misinterprets query intent.

**Solution:**
- Add more metadata field descriptions
- Provide examples of expected filters
- Use a more capable LLM

### Slow retrieval with many documents

**Cause:** FAISS and Chroma scale well but not infinitely.

**Solution:**
- Implement pagination
- Use approximate nearest neighbor (ANN) indexes
- Consider dedicated vector databases for scale

---

## Additional Resources

- [LangChain Retrieval Documentation](https://python.langchain.com/docs/modules/data_connection/retrievers/)
- [MultiQuery Retriever Guide](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/MultiQueryRetriever)
- [Contextual Compression](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/contextual_compression)
- [Ensemble Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/how_to/ensemble)