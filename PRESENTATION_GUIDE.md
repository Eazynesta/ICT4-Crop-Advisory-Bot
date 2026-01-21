# ICT4 Crop Advisory Bot - Comprehensive Technical Explanation

## A Presentation Guide for Understanding Every Aspect of the Code

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What is RAG and Why We Use It](#2-what-is-rag-and-why-we-use-it)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Detailed Code Walkthrough](#5-detailed-code-walkthrough)
6. [Data Flow Explained](#6-data-flow-explained)
7. [Key Technologies and Libraries](#7-key-technologies-and-libraries)
8. [Important Design Decisions](#8-important-design-decisions)
9. [How to Explain Each Component](#9-how-to-explain-each-component)
10. [Potential Questions and Answers](#10-potential-questions-and-answers)

---

## 1. Project Overview

### What is this project?

The **ICT4 Crop Advisory Bot** is an AI-powered chatbot designed to help Kenyan farmers get instant, reliable agricultural advice. Instead of searching through lengthy PDF documents or waiting for agricultural extension officers, farmers can simply ask questions in natural language and receive accurate, source-cited answers.

### The Problem We're Solving

- Farmers need quick access to agricultural information
- Traditional methods (extension officers, manuals) are slow or inaccessible
- Generic AI chatbots like ChatGPT can "hallucinate" (make up) incorrect farming advice
- Our solution grounds all answers in verified agricultural documents

### Key Features

| Feature | Description |
|---------|-------------|
| **Document-Grounded Responses** | Every answer comes from our agricultural PDFs, not made-up information |
| **Source Citations** | Each response tells you exactly which document and page the information came from |
| **Natural Language Interface** | Farmers can ask questions in plain English |
| **Semantic Search** | The system understands the *meaning* of questions, not just keywords |

---

## 2. What is RAG and Why We Use It

### The Core Concept: Retrieval-Augmented Generation (RAG)

**RAG** is a technique that combines two AI capabilities:

1. **Retrieval**: Finding relevant information from a knowledge base
2. **Generation**: Using a large language model (LLM) to generate human-like responses

### Why RAG Instead of Just Using ChatGPT?

| Approach | Problem |
|----------|---------|
| **Plain ChatGPT** | Can hallucinate (make up) information; doesn't have access to our specific agricultural documents |
| **Keyword Search** | Only finds exact word matches; misses relevant content with different wording |
| **RAG (Our Approach)** | Finds semantically relevant content AND generates accurate, grounded responses |

### How RAG Works (Simple Explanation)

```
[User Question]
      |
      v
[Step 1: RETRIEVAL]
"Find the 4 most relevant chunks from our agricultural documents"
      |
      v
[Step 2: AUGMENTATION]
"Add those chunks as context to the question"
      |
      v
[Step 3: GENERATION]
"GPT-4 generates an answer using ONLY the provided context"
      |
      v
[Grounded Answer with Sources]
```

### The Key Insight

> "We don't ask GPT-4 to answer from its general knowledge. We give it specific agricultural documents and say: 'Answer this question using ONLY this information.'"

---

## 3. System Architecture

### High-Level Architecture Diagram

```
+--------------------------------------------------+
|              USER INTERFACE (Streamlit)           |
|                    app.py                         |
|    [Chat Input] --> [Display] --> [History]       |
+--------------------------------------------------+
                        |
                        v
+--------------------------------------------------+
|              RAG PIPELINE (chain.py)              |
|                                                   |
|  Question --> Retriever --> LLM --> Answer        |
+--------------------------------------------------+
         |                           |
         v                           v
+------------------+      +---------------------+
| FAISS Vector     |      | Azure OpenAI        |
| Database         |      | GPT-4.1             |
| (embeddings.py)  |      | (chain.py)          |
+------------------+      +---------------------+
         |
         v
+--------------------------------------------------+
|           DOCUMENT PROCESSING                     |
|              (document_loader.py)                 |
|                                                   |
|  PDFs --> Extract Text --> Split into Chunks      |
+--------------------------------------------------+
```

### Two Phases of Operation

**Phase 1: Offline Indexing (Run Once)**
- Load PDF documents
- Split them into chunks
- Convert chunks to vector embeddings
- Store in FAISS database

**Phase 2: Online Query (Every User Question)**
- User asks a question
- Find relevant chunks from FAISS
- Send chunks + question to GPT-4
- Return grounded answer

---

## 4. Project Structure

```
ICT4-Crop-Advisory-Bot/
|
├── app.py                    # Main web application (what users see)
|
├── src/                      # Core logic modules
│   ├── document_loader.py    # Loads and splits PDF documents
│   ├── embeddings.py         # Creates and manages the vector database
│   ├── chain.py              # Orchestrates the RAG pipeline
│   └── __init__.py           # Makes src a Python package
|
├── data/
│   ├── pdfs/                 # Source agricultural documents
│   │   ├── farming-guide.pdf
│   │   ├── farm-management-handbook.pdf
│   │   └── Compilation_techniques_organic_agriculture_rev.pdf
│   |
│   └── faiss_index/          # Pre-computed vector database
│       ├── index.faiss       # The actual vector index
│       └── index.pkl         # Metadata (document sources, pages)
|
├── requirements.txt          # Python dependencies
├── .env                      # API credentials (not in git)
└── README.md                 # Quick start guide
```

---

## 5. Detailed Code Walkthrough

### 5.1 document_loader.py - "The Document Processor"

**Purpose**: Load PDFs and split them into manageable chunks

#### Function 1: `load_pdfs(pdf_directory)`

```python
def load_pdfs(pdf_directory: str) -> list:
    """
    What it does:
    1. Looks in the specified folder for PDF files
    2. Uses PyPDFLoader to extract text from each PDF
    3. Returns a list of Document objects (text + metadata)
    """
```

**Key Points to Explain**:
- Each PDF page becomes a separate Document object
- Metadata (filename, page number) is preserved for citations later
- Error handling ensures one bad PDF doesn't crash the whole system

#### Function 2: `split_documents(documents, chunk_size=1000, chunk_overlap=200)`

```python
def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    What it does:
    1. Takes the full documents
    2. Splits them into smaller chunks of ~1000 characters
    3. Keeps 200-character overlap between chunks
    """
```

**Why We Split Documents**:
- LLMs have context limits (can only process so much text at once)
- Smaller chunks = more precise retrieval
- Overlap ensures we don't lose information at chunk boundaries

**The Splitting Strategy (RecursiveCharacterTextSplitter)**:
```
Tries to split on: "\n\n" (paragraphs) first
If chunk still too big: "\n" (lines)
If still too big: " " (words)
Last resort: "" (characters)
```

This preserves semantic meaning by respecting natural text boundaries.

---

### 5.2 embeddings.py - "The Vector Database Manager"

**Purpose**: Convert text chunks into vectors and manage the FAISS database

#### What are "Embeddings"?

Embeddings are numerical representations of text that capture semantic meaning:

```
"How do I control pests?" --> [0.234, -0.891, 0.445, ..., 0.123]  (1536 numbers)
"pest management methods" --> [0.231, -0.889, 0.442, ..., 0.125]  (similar numbers!)
"favorite pizza toppings" --> [0.891, 0.234, -0.667, ..., 0.999]  (very different!)
```

Similar meanings = similar vectors = close together in vector space

#### Function 1: `get_embeddings()`

```python
def get_embeddings() -> AzureOpenAIEmbeddings:
    """
    Creates a connection to Azure OpenAI's embedding model.
    We use 'text-embedding-3-small' which produces 1536-dimensional vectors.
    """
```

#### Function 2: `create_vector_store(chunks)`

```python
def create_vector_store(chunks: list) -> FAISS:
    """
    What it does:
    1. Takes all document chunks
    2. Converts each chunk to a 1536-dimensional vector
    3. Stores vectors in a FAISS index for fast similarity search
    4. Saves the index to disk
    """
```

**Critical Feature - Batch Processing**:
```python
BATCH_SIZE = 50      # Process 50 chunks at a time
BATCH_DELAY = 5      # Wait 5 seconds between batches
```

Why? Azure has rate limits. Without batching, we'd get "429 Too Many Requests" errors.

#### Function 3: `load_vector_store()`

```python
def load_vector_store() -> FAISS:
    """
    Loads the pre-computed FAISS index from disk.
    Much faster than re-creating it every time!
    """
```

#### What is FAISS?

**FAISS** (Facebook AI Similarity Search) is a library for efficient similarity search:
- Can search millions of vectors in milliseconds
- Uses clever data structures and algorithms
- Perfect for finding "which chunks are most similar to this question?"

---

### 5.3 chain.py - "The Brain/Orchestrator"

**Purpose**: Ties everything together into a working RAG pipeline

#### Function 1: `get_llm()`

```python
def get_llm() -> AzureChatOpenAI:
    """
    Creates a connection to Azure OpenAI's GPT-4.1 model.

    Key setting: temperature=0.3 (low)
    - Higher temperature = more creative/random
    - Lower temperature = more factual/consistent
    - We use 0.3 because we want accurate agricultural advice, not creative writing
    """
```

#### Function 2: `format_docs(docs)`

```python
def format_docs(docs: list) -> str:
    """
    Takes the retrieved document chunks and formats them nicely.

    Output format:
    [Source 1 - farming-guide.pdf, Page 42]
    <actual content from the chunk>

    ---

    [Source 2 - farm-management-handbook.pdf, Page 15]
    <actual content from the chunk>
    """
```

This is crucial for **source attribution** - users know exactly where information came from.

#### Function 3: `create_rag_chain()` - THE MOST IMPORTANT FUNCTION

```python
def create_rag_chain():
    """
    This is where the magic happens. Creates the complete RAG pipeline.
    """
```

**Step-by-Step Breakdown**:

**Step 1: Load the Vector Store**
```python
vector_store = load_vector_store()
retriever = vector_store.as_retriever(search_kwargs={"k": 4})
```
- `k=4` means "retrieve the 4 most similar chunks"

**Step 2: Create the Prompt Template**
```python
prompt = ChatPromptTemplate.from_template("""
You are an expert agricultural advisor for Kenyan farmers...

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the context provided
2. If the context doesn't contain enough information, say so
3. Always mention which source(s) you used
4. Give practical, actionable advice
5. Use simple language

CONTEXT FROM AGRICULTURAL DOCUMENTS:
{context}

---
FARMER'S QUESTION: {question}

HELPFUL ANSWER:
""")
```

**This prompt is critical** - it instructs GPT-4 to:
- ONLY use the provided context (no hallucination)
- Cite sources
- Give practical advice
- Use simple language for farmers

**Step 3: Build the Chain (LangChain Expression Language)**
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Let me break this down:

```
Question: "How do I control locusts?"
                |
                v
    +-----------+-----------+
    |                       |
    v                       v
[retriever]           [RunnablePassthrough]
    |                       |
    v                       v
[format_docs]         "How do I control locusts?"
    |                       |
    v                       v
"[Source 1...]        "How do I control locusts?"
content about         (unchanged)
locusts..."
    |                       |
    +----------+------------+
               |
               v
         [prompt template]
               |
               v
    Full prompt with context + question
               |
               v
            [llm]
               |
               v
       GPT-4 response
               |
               v
      [StrOutputParser]
               |
               v
    Final string answer
```

---

### 5.4 app.py - "The User Interface"

**Purpose**: Create the web-based chat interface using Streamlit

#### Key Components:

**1. Page Configuration**
```python
st.set_page_config(page_title="Agriculture Crop Advisory Bot", page_icon="...")
```

**2. Cached Chain Initialization**
```python
@st.cache_resource
def get_chain():
    return create_rag_chain()
```

`@st.cache_resource` means: "Only create the chain once, then reuse it for all requests"

**3. Session State for Chat History**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

This keeps track of the conversation so users can see their chat history.

**4. Chat Interface**
```python
if prompt := st.chat_input("Ask a question about farming..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get AI response
    with st.spinner("Searching agricultural knowledge base..."):
        response = rag_chain.invoke(prompt)

    # Add AI response to history
    st.session_state.messages.append({"role": "assistant", "content": response})
```

---

## 6. Data Flow Explained

### Phase 1: Document Indexing (Run Once)

```
[PDF Files]
     |
     | PyPDFLoader extracts text
     v
[Raw Documents with Metadata]
     |
     | RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
     v
[~277 Text Chunks]
     |
     | Azure OpenAI text-embedding-3-small
     v
[~277 Vectors (1536 dimensions each)]
     |
     | FAISS indexing
     v
[FAISS Index saved to disk]
```

### Phase 2: Query Processing (Every Question)

```
[User Question: "How do I control pests?"]
     |
     | Azure OpenAI converts to vector
     v
[Question Vector: [0.234, -0.891, ...]]
     |
     | FAISS similarity search (cosine similarity)
     v
[Top 4 Most Similar Chunks]
     |
     | format_docs adds citations
     v
[Formatted Context with Sources]
     |
     | Inserted into prompt template
     v
[Complete Prompt to GPT-4]
     |
     | GPT-4.1 generates response
     v
[Grounded Answer with Citations]
     |
     | Displayed to user
     v
[User sees helpful farming advice!]
```

---

## 7. Key Technologies and Libraries

### LangChain
**What it is**: A framework for building applications with LLMs
**Why we use it**:
- Provides pre-built components (loaders, splitters, chains)
- Makes it easy to connect different AI services
- Handles the complex orchestration for us

### FAISS (Facebook AI Similarity Search)
**What it is**: A library for efficient vector similarity search
**Why we use it**:
- Extremely fast (searches millions of vectors in milliseconds)
- No server needed (runs locally)
- Battle-tested by Meta/Facebook

### Streamlit
**What it is**: A Python framework for building web apps
**Why we use it**:
- Simple to use (just Python, no HTML/CSS/JS needed)
- Great for data science and AI applications
- Built-in chat components

### Azure OpenAI
**What it is**: Microsoft's hosted version of OpenAI models
**Why we use it**:
- Enterprise-grade security and reliability
- Same models as OpenAI (GPT-4, embeddings)
- Better for production applications

---

## 8. Important Design Decisions

### Decision 1: Temperature = 0.3

**What**: LLM temperature controls randomness/creativity
**Our choice**: 0.3 (relatively low)
**Why**: Agricultural advice needs to be accurate and consistent, not creative

### Decision 2: Chunk Size = 1000, Overlap = 200

**What**: How we split documents into pieces
**Our choice**: 1000 characters with 200-character overlap
**Why**:
- 1000 chars is enough context without being too long
- 200-char overlap prevents losing information at boundaries
- Balanced retrieval precision with context completeness

### Decision 3: k = 4 (Retrieve 4 Chunks)

**What**: How many document chunks to retrieve per question
**Our choice**: 4 chunks
**Why**:
- Enough context for comprehensive answers
- Not so much that we overwhelm the LLM
- Good balance between coverage and relevance

### Decision 4: Batch Processing with Rate Limits

**What**: Process embeddings 50 at a time with 5-second delays
**Why**: Azure OpenAI has rate limits; without batching, we get errors

---

## 9. How to Explain Each Component

### When Explaining document_loader.py:

> "This module handles the first step of our pipeline. It takes our agricultural PDF documents and converts them into a format our system can work with. First, it extracts the text from each PDF page using PyPDFLoader. Then, it splits the text into smaller chunks of about 1000 characters. We use overlapping chunks (200 characters) to make sure we don't lose important information that might span two chunks. This is similar to how you might highlight important passages in a textbook - we're essentially creating thousands of small, searchable passages from our documents."

### When Explaining embeddings.py:

> "This module creates our searchable knowledge base. The key concept here is 'embeddings' - we convert each text chunk into a list of 1536 numbers that represent its meaning. Similar texts get similar numbers. We use Azure OpenAI's embedding model for this conversion, then store all these vectors in a FAISS database. FAISS is a library from Facebook that can search through millions of vectors in milliseconds. When a user asks a question, we convert their question to a vector and find the most similar document chunks - that's how we find relevant information."

### When Explaining chain.py:

> "This is the brain of our system - it orchestrates everything. When a user asks a question, this module: (1) uses the FAISS database to find the 4 most relevant document chunks, (2) formats those chunks with source citations, (3) creates a carefully crafted prompt that instructs GPT-4 to answer using ONLY the provided context, and (4) sends everything to GPT-4 and returns the response. The key innovation is that we're not just asking GPT-4 to answer from its general knowledge - we're giving it specific, verified information and saying 'use only this to answer.'"

### When Explaining app.py:

> "This is the user interface built with Streamlit. It provides a familiar chat interface where farmers can type questions and see responses. The important features are: session state that maintains conversation history, caching that prevents the system from reinitializing on every message, and a clean UI with a sidebar for additional information. It's designed to be simple and intuitive for farmers who may not be tech-savvy."

---

## 10. Potential Questions and Answers

### Q: "Why not just use ChatGPT directly?"

**A**: ChatGPT has two problems for our use case: (1) It doesn't have access to our specific agricultural documents, and (2) It can "hallucinate" - confidently give incorrect information. Our RAG approach ensures every answer comes from verified agricultural documents, and we can cite exactly where the information came from.

### Q: "What happens if the documents don't contain relevant information?"

**A**: Our prompt explicitly instructs GPT-4 to say "I don't have enough information to answer this" if the retrieved context isn't relevant. This prevents the system from making up answers.

### Q: "Why FAISS instead of a traditional database?"

**A**: Traditional databases use keyword matching - they'd only find documents with the exact words in the query. FAISS uses semantic search - it understands meaning. So "pest control" would also match documents about "insect management" even if those exact words aren't used.

### Q: "How accurate is the system?"

**A**: The system is as accurate as the source documents. Since we use verified agricultural guides from reliable sources, and the system only answers from these documents, the accuracy is high. The low temperature setting (0.3) also ensures consistent, factual responses.

### Q: "Can this scale to more documents?"

**A**: Yes! FAISS is designed to handle millions of vectors. We could add hundreds more agricultural documents without significant performance impact. The batch processing we implemented also handles rate limits during indexing.

### Q: "Why Azure OpenAI instead of OpenAI directly?"

**A**: Azure OpenAI provides enterprise-grade features: better security, reliability, and compliance. For a production application serving Kenyan farmers, this infrastructure is more appropriate than a direct API connection.

---

## Quick Reference Card

| Component | File | One-Line Purpose |
|-----------|------|------------------|
| Document Loading | document_loader.py | Extract and chunk PDFs |
| Vector Database | embeddings.py | Create searchable embeddings |
| RAG Pipeline | chain.py | Orchestrate retrieval + generation |
| User Interface | app.py | Chat interface for farmers |

| Technology | Purpose |
|------------|---------|
| LangChain | LLM orchestration framework |
| FAISS | Fast vector similarity search |
| Streamlit | Web interface |
| Azure OpenAI | LLM and embedding APIs |
| PyPDF | PDF text extraction |

| Key Number | What It Means |
|------------|---------------|
| 1000 | Characters per chunk |
| 200 | Overlap between chunks |
| 1536 | Dimensions in each embedding vector |
| 4 | Number of chunks retrieved per query |
| 0.3 | LLM temperature (low = factual) |
| 50 | Batch size for embedding requests |

---

**Good luck with your presentation!**

*Remember: The key message is that we built a system that gives farmers accurate, source-cited agricultural advice by combining semantic search (finding relevant information) with language generation (creating helpful responses), while preventing AI hallucination by grounding all answers in verified documents.*
