# ICT4-Crop-Advisory-Bot
# 🌾 Agriculture Crop Advisory Bot

An AI-powered chatbot that provides instant, accurate agricultural advice to Kenyan farmers. Built using Retrieval-Augmented Generation (RAG) to ensure responses are grounded in official agricultural documents.

## 📋 Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Adding New Documents](#adding-new-documents)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Document-Grounded Responses**: All answers are sourced from official agricultural PDFs
- **Source Citations**: Every response includes references to specific documents and pages
- **Semantic Search**: Uses vector embeddings to find the most relevant information
- **Simple Web Interface**: Easy-to-use chat interface built with Streamlit
- **Incremental Indexing**: Add new documents without re-processing existing ones

## 🔧 How It Works

This bot uses **Retrieval-Augmented Generation (RAG)**:

1. **Indexing**: Agricultural PDFs are split into chunks and converted to vector embeddings
2. **Retrieval**: When you ask a question, the system finds the most relevant chunks using semantic search
3. **Generation**: The LLM generates a helpful answer based on the retrieved context

## 📦 Prerequisites

- **Python 3.10+** (tested on Python 3.14)
- **Azure OpenAI Account** with the following deployments:
  - An embeddings model (e.g., `text-embedding-3-small`)
  - A chat model (e.g., `gpt-4.1`, `gpt-4o`, or `gpt-35-turbo`)

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/agri-advisory-bot.git
cd agri-advisory-bot
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Required Directories

```bash
mkdir -p data/pdfs data/faiss_index
```

## ⚙️ Configuration

### 1. Create Environment File

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01

# Deployment Names (use your actual deployment names from Azure Portal)
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```

### 2. Find Your Azure Credentials

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to your Azure OpenAI resource
3. Go to **Keys and Endpoint** to find your API key and endpoint
4. Go to **Model Deployments** to find your deployment names

## 📖 Usage

### Step 1: Add Agricultural PDFs

Place your agricultural PDF documents in the `data/pdfs/` folder:

```bash
cp /path/to/your/farming-guide.pdf data/pdfs/
```

### Step 2: Index the Documents

Run the indexing script to process your PDFs and create the vector store:

```bash
python src/embeddings.py
```

You should see output like:
```
Found 1 PDF file(s)
Loading: farming-guide.pdf
  - Extracted 83 page(s)
Total pages loaded: 83
Split into 277 chunks
Creating vector store...
Vector store created and saved to .../data/faiss_index
```

### Step 3: Test the RAG Chain (Optional)

Verify the system works by running a test query:

```bash
python src/chain.py
```

### Step 4: Launch the Web Interface

Start the Streamlit app:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📄 Adding New Documents

### Option 1: Re-index Everything

1. Add new PDFs to `data/pdfs/`
2. Run: `python src/embeddings.py`

This reprocesses all documents (simple but slower).

### Option 2: Incremental Add (Recommended)

1. Add new PDFs to `data/pdfs/`
2. Run: `python src/add_documents.py`

This only processes new documents and adds them to the existing index.

**First-time setup for incremental indexing:**

If you've already indexed documents, initialize the tracking file:

```bash
echo '["farming-guide.pdf"]' > data/indexed_files.json
```

Replace `farming-guide.pdf` with your actual indexed file names.

## 📁 Project Structure

```
agri-advisory-bot/
├── data/
│   ├── pdfs/                   # Place your agricultural PDFs here
│   ├── faiss_index/            # Vector store (auto-generated)
│   └── indexed_files.json      # Tracks indexed files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # PDF loading and text splitting
│   ├── embeddings.py           # Vector store creation
│   ├── chain.py                # RAG chain (retrieval + generation)
│   └── add_documents.py        # Incremental document indexing
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md
```

## 🔍 Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'langchain.prompts'`

LangChain has restructured their packages. Use these imports instead:
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
```

#### 2. `ModuleNotFoundError: No module named 'langchain.text_splitter'`

Install and use the new package:
```bash
pip install langchain-text-splitters
```
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

#### 3. ChromaDB Installation Fails (Python 3.14)

ChromaDB has compatibility issues with Python 3.14. Use FAISS instead (already configured in this project).

#### 4. Pydantic Warning

```
UserWarning: Core Pydantic V1 functionality isn't compatible with Python 3.14
```

This is a harmless warning and doesn't affect functionality.

#### 5. Azure OpenAI Authentication Error

- Verify your API key is correct in `.env`
- Check that your endpoint URL ends with a `/`
- Ensure your deployment names match exactly what's in Azure Portal

#### 6. No Results from Search

- Make sure you've run `python src/embeddings.py` after adding PDFs
- Check that PDFs are in `data/pdfs/` folder
- Verify the FAISS index exists in `data/faiss_index/`

### Getting Help

If you encounter issues:

1. Check the error message carefully
2. Ensure all environment variables are set correctly
3. Verify your Azure OpenAI deployments are active
4. Make sure all dependencies are installed: `pip install -r requirements.txt`

## 📝 Requirements.txt

```
langchain
langchain-community
langchain-openai
langchain-text-splitters
faiss-cpu
pypdf
python-dotenv
streamlit
```

## 👥 Authors

- **Group 15** - CS Capstone Project
  - P15/2130/2021 - Nesta Mwangi
  - P15/33466/2015 - Brian Gacheru Mungai  
  - P15/138515/2019 - Dalmus Maurice Otieno

## 📄 License

This project is for educational purposes as part of a university capstone project.

---