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
git clone https://github.com/Eazynesta/ICT4-Crop-Advisory-Bot.git
cd ICT4-Crop-Advisory-Bot
```


### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Create Environment File

Create a `.env` file in the project root:

```env
# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4.1
```


### Step 1: Add Agricultural PDFs

Place your agricultural PDF documents in the `data/pdfs/` folder:

### Step 2: Index the Documents

Run the indexing script to process the PDFs and create the vector store:

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

## 📁 Project Structure

```
agri-advisory-bot/
├── data/
│   ├── pdfs/                   # Place the agricultural PDFs here
│   ├── faiss_index/            # Vector store (auto-generated)
│   └── indexed_files.json      # Tracks indexed files (auto-generated)
├── src/
│   ├── __init__.py
│   ├── document_loader.py      # PDF loading and text splitting
│   ├── embeddings.py           # Vector store creation
│   ├── chain.py                # RAG chain (retrieval + generation)
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
└── README.md
```