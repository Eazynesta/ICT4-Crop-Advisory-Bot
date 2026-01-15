import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from document_loader import load_pdfs, split_documents

# Load environment variables
load_dotenv()

# Constants
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "faiss_index")
PDF_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")


def get_embeddings():
    """Initialize Azure OpenAI embeddings."""
    return AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
    )


def create_vector_store(chunks: list) -> FAISS:
    """
    Create a FAISS vector store from document chunks.
    """
    print("Creating vector store...")
    print(f"Embedding {len(chunks)} chunks (this may take a minute)...")
    
    embeddings = get_embeddings()
    
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    
    # Save to disk
    vector_store.save_local(FAISS_INDEX_DIR)
    print(f"Vector store created and saved to {FAISS_INDEX_DIR}")
    
    return vector_store


def load_vector_store() -> FAISS:
    """
    Load an existing FAISS vector store.
    """
    embeddings = get_embeddings()
    
    vector_store = FAISS.load_local(
        FAISS_INDEX_DIR, 
        embeddings,
        allow_dangerous_deserialization=True  # Safe since we created the index ourselves
    )
    
    print(f"Loaded vector store from {FAISS_INDEX_DIR}")
    return vector_store


def index_documents():
    """
    Full indexing pipeline: Load PDFs -> Split -> Embed -> Store
    """
    # Load and split documents
    documents = load_pdfs(PDF_DIR)
    
    if not documents:
        print("No documents to index!")
        return None
    
    chunks = split_documents(documents)
    
    # Create vector store
    vector_store = create_vector_store(chunks)
    
    return vector_store


# Test the indexing
if __name__ == "__main__":
    # Run full indexing
    vector_store = index_documents()
    
    if vector_store:
        # Test a sample search
        print("\n--- Testing Search ---")
        query = "How do I control locusts?"
        results = vector_store.similarity_search(query, k=3)
        
        print(f"Query: {query}\n")
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Content: {doc.page_content[:300]}...")
            print(f"  Source: Page {doc.metadata.get('page', 'N/A')}")
            print()