import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdfs(pdf_directory: str) -> list:
    """
    Load all PDFs from a directory and return a list of documents.
    """
    documents = []
    
    # Check if directory exists
    if not os.path.exists(pdf_directory):
        print(f"Directory {pdf_directory} does not exist!")
        return documents
    
    # Get all PDF files
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_directory}")
        return documents
    
    print(f"Found {len(pdf_files)} PDF file(s)")
    
    # Load each PDF
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_directory, pdf_file)
        print(f"Loading: {pdf_file}")
        
        try:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)
            print(f"  - Extracted {len(docs)} page(s)")
        except Exception as e:
            print(f"  - Error loading {pdf_file}: {e}")
    
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def split_documents(documents: list, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    """
    Split documents into smaller chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


# Test the loader
if __name__ == "__main__":
    # Test loading
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "data", "pdfs")
    
    docs = load_pdfs(pdf_dir)
    
    if docs:
        # Test splitting
        chunks = split_documents(docs)
        
        # Show a sample chunk
        if chunks:
            print("\n--- Sample Chunk ---")
            print(f"Content: {chunks[0].page_content[:500]}...")
            print(f"\nMetadata: {chunks[0].metadata}")