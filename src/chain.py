import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from embeddings import load_vector_store

# Load environment variables
load_dotenv()


def get_llm():
    """Initialize Azure OpenAI Chat model."""
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0.3,  # Lower temperature for factual responses
    )


def format_docs(docs):
    """Format retrieved documents into a single string with sources."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get('page', 'Unknown')
        source = doc.metadata.get('source', 'Unknown')
        filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
        formatted.append(f"[Source {i} - {filename}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def create_rag_chain():
    """Create the RAG chain combining retrieval and generation."""
    
    # Load vector store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Create prompt template
    template = """You are an expert agricultural advisor for Kenyan farmers. Your role is to provide 
accurate, practical advice based on the agricultural documents provided.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context below
2. If the context doesn't contain enough information to answer, say so clearly
3. Always mention which source(s) your information comes from
4. Give practical, actionable advice when possible
5. Use simple language that farmers can understand

CONTEXT FROM AGRICULTURAL DOCUMENTS:
{context}

---

FARMER'S QUESTION: {question}

HELPFUL ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)
    
    # Get LLM
    llm = get_llm()
    
    # Build the chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def ask(question: str) -> str:
    """Ask a question and get an answer from the RAG system."""
    chain = create_rag_chain()
    response = chain.invoke(question)
    return response


# Test the chain
if __name__ == "__main__":
    print("=" * 60)
    print("AGRICULTURE CROP ADVISORY BOT - Test")
    print("=" * 60)
    
    test_questions = [
        "How do I control locusts on my farm?",
        "What crops are suitable for arid regions?",
        "How should I prepare my soil for planting?",
    ]
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"QUESTION: {question}")
        print("=" * 60)
        
        answer = ask(question)
        print(f"\nANSWER:\n{answer}")
        
        # Only test one question for now
        break