import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from embeddings import load_vector_store

# Load environment variables
load_dotenv()


def format_chat_history(messages: list) -> str:
    """Format chat history into a string for the prompt."""
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages:
        role = "Farmer" if msg["role"] == "user" else "Advisor"
        formatted.append(f"{role}: {msg['content']}")

    return "\n".join(formatted)


def needs_retrieval(question: str, llm) -> bool:
    """
    Determine if a user's message requires document retrieval.
    Returns True for agricultural questions, False for greetings/chitchat.
    """
    classifier_prompt = ChatPromptTemplate.from_template(
        """You are a message classifier for an agricultural advisory chatbot.
Determine if the following message requires retrieving information from agricultural documents.

Messages that DO NOT need retrieval:
- Greetings (hello, hi, good morning, etc.)
- Farewells (bye, goodbye, thank you, etc.)
- Simple chitchat or pleasantries
- Questions about the bot itself (who are you, what can you do)

Messages that DO need retrieval:
- Questions about farming, crops, soil, pests, weather, livestock
- Agricultural advice requests
- Questions about specific farming practices or techniques

Message: {question}

Respond with ONLY "YES" if retrieval is needed, or "NO" if it's just chitchat/greeting."""
    )

    chain = classifier_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question}).strip().upper()
    return result == "YES"


def get_chitchat_response(question: str, chat_history: list, llm) -> str:
    """Generate a friendly response for greetings and chitchat."""
    chitchat_prompt = ChatPromptTemplate.from_template(
        """You are a friendly agricultural advisor for Kenyan farmers.
Respond to this greeting or chitchat in a warm, helpful manner.
Keep your response brief and friendly.
If appropriate, let them know you're here to help with agricultural questions.

Previous conversation:
{chat_history}

User message: {question}

Response:"""
    )

    chain = chitchat_prompt | llm | StrOutputParser()
    history_str = format_chat_history(chat_history) if chat_history else "No previous conversation."
    return chain.invoke({"question": question, "chat_history": history_str})


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


def create_rag_chain_with_sources():
    """Create a RAG chain that returns both the answer and source documents."""

    # Load vector store
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Create prompt template with chat history
    template = """You are an expert agricultural advisor for Kenyan farmers. Your role is to provide
accurate, practical advice based on the agricultural documents provided.

IMPORTANT INSTRUCTIONS:
1. Base your answer ONLY on the provided context below
2. If the context doesn't contain enough information to answer, say so clearly
3. Always mention which source(s) your information comes from
4. Give practical, actionable advice when possible
5. Use simple language that farmers can understand
6. Consider the conversation history when answering follow-up questions

PREVIOUS CONVERSATION:
{chat_history}

---

CONTEXT FROM AGRICULTURAL DOCUMENTS:
{context}

---

FARMER'S QUESTION: {question}

HELPFUL ANSWER:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Get LLM
    llm = get_llm()

    def retrieve_and_answer(question: str, chat_history: list = None) -> dict:
        """Retrieve documents and generate answer, returning both."""
        if chat_history is None:
            chat_history = []

        # Check if retrieval is needed
        if not needs_retrieval(question, llm):
            # Handle greeting/chitchat without retrieval
            answer = get_chitchat_response(question, chat_history, llm)
            return {
                "answer": answer,
                "sources": []  # No sources for chitchat
            }

        # Get the relevant documents
        docs = retriever.invoke(question)

        # Format context for the prompt
        context = format_docs(docs)

        # Format chat history
        history_str = format_chat_history(chat_history)

        # Generate the answer
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": history_str
        })

        # Extract source information from documents
        sources = []
        for doc in docs:
            page = doc.metadata.get('page', 'Unknown')
            source = doc.metadata.get('source', 'Unknown')
            filename = os.path.basename(source) if source != 'Unknown' else 'Unknown'
            # Get a preview of the content (first 200 chars)
            preview = doc.page_content[:200].replace('\n', ' ').strip()
            if len(doc.page_content) > 200:
                preview += "..."
            sources.append({
                "filename": filename,
                "page": page,
                "preview": preview
            })

        return {
            "answer": answer,
            "sources": sources
        }

    return retrieve_and_answer


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