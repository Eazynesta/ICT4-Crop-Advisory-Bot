import streamlit as st
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from chain import create_rag_chain

# Page configuration
st.set_page_config(
    page_title="Agriculture Crop Advisory Bot",
    page_icon="🌾",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>🌾 Agriculture Crop Advisory Bot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Get instant, expert advice on crops, pests, and farming practices in Kenya</p>", unsafe_allow_html=True)

st.divider()

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize RAG chain (cached)
@st.cache_resource
def get_chain():
    return create_rag_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about farming... (e.g., How do I control locusts?)"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching agricultural documents..."):
            try:
                chain = get_chain()
                response = chain.invoke(prompt)
                st.markdown(response)
                
                # Add assistant response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar with info
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This bot provides agricultural advice based on official Kenyan farming documents.
    
    **What you can ask:**
    - Pest and disease control
    - Crop management practices
    - Soil preparation
    - Climate-smart agriculture
    
    **Data Sources:**
    - Kenya Agricultural & Livestock Research Organization (KALRO)
    - Emergency Locust Response Program (ELRP)
    """)
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("*Built for CS Capstone Project*")