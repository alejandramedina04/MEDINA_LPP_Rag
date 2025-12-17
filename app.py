# =============================================================================
# Streamlit Application for RAG Assistant
# =============================================================================
# This is the user interface. It imports our backend modules and handles:
# - Page layout and configuration
# - User input (chat, sidebar settings)
# - Displaying responses and sources
# - Session state management
# =============================================================================

import streamlit as st
import os
from backend.database import RAGDatabase
from backend.agent import RAGAgent
import config

# -----------------------------------------------------------------------------
# Page Configuration (must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="📖",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
# IMPORTANT: Streamlit reruns this entire file every time the user:
# - Types a message
# - Clicks a button  
# - Moves a slider
#
# Without session state, our chat history would vanish on every interaction.
# st.session_state store a dictionary on session information that should persist
# through user interactions.
# 
# Pattern: if 'key' not in st.session_state: st.session_state.key = default
# -----------------------------------------------------------------------------

# Chat history - list of {"role": "user"/"assistant", "content": "..."}
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Database path from sidebar
if 'db_path' not in st.session_state:
    st.session_state.db_path = config.DEFAULT_DB_PATH

# Number of results to retrieve
if 'top_k' not in st.session_state:
    st.session_state.top_k = config.DEFAULT_TOP_K

# Database instance (expensive to create, so we cache it)
if 'database' not in st.session_state:
    st.session_state.database = None


# -----------------------------------------------------------------------------
# Sidebar - User Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # TO DO: API Key input
    # Make sure the user can hide their input
    api_key = st.text_input("OpenAI API Key",
                            type = "password",
                            help = "Enter your OpenAI API Key")
    
    # TO DO: Database path input via config.py
    db_path = st.text_input("Database Path",
                            value = config.DEFAULT_DB_PATH,
                            help = "Path to your DuckDB vector database")
    
    # Store in session state
    st.session_state.db_path = db_path
    
    # TO DO: Top K results with an interactive slider bar
    top_k = st.slider("Results per Query",
                      min_value = 3,
                      max_value = 20,
                      value = config.DEFAULT_TOP_K,
                      help = "Number of chunks to retrieve per search")
    # Store in session state
    st.session_state.top_k = top_k
    
    # TO DO: Model selection dropdown menu
    model_choice = st.selectbox("LLM Model",
                                config.AVAILABLE_MODELS,
                                index = 0)
    
    # TO DO: Create max_iter slider
    # Max iterations for tool calls
    max_iter = st.slider("Max Tool Calls",
                         min_value = 1,
                         max_value = 5, 
                         value = config.DEFAULT_MAX_ITER,
                         help = "Maximum number of database queries per question" )
    
    st.divider()
    
    # TO DO: Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
  
    
    # TO DO: Update for your project
    st.markdown("""
    ### About
    This RAG assistant answers questions about artificial intelligence, focusing on:
    - How AI is used in research, education, and daily life 
    - Risks such as bias, safety concerns, and black-box systems
    - Ethical, social, and policy questions about modern AI 
    The assistant uses curated PDFs, web articles, YouTube transcripts, and text files about AI.
    """)

# -----------------------------------------------------------------------------
# Main App Header
# -----------------------------------------------------------------------------
st.title("🤖 AI Benefits and Risks Assistant")
st.markdown("Ask questions about artificial intelligence and receive AI-powered answers based on curated content.")


# -----------------------------------------------------------------------------
# Database Connection
# -----------------------------------------------------------------------------
# Recreate database instance if path changed from default in case multiple duckdb files
if not st.session_state.database or st.session_state.database.db_path != db_path:
    st.session_state.database = RAGDatabase(db_path)

# Test and display connection status
if not st.session_state.database.test_connection():
    st.error(f"❌ Database not found at: `{db_path}`")
    st.info("Please update the database path in the sidebar.")
    if not os.path.exists(db_path):
        st.stop() # Stop execution here - can't continue without database
else:
    st.success(f"✅ Database connected: `{db_path}`")

# -----------------------------------------------------------------------------
# API Key Check
# -----------------------------------------------------------------------------
if not api_key:
    st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
    st.stop()

# Set the API key as environment variable (OpenAI client reads from here)
os.environ["OPENAI_API_KEY"] = api_key

# -----------------------------------------------------------------------------
# Display Chat History
# -----------------------------------------------------------------------------
# Loop through all previous messages and display them
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources for assistant messages if available
        if message["role"] == "assistant" and message.get("sources"):

            # Creates expander bar and allows user to view sources
            with st.expander(f"View Sources ({len(message['sources'])} passages retrieved)"):
                
                # Loops over source number and content
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                    st.text_area(
                        f"Passage {i}",
                        source["text"],
                        height=150,
                        key=f"source_{id(message)}_{i}",
                        label_visibility="collapsed"
                    )
                    st.divider()

# -----------------------------------------------------------------------------
# Chat Input and Response Generation
# -----------------------------------------------------------------------------
# st.chat_input returns None until user submits, then returns their text
# The := (walrus operator) assigns AND checks in one line

if prompt := st.chat_input("Ask a question about Artificial Intelligence..."):
    
    # TO DO: Add user message to history
    st.session_state.messages.append({"role":"user", "content":prompt})

    # TO DO: Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching database and generating answer..."):
            try:
                # TO DO: Initialize Agent
                agent = RAGAgent(db = st.session_state.database,
                                 model_name = model_choice,
                                 max_iter = max_iter)
                
                # Get answer
                result = agent.ask(prompt)
                response = result["answer"]
                sources = result["sources"]
                
                st.markdown(response)
                
                # Display sources immediately
                if sources:
                    with st.expander(f"📚 View Sources ({len(sources)} passages retrieved)"):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (Similarity: {source['similarity']:.3f})")
                            st.text_area(
                                f"Passage {i}",
                                source["text"],
                                height=150,
                                key=f"source_new_{i}",
                                label_visibility="collapsed"
                            )
                            if i < len(sources):
                                st.divider()
                
                # Add to history with sources
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })

# TO DO: Example questions in an expander
with st.expander("💡 Example Questions"):
    examples = [
        "What are the main benefits and risks of modern AI?",
        "Explain what a 'black box' AI model is and why it can be a problem.",
        "How is AI being used in research, education, and everyday life?",
        "What ethical issues does AI raise around bias and fairness?",
        "How can we make AI systems more transparent and explainable?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            # Simulate entering the question
            st.session_state.messages.append({"role": "user", "content": example})
            st.rerun()
