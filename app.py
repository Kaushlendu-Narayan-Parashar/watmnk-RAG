import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import chat

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="WattMonk RAG Chatbot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_intent" not in st.session_state:
    st.session_state.last_intent = None


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚡ WattMonk RAG Chatbot")
    st.divider()
    
    # App description
    st.markdown("""
    ### About This App
    
    This is a **Retrieval-Augmented Generation (RAG)** chatbot powered by:
    - 📚 **ChromaDB** for vector storage
    - 🔍 **Semantic Search** for context retrieval
    - 🤖 **Google Gemini** for intelligent responses
    
    **Knowledge Bases:**
    - ⚡ **NEC** - National Electrical Code standards
    - 🌞 **Wattmonk** - Solar installation and services
    - 📖 **General** - General knowledge synthesis
    """)
    
    st.divider()
    
    # Last source badge
    st.subheader("Last Response Source")
    if st.session_state.last_intent:
        intent = st.session_state.last_intent
        if intent == "NEC":
            st.success("⚡ NEC Code Standards")
        elif intent == "Wattmonk":
            st.info("🌞 Wattmonk Services")
        else:
            st.warning("📖 General Knowledge")
    else:
        st.caption("No chat history yet")
    
    st.divider()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_intent = None
        st.rerun()


# ============================================================================
# MAIN PAGE
# ============================================================================

st.title("🔋 WattMonk RAG Chatbot")
st.markdown("*Ask me about NEC Electrical Codes or Wattmonk Services*")

st.divider()


# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            st.write(message["content"])
            
            # Show source label below assistant message
            if "source" in message:
                source = message["source"]
                if source == "NEC":
                    st.caption("⚡ Source: NEC Code Standards")
                elif source == "Wattmonk":
                    st.caption("🌞 Source: Wattmonk Services")
                else:
                    st.caption("📖 Source: General Knowledge")


# Chat input
user_input = st.chat_input("Ask about NEC codes or Wattmonk...")

if user_input:
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Get response from RAG pipeline
    with st.spinner("🤔 Thinking..."):
        result = chat(user_input, chat_history=st.session_state.chat_history)
    
    # Store response and source
    assistant_message = {
        "role": "assistant",
        "content": result["response"],
        "source": result["source"],
        "intent": result["intent"]
    }
    st.session_state.chat_history.append(assistant_message)
    st.session_state.last_intent = result["intent"]
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(result["response"])
        
        # Show source label
        if result["intent"] == "NEC":
            st.caption("⚡ Source: NEC Code Standards")
        elif result["intent"] == "Wattmonk":
            st.caption("🌞 Source: Wattmonk Services")
        else:
            st.caption("📖 Source: General Knowledge")
    
    # Rerun to update chat display
    st.rerun()
