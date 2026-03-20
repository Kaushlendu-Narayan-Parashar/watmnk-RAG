import os
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

print(f"API Key loaded: {'Yes' if os.getenv('GOOGLE_API_KEY') else 'NO - KEY MISSING'}")

# Configure Google API
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
client = None  # Not needed with google.generativeai


# ============================================================================
# 1. INTENT CLASSIFIER
# ============================================================================

def classify_intent(query):
    """
    Classify the intent of a query into one of three categories.
    
    Returns: "NEC", "Wattmonk", or "General"
    """
    query_lower = query.lower()
    
    # NEC keywords
    nec_keywords = [
        "nec", "code", "article", "electrical", "wire", "circuit",
        "voltage", "ampere", "grounding", "conduit", "panel"
    ]
    
    # Wattmonk keywords
    wattmonk_keywords = [
        "wattmonk", "service", "proposal", "permit", "planset",
        "solar", "pto", "interconnection", "site survey", "pe stamp", "zippy"
    ]
    
    # Check for NEC keywords
    nec_count = sum(1 for keyword in nec_keywords if keyword in query_lower)
    
    # Check for Wattmonk keywords
    wattmonk_count = sum(1 for keyword in wattmonk_keywords if keyword in query_lower)
    
    # Determine intent based on keyword counts
    if nec_count > wattmonk_count and nec_count > 0:
        return "NEC"
    elif wattmonk_count > nec_count and wattmonk_count > 0:
        return "Wattmonk"
    else:
        return "General"


# ============================================================================
# 2. CONTEXT RETRIEVER
# ============================================================================

def get_context(query, intent, k=3):
    """
    Retrieve relevant context from ChromaDB.
    
    Args:
        query: The user's query string
        intent: Classification ("NEC", "Wattmonk", or "General")
        k: Number of results to return (default 3)
    
    Returns: String concatenating top k results with source metadata
    """
    try:
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Load ChromaDB
        vectorstore = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings,
            collection_name="wattmonk_rag"
        )
        
        # Perform search with optional filtering
        if intent in ["NEC", "Wattmonk"]:
            # Filter by source metadata
            results = vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter={"source": intent}
            )
        else:
            # Search across all documents
            results = vectorstore.similarity_search_with_score(query, k=k)
        
        # Concatenate results into context string
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "N/A")
            content = doc.page_content
            
            context_parts.append(
                f"[Document {i}] Source: {source}, Page: {page}\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        return context if context else ""
    
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""


# ============================================================================
# 3. RESPONSE GENERATOR
# ============================================================================

def generate_response(query, context, chat_history=[], use_fallback=False):
    """
    Generate a response using Google Gemini API with retry logic.
    Falls back to error message if API fails after 3 retries.
    
    Args:
        query: The user's current query
        context: Retrieved context from ChromaDB
        chat_history: List of previous messages (last 5 will be included)
        use_fallback: Force using error fallback instead of API
    
    Returns: Response string from Gemini or fallback message
    """
    if use_fallback:
        return "I'm temporarily unavailable due to API limits. Please try again in a moment."
    
    if not API_KEY:
        return "I'm temporarily unavailable due to API limits. Please try again in a moment."
    
    # Build system prompt
    system_prompt = """You are a helpful assistant for a RAG-based chatbot specializing in 
electrical standards and solar installation services. 

Answer the user's question using the provided context. If no relevant context is available, 
answer from your general knowledge but clearly state so. 

Always mention your source when referencing information: NEC (National Electrical Code), 
Wattmonk (solar installation service), or General Knowledge.

Be concise and accurate."""
    
    # Build conversation history section (last 5 exchanges for context)
    conversation_history = ""
    if chat_history:
        # Get last 10 messages (5 user-assistant pairs) to maintain context window
        recent_history = chat_history[-10:]
        history_parts = []
        for msg in recent_history:
            role = msg.get("role", "").capitalize()
            content = msg.get("content", "")
            history_parts.append(f"{role}: {content}")
        
        if history_parts:
            conversation_history = "\n".join(history_parts)
    
    # Build full prompt with conversation history
    if conversation_history:
        full_prompt = f"""{system_prompt}

Previous Conversation:
{conversation_history}

Context Information:
{context}

Current Question: {query}

Please answer the question based on the context provided, while maintaining consistency with the previous conversation."""
    else:
        full_prompt = f"""{system_prompt}

Context Information:
{context}

User Question: {query}

Please answer the question based on the context provided above."""
    
    # Retry logic: up to 3 attempts
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:

            # Call Gemini API with gemini-2.5-flash
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(full_prompt)
            
            return response.text
        
        except Exception as e:
            error_str = str(e)
            
            print(f"Full error: {error_str}")
            
            # Check if it's a rate limit error (429)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries:
                    print("Rate limit hit, waiting 60 seconds...")
                    time.sleep(60)
                    continue
                else:
                    # All retries exhausted
                    return "I'm temporarily unavailable due to API limits. Please try again in a moment."
            else:
                # Non-rate-limit error, return immediately
                print(f"Actual error on attempt {attempt}: {error_str}")
                return f"Error: {error_str}"
    
    # Should not reach here, but return fallback just in case
    return "I'm temporarily unavailable due to API limits. Please try again in a moment."


# ============================================================================
# 4. MAIN CHAT FUNCTION
# ============================================================================

def chat(query, chat_history=[], use_fallback=False):
    """
    Main orchestration function that combines intent classification,
    context retrieval, and response generation.
    
    Args:
        query: User's input query
        chat_history: List of previous messages
        use_fallback: Use template-based response if True
    
    Returns: Dictionary with response, intent, and source information
    """
    # Step 1: Classify intent
    intent = classify_intent(query)
    print(f"[Intent] Classified as: {intent}")
    
    # Step 2: Retrieve context
    context = get_context(query, intent)
    print(f"[Context] Retrieved {len(context)} characters of context")
    
    # Step 3: Generate response
    response = generate_response(query, context, chat_history, use_fallback=use_fallback)
    
    # Determine source from intent
    source = "NEC" if intent == "NEC" else "Wattmonk" if intent == "Wattmonk" else "General Knowledge"
    
    # Build result
    result = {
        "response": response,
        "intent": intent,
        "source": source
    }
    
    return result


# ============================================================================
# TEST/DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG PIPELINE TEST")
    print("=" * 70)
    
    # Test queries
    test_queries = [
        "What are the NEC code requirements for photovoltaic system installation?",
        "Tell me about Wattmonk's solar installation service and permit process",
        "What is the difference between AC and DC circuits?"
    ]
    
    chat_history = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 70)
        
        result = chat(query, chat_history)
        
        print(f"\n[Intent] {result['intent']}")
        print(f"[Source] {result['source']}")
        print(f"\n[Response]\n{result['response']}")
        
        # Add to history
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": result['response']})
        
        print("\n" + "=" * 70)
    
    print("Test completed!")
