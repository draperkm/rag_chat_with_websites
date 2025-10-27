"""
RAG-based Web Page Chatbot - Main Application

This Streamlit application implements a Retrieval-Augmented Generation (RAG) chatbot
that can answer questions about any website's content. The app:
1. Loads and processes content from a user-provided URL
2. Stores the content as embeddings in Pinecone vector database
3. Uses semantic search to find relevant context for user queries
4. Generates informed responses using GPT-4 with the retrieved context

Author: Jean Charles
"""

from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Fix for KMP library conflicts on some systems
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Fix for HuggingFace tokenizers forking warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# LangChain imports for LLM
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Streamlit for web interface
import streamlit as st

# Import utility functions from utils.py
from utils import (
    create_webpage_chunks,
    find_context_chunks,
    query_refiner,
    get_conversation_string,
    get_api_key,
    get_pinecone_index
)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Web Page Chatbot",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌐 Chat with Any Web Page")
st.markdown(
    """
    Enter a website URL in the sidebar, and I'll help you explore its content through conversation.
    I use Retrieval-Augmented Generation to provide accurate, context-aware answers.
    """
)

# ============================================================================
# Sidebar - User Input
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    # Fixed Pinecone index name
    pinecone_index_name = "rag-chatbot"

    # Initialize webpage list in session state
    if 'webpages' not in st.session_state:
        st.session_state['webpages'] = []

    # Initialize unique session namespace for Pinecone isolation
    if 'session_namespace' not in st.session_state:
        import uuid
        st.session_state['session_namespace'] = f"session_{uuid.uuid4().hex[:8]}"

    # Display loaded webpages count
    st.markdown(f"**Loaded Webpages: {len(st.session_state['webpages'])}/5**")

    # Show list of loaded webpages
    if st.session_state['webpages']:
        with st.expander("📄 View Loaded Pages", expanded=False):
            for idx, url in enumerate(st.session_state['webpages'], 1):
                # Truncate long URLs for display
                display_url = url if len(url) <= 40 else url[:37] + "..."
                st.text(f"{idx}. {display_url}")

    st.markdown("---")

    # Only show URL input if less than 5 webpages loaded
    if len(st.session_state['webpages']) < 5:
        st.markdown("Enter a webpage URL to add to the knowledge base:")

        website_url = st.text_input(
            "Website URL",
            placeholder="https://example.com",
            help="Enter the full URL including https://"
        )
    else:
        website_url = None
        st.warning("⚠️ Maximum 5 webpages reached. Use the restart button to clear and start over.")

    # Restart button
    st.markdown("---")
    if st.button("🔄 Restart Session", help="Clear all loaded webpages and restart", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    st.markdown("---")
    st.markdown(
        """
        ### How it works:
        1. **Load**: The webpage content is fetched and processed
        2. **Embed**: Text is converted to vectors and stored in Pinecone
        3. **Retrieve**: Relevant chunks are found using semantic search
        4. **Generate**: GPT-4 creates answers based on retrieved context
        """
    )

    st.markdown("---")
    st.info(
        """
        ℹ️ **Free to Use**

        This demo uses the developer's API keys.
        No setup required - just enter a URL and start chatting!

        For heavy usage or customization, please [clone the repo](https://github.com/yourusername/rag-chatbot) and use your own keys.
        """
    )

# ============================================================================
# Main Application Logic
# ============================================================================

# Process new webpage if URL is provided and not already in the list
if website_url and website_url != "" and website_url not in st.session_state['webpages']:
    with st.spinner(f"🔄 Loading and processing webpage {len(st.session_state['webpages']) + 1}/5..."):
        try:
            # Split the webpage into chunks
            website_chunks = create_webpage_chunks(website_url)

            # Initialize embeddings model
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L12-v2"
            )

            # Store chunks in Pinecone vector database with session namespace
            # Note: from_documents adds to the existing index without clearing it
            # Using namespace ensures each session's data is isolated
            docsearch = PineconeVectorStore.from_documents(
                documents=website_chunks,
                embedding=embeddings,
                index_name=pinecone_index_name,
                namespace=st.session_state['session_namespace']
            )

            # Add the URL to the list of processed webpages
            st.session_state['webpages'].append(website_url)
            st.success(f"✅ Webpage {len(st.session_state['webpages'])}/5 processed successfully!")

            # Clear the text input by rerunning
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error processing webpage: {str(e)}")

# Main application logic
if len(st.session_state['webpages']) == 0:
    # Show instructions when no webpages are loaded
    st.info("👈 Please enter a website URL in the sidebar to get started")

    st.markdown(
        """
        ### How it works:
        - Add up to **5 webpages** to your knowledge base
        - Each webpage is processed and stored as embeddings
        - Ask questions about any of the loaded pages
        - The AI will search across all loaded content to find relevant answers

        ### Example Use Cases:
        - 📚 Explore multiple documentation pages
        - 📰 Analyze related news articles
        - 🔬 Research scientific papers on a topic
        - 📝 Compare multiple blog posts
        - 💼 Review company information from different sources
        """
    )
else:
    # ========================================================================
    # Initialize Session State for Chat
    # ========================================================================

    # Initialize conversation history
    if 'responses' not in st.session_state:
        num_pages = len(st.session_state['webpages'])
        page_text = "webpage" if num_pages == 1 else "webpages"
        st.session_state['responses'] = [
            f"Hello! I've processed {num_pages} {page_text}. What would you like to know?"
        ]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    # Initialize message history for LLM (keeps last 6 messages = 3 exchanges)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # ========================================================================
    # Initialize LLM
    # ========================================================================

    # Initialize GPT-4 model with API key
    llm = ChatOpenAI(
        model_name='gpt-4',
        temperature=0.7,  # Balance between creativity and consistency
        openai_api_key=get_api_key('OPENAI_API_KEY')
    )

    # System message that defines the assistant's behavior
    num_pages = len(st.session_state['webpages'])
    page_text = "webpage" if num_pages == 1 else f"{num_pages} webpages"
    SYSTEM_PROMPT = f"""You are a helpful AI assistant that answers questions about {page_text} content.

Instructions:
- Answer questions truthfully using the provided context from the loaded webpages
- If the answer isn't in the context, you may use general knowledge but clearly state this
- For answers longer than three sentences, use bullet points with a brief intro and summary
- Balance answer length with question complexity
- Always make good use of the provided context
- When answering from multiple sources, synthesize the information coherently

Maintain a professional, informative, and friendly tone."""

    # ========================================================================
    # Chat Interface
    # ========================================================================

    # Container for displaying chat history
    response_container = st.container()

    # Container for user input
    textcontainer = st.container()

    # Process user input
    with textcontainer:
        query = st.chat_input(
            "Ask a question about the webpage...",
            key="input"
        )

        if query:
            with st.spinner("🤔 Thinking..."):
                # Get conversation history
                conversation_string = get_conversation_string()

                # Refine query using conversation context
                refined_query = query_refiner(conversation_string, query)

                # Display refined query in expander (optional debug info)
                with st.expander("🔍 Query Processing Details", expanded=False):
                    st.write("**Original Query:**", query)
                    st.write("**Refined Query:**", refined_query)
                    if conversation_string:
                        st.write("**Conversation Context:**")
                        st.code(conversation_string, language="")

                # Find relevant context chunks from vector database
                context = find_context_chunks(
                    refined_query,
                    index_name=pinecone_index_name,
                    namespace=st.session_state['session_namespace']
                )

                # Build messages for the LLM
                # Keep only the last 3 exchanges (6 messages) for context window management
                recent_messages = st.session_state.messages[-6:] if len(st.session_state.messages) > 6 else st.session_state.messages

                messages = [SystemMessage(content=SYSTEM_PROMPT)]
                messages.extend(recent_messages)
                messages.append(HumanMessage(content=f"Context:\n{context}\n\nQuery:\n{refined_query}"))

                # Generate response using LLM
                response = llm.invoke(messages)
                response_text = response.content

                # Store in message history
                st.session_state.messages.append(HumanMessage(content=refined_query))
                st.session_state.messages.append(AIMessage(content=response_text))

            # Store the exchange in session state for display
            st.session_state.requests.append(query)  # Use original query for display
            st.session_state.responses.append(response_text)

    # ========================================================================
    # Display Chat History
    # ========================================================================

    with response_container:
        if st.session_state['responses']:
            # Display all messages in chronological order
            for i in range(len(st.session_state['responses'])):
                # Display bot response
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(st.session_state['responses'][i])

                # Display user query (if exists)
                if i < len(st.session_state['requests']):
                    with st.chat_message("user", avatar="👤"):
                        st.write(st.session_state['requests'][i])
