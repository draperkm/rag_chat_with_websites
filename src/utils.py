"""
Utility functions for the RAG chatbot application.

This module contains helper functions for:
- Initializing and managing Pinecone vector database
- Processing and chunking web page content
- Performing semantic search for relevant context
- Refining user queries using OpenAI
- Managing conversation history
"""

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ============================================================================
# Configuration and Initialization
# ============================================================================

def get_api_key(key_name):
    """
    Get API key from Streamlit secrets (for deployed app) or environment variables (for local development).

    This allows the deployed app to use the developer's API keys from Streamlit Cloud secrets,
    while local users need to provide their own keys in a .env file.

    Args:
        key_name (str): Name of the API key to retrieve

    Returns:
        str: The API key value
    """
    # Try to get from Streamlit secrets first (for deployed app)
    try:
        return st.secrets[key_name]
    except (AttributeError, KeyError, FileNotFoundError):
        # Fall back to environment variable (for local development)
        return os.environ.get(key_name)

# Load API keys (works for both Streamlit Cloud deployment and local development)
OPENAI_API_KEY = get_api_key('OPENAI_API_KEY')
PINECONE_API_KEY = get_api_key('PINECONE_API_KEY')

# Initialize OpenAI client (OpenAI 1.0+ API)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize the sentence transformer model for encoding queries
# all-MiniLM-L6-v2 produces 384-dimensional embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize embeddings for LangChain integration
# all-MiniLM-L12-v2 is slightly more accurate than L6-v2
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

# Pinecone index configuration
index_name = "rag-chatbot"  # Changed to a new index name

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the Pinecone index if it doesn't exist
# Note: We recreate if it exists but with wrong dimensions
existing_indexes = pc.list_indexes().names()
if index_name in existing_indexes:
    # Check if index has correct dimension (384 for all-MiniLM-L6-v2)
    index_info = pc.describe_index(index_name)
    if index_info.dimension != 384:
        # Delete and recreate with correct dimension
        pc.delete_index(index_name)
        import time
        time.sleep(1)
        existing_indexes.remove(index_name)

if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match the embedding model's output dimension (all-MiniLM-L6-v2)
        metric="cosine",  # Cosine similarity for semantic search
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # Changed to us-east-1 for better reliability
        )
    )
    # Wait for index to be ready
    import time
    time.sleep(10)  # Increased wait time for index creation

# Get reference to the index
def get_pinecone_index(name="rag-chatbot"):
    """
    Get a reference to the Pinecone index.

    Args:
        name (str): Name of the Pinecone index. Defaults to "rag-chatbot".

    Returns:
        Index: Pinecone Index object
    """
    return pc.Index(name)

# For backward compatibility, create a function to get the index
index = None  # Will be initialized when needed

# ============================================================================
# Document Processing Functions
# ============================================================================

def create_webpage_chunks(url):
    """
    Load a webpage and split it into manageable text chunks.

    This function fetches content from a URL and breaks it down into smaller pieces
    that can be embedded and stored in the vector database. Smaller chunks allow
    for more precise retrieval of relevant information.

    Args:
        url (str): The URL of the webpage to process

    Returns:
        list: A list of Document objects containing the chunked text

    Note:
        - chunk_size=1000: Maximum characters per chunk
        - chunk_overlap=0: No overlap between chunks (can be increased to preserve context)
    """
    # Load the webpage content using LangChain's WebBaseLoader
    loader = WebBaseLoader(url)
    document = loader.load()

    # Split the document into chunks for better retrieval granularity
    # RecursiveCharacterTextSplitter tries to keep paragraphs together
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )
    document_chunks = text_splitter.split_documents(document)

    return document_chunks


def find_context_chunks(input, index_name="rag-chatbot", namespace=None):
    """
    Find the most relevant text chunks from the vector database based on the input query.

    This function performs semantic search by:
    1. Converting the input text to an embedding vector
    2. Querying Pinecone for the most similar vectors
    3. Returning the text content of the top matches

    Args:
        input (str): The query text to search for
        index_name (str): Name of the Pinecone index to query. Defaults to "rag-chatbot".
        namespace (str): Optional namespace to query within. Defaults to None.

    Returns:
        str: Concatenated text from the top matching chunks, or "No matches found."

    Note:
        The function retrieves up to 3 most relevant chunks and combines them
        to provide comprehensive context to the language model.
    """
    # Get the Pinecone index
    idx = get_pinecone_index(name=index_name)

    # Convert the input query to an embedding vector
    input_em = model.encode(input).tolist()

    # Query Pinecone for the top 3 most similar vectors
    query_kwargs = {
        "vector": input_em,
        "top_k": 3,
        "include_metadata": True  # Include the original text in results
    }

    # Add namespace if provided
    if namespace:
        query_kwargs["namespace"] = namespace

    result = idx.query(**query_kwargs)

    # Extract and combine the text from matching chunks
    # The more matches we have, the more context we can provide
    if len(result['matches']) >= 3:
        # Combine all three matches for maximum context
        context = (
            result['matches'][0]['metadata']['text'] + "\n" +
            result['matches'][1]['metadata']['text'] + "\n" +
            result['matches'][2]['metadata']['text']
        )
        print(f"Found 3 relevant chunks")
        return context

    elif len(result['matches']) >= 2:
        # Combine two matches
        context = (
            result['matches'][0]['metadata']['text'] + "\n" +
            result['matches'][1]['metadata']['text']
        )
        print(f"Found 2 relevant chunks")
        return context

    elif len(result['matches']) == 1:
        # Return single match
        print(f"Found 1 relevant chunk")
        return result['matches'][0]['metadata']['text']

    else:
        # No relevant chunks found
        print("No matches found.")
        return "No matches found."


# ============================================================================
# Query Processing Functions
# ============================================================================

def query_refiner(conversation, query):
    """
    Refine the user's query using conversation context to improve retrieval accuracy.

    This function uses GPT-4 to reformulate the user's question by considering
    the conversation history. This is crucial for:
    - Resolving pronouns and references to previous messages
    - Adding context from earlier in the conversation
    - Creating a more precise search query for the knowledge base

    Args:
        conversation (str): The formatted conversation history
        query (str): The user's current query

    Returns:
        str: A refined, context-aware version of the query

    Example:
        User previously asked: "What is RAG?"
        User now asks: "How does it work?"
        Refined query: "How does Retrieval-Augmented Generation work?"
    """
    # Use OpenAI 1.0+ API
    response = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base."
            },
            {
                "role": "user",
                "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}"
            }
        ],
        temperature=0.7,  # Moderate creativity
        max_tokens=256,   # Limit response length
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content


def get_conversation_string():
    """
    Format the conversation history as a string for context.

    This function retrieves the conversation from Streamlit's session state
    and formats it as a readable dialogue between Human and Bot. This formatted
    string is used by the query_refiner to understand context.

    Returns:
        str: Formatted conversation history

    Note:
        The indexing uses [i] for requests and [i+1] for responses because
        responses is initialized with a welcome message.
    """
    conversation_string = ""

    # Iterate through the conversation history
    # responses[0] is the initial greeting, so we skip it by using [i+1]
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"

    return conversation_string
