from dotenv import load_dotenv
load_dotenv() # Load the .env file
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
#from langchain_community.vectorstores import Pinecone
import openai
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
index_name = "rag2"
print(index_name)
pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'))
#pc = pinecone.init(api_key= os.environ.get('PINECONE_API_KEY'))

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
        )
else:
    pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-west-2"
        )
        )
    
index = pc.Index('rag2')

def create_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    document_chunks = text_splitter.split_documents(document)

    #Pinecone.from_documents(document_chunks, embeddings, index_name=index_name)
    
    return document_chunks

def find_match(input):
    input_em = model.encode(input).tolist()
    # Correct call with keyword arguments
    result = index.query(vector=input_em, top_k=3, include_metadata=True)
    # Check if there are at least 2 matches
    if len(result['matches']) >= 3:
        # If there are 2 or more matches, return the text from the first two matches
        print(f'Result of 2 {result["matches"][0]["metadata"]["text"]} AND {result["matches"][1]["metadata"]["text"]} AND {result["matches"][2]["metadata"]["text"]}')
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text'] + "\n" + result['matches'][2]['metadata']['text']
    elif len(result['matches']) >= 2:
        # If there's only one match, return its text
        print(f'Result of 1 {result["matches"][0]["metadata"]["text"]} AND {result["matches"][1]["metadata"]["text"]}')
        return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']
    elif len(result['matches']) == 1:
        # If there's only one match, return its text
        print(f'Result of 1 {result["matches"][0]["metadata"]["text"]}')
        return result['matches'][0]['metadata']['text']
    else:
        # If there are no matches, return a default message or handle as needed
        print("No matches found.")
        return "No matches found."

def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base."},
            {"role": "user", "content": f"CONVERSATION LOG: \n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    #return response['choices'][0]['text']
    return response.choices[0].message['content']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string