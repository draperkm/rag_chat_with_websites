from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# Import all the tools needed from Langchain
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryMemory
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)

from langchain.vectorstores import Pinecone as PineconeVectorStore
# Import the Streamlit chat
import streamlit as st
from streamlit_chat import message
from utils import *

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# app config
#st.set_page_config(page_title="Chat with websites", page_icon="🌎")
st.title("Chat with the web page 🌎 🌐 📡 🛰️")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    # create a session state for both response and request variables
    if 'website' not in st.session_state:
        st.session_state['website'] = ['yes']
        website_chunks = create_webpage_chunks(website_url)
        #pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'))
        #index = pc.Index('rag2')
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")
        docsearch = PineconeVectorStore.from_documents(website_chunks, embeddings, index_name='rag2')
        
    if 'responses' not in st.session_state:
        st.session_state['responses'] = ['Hello, what do you want to know?']
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []


    # create a chat agent instance
    llm = ChatOpenAI(
        model_name = 'gpt-4'
        )

    # manage conversation memory in from of buffer
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(
            k = 3, 
            return_messages = True)
        
    # create promt templates for system and human, and chat(system + human)
    system_msg_template = SystemMessagePromptTemplate.from_template(
        template="""Answer the question as truthfully as possible using the provided context, 
                    and if the answer is not contained within the text below, you can use 
                    general knowledge, but you must specify that the information are not from
                    the provided context.
                    When the answer longer than three sentences, formulate it in such a way
                    that is summarized in bullet points, giving a short introduction
                    and a final short summary.
                    In any case make sure to make good use of the context provided.
                    Make sure to give complete answers but balance the lenght of them on the 
                    lenght of the questions'"""
    )

    human_msg_template = HumanMessagePromptTemplate.from_template(
        template="{input}"
        )

    prompt_template = ChatPromptTemplate.from_messages([
        system_msg_template, 
        MessagesPlaceholder(variable_name="history"), 
        human_msg_template
        ])

    conversation = ConversationChain(
        memory=st.session_state.buffer_memory, 
        prompt=prompt_template, 
        llm=llm, 
        verbose=True
        )

    # container for chat history
    response_container = st.container()

    # container for text box
    textcontainer = st.container()

    # create conversation (streamlit) logic
    with textcontainer:
        query = st.chat_input("", key="input")
        if query:
            with st.spinner("typing..."):
                conversation_string = get_conversation_string()
                refined_query = query_refiner(
                    conversation_string, 
                    query
                    )
                st.subheader("Refined Query:")
                st.write(refined_query)
                if conversation_string:
                    st.subheader("Conversation String:")
                    st.code(conversation_string)
                context = find_context_chunks(refined_query)
            
                #print(context)  
                response = conversation.predict(
                    input=f"Context:\n {context} \n\n Query:\n{refined_query}"
                    )

            st.session_state.requests.append(refined_query)
            st.session_state.responses.append(response) 


    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                with st.chat_message("user", avatar="🔗"):
                    st.write(
                        st.session_state['responses'][i],
                        key = str(i)
                        )
                if i < len(st.session_state['requests']):
                    with st.chat_message("ai", avatar="💬"):
                        st.write(
                            st.session_state["requests"][i], 
                            is_user=True,
                            key = str(i) + "_user"
                            )