# Building a RAG application from scratch

In this tutorial, you will build a Streamlit LLM app that can generate text from a user-provided prompt. This Python app will use the LangChain framework and Streamlit. Optionally, you can deploy your app to Streamlit Community Cloud when you're done.

# Limits of LLMs

A large language model is built to analyze and comprehend textual inputs, or prompts, and produce corresponding textual outputs, or responses. These models undergo training using a vast collection of unannotated text data, enabling them to discern linguistic structures and accumulate knowledge about words relationships. The fundamental disparity between a regular language model and a large language model lies in the magnitude of parameters employed for their operation.

![Schema3](docs/schema3.jpg)

Some notable LLMs are OpenAI's GPT series of models (e.g., GPT-3.5 and GPT-4, used in ChatGPT and Microsoft Copilot), Google's PaLM and Gemini (the latter of which is currently used in the chatbot of the same name), xAI's Grok, Meta's LLaMA family of open-source models, Anthropic's Claude models, and Mistral AI's open source models [11].

## Grounding LLMs to solve generation problems

Extensive pre-trained language models have demonstrated the capability to encapsulate factual information within their parameters, attaining unparalleled performance on subsequent NLP tasks when adequately fine-tuned. **Nonetheless, their proficiency in accessing and accurately manipulating this embedded knowledge remains constrained**. Therefore, in tasks that heavily rely on knowledge, their effectiveness falls short of specialized task-specific architectures. [5].

![RAG_Architecture](docs/rag_expand.jpg) [7]

The Retrieval-Augmented Generation (RAG) introduces a nuanced approach to handling and generating information, which can be contrasted using the "closed book" vs. "open book" analogy and explaining the advantages of "grounding".

### Closed book vs open book:
In the **closed book** paradigm, a language model generates answers based solely on the knowledge it has internalized during its training phase. It doesn't access external information or databases at the time of inference. This approach relies on the model's ability to store and recall facts, concepts, and relationships from its training data. While this can be highly effective for a wide range of tasks, the limitations are evident in terms of the freshness, specificity, and verifiability of the information provided [4].
Contrarily, the **open book** approach integrates external knowledge sources during the inference phase, allowing the model to retrieve and use the most current and relevant information for generating responses. The RAG paradigm is a prominent example of the open book approach, combining the strengths of retrieval-based and generative models to produce more accurate, reliable, and transparent outputs [4].

### Grounding:

Grounding in the context of LLMs, particularly within the RAG paradigm, refers to the model's ability to anchor its responses in real-world knowledge that can be traced back to specific sources. [4]
By leveraging external sources for information retrieval, RAG and similar models are less likely to "hallucinate" because their responses are based on existing content. This reliance on external data acts as a check against the model's propensity to generate unsupported statements.
Another significant advantage of grounding is the ability to provide citations and attributions, pointing back to the source of the information. This not only enhances the credibility of the responses but also allows users to verify the information independently. **In knowledge-intensive tasks, where accuracy and reliability are paramount, the ability to cite sources directly is a substantial benefit**.

In conclusion RAGs find their true motivation, in delimiting the LLM to act only on a limited set of data, making fine-tuning not strictly necessary, resulting in time saving and cost saving, even if there is a threshold where fine-tuning would be preferable.

# 💡 Retrieval-Augmented Generation (RAG)

A Retrieval-Augmented Generation (RAG) is a method to improve domain-specific responses of large language models [13]. The process starts with a retrieval task, searching for information semantically relevant to the user query within a specially created knowledge database. This database, known as a **vector store**, contains **embeddings (vectors)** that represent the documentation from which the model aims to extract information to include in a final enhanced prompt for the language model. The relevant context extracted in this search is then combined with the original prompt, extending the model's context window with necessary information. This preparatory step effectively increases the reliability of the model's responses by expanding the original prompt with pertinent data that the model will use to ground the response, and it is what characterizes a RAG application. A typical RAG setup consists of:

### Loading the documents in a vector database: 

A vector database stores data as high-dimensional vectors, mathematical entities representing data attributes or features. These vectors, varying in dimensionality from tens to thousands, encapsulate the complexity and detail of the data, which could include text, images, audio, or video. They are created through transformations or embedding functions applied to raw data, employing techniques from machine learning, word embeddings, or feature extraction algorithms. The primary benefit of a vector database is its capability for rapid and precise similarity searches and data retrieval. Unlike traditional query methods that rely on exact matches or predefined criteria, a vector database enables searches for data that are most similar or relevant based on their semantic or contextual significance, utilizing vector distance or similarity measures [14].

![Langchain](docs/langchain.jpg)

### Enhancing the prompt after retrieving the relevant documents:

After enhancing the prompt with retrieved documents or snippets, these are given to a large language model. The model incorporates this additional context to generate responses that are more detailed and relevant, drawing on the factual content of the provided documents. This process allows the language model to produce answers that not only adhere more closely to the specifics of the query but also maintain a higher level of accuracy by leveraging the external information. [2].

![RAG Diagram](docs/rag.jpg)

# 🛠️ Code walkthrough

## Requirements

The following libraries are necessary for setting up our development environment. By ensuring these tools and libraries are installed, we guarantee that our code will execute without issues, allowing our chatbot to operate as planned. It is important to notice that the `Python 3.10.8` version is used.


```Python
- `streamlit`
- `streamlit_chat`
- `langchain`
- `sentence_transformers`
- `openai`
- `unstructured`
- `unstructured[pdf]`
- `pinecone-client`
```

Streamlit: This library helps us to create interactive web apps for machine learning and data science projects.

## Creating a Vestor Store with Pinecone

Creating a vector store involves storing and managing high-dimensional vectors, which are numerical representations of data points, such as text embeddings or image features. Vector stores enable efficient similarity search and retrieval of relevant data points based on their vector representations. `Pinecone` is a managed vector database service that provides scalable and efficient vector storage and similarity search capabilities, and its library is used to create and interact with a vector store.

The Pinecone class is the main class and represents a connection to the Pinecone vector database service, providing methods for initializing the connection, creating and managing indexes, and performing vector operations.

By calling `.create_index` the Pinecone class create an index which is a data structure used in the Pinecone vector database to efficiently store, manage, and search high-dimensional vectors. It is designed to enable fast similarity searches and nearest neighbor queries on large-scale vector datasets. The hyperparameters of the type of index are specified in the function.

```Python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key= os.environ.get('PINECONE_API_KEY'))

if index_name not in pc.list_indexes().names():
    pc.create_index(name = index_name,
    dimension = 384,
    metric = "cosine",
    spec = ServerlessSpec(
    cloud = "aws",
    region = "us-west-2"))
```

The content of a webpage is loaded and then splitted into chunks. Those will be uploaded in the in the vector store as embedded vectors. 

The `RecursiveCharacterTextSplitter`, from the `Langchain` library, provides a flexible and customizable way to split long texts into manageable chunks for further processing or analysis. The `chunk_size` and `chunk_overlap` parameters work together to control the granularity and context preservation of the text splitting process. The optimal values depend on the specific requirements of your application, the nature of the text being split, and the downstream tasks that will process the chunks. After experimenting, in this case a chunk_size of 1000 with an overlap of 200 have been selected.

```Python
def create_webpage_chunks(url):

    loader = WebBaseLoader(url)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(document)
    
    return document_chunks
```

The embedding of the vector store are created by calling `.from_documents` function, using an embedding model, the chunks to store and the name of the vector store in Pinecone.

The embedding model is `all-MiniLM-L12-v2`, from the `sentence_transformers` library, which is a sentence and short paragraph encoder. Given an input text, it outputs a vector which captures the semantic information. It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search.

```Python
from langchain.vectorstores import Pinecone as PineconeVectorStore
from sentence_transformers import SentenceTransformer

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L12-v2")

website_chunks = create_webpage_chunks(website_url)

docsearch = PineconeVectorStore.from_documents(website_chunks, embeddings, index_name='rag')
```

## Get the query and redefine it to boost performance

The `query_refiner()` function makes use of `openai.ChatCompletion.create()` to create an enhanced query, that will increase the precision of the response to the final promt. Different roles are used to because newer models are trained to adhere to system messages. A system message is never part of the conversation and never accessible to the end-user. Therefore, it can be used to control the scope of the model’s interactions with the end-user. The user message can be used to ground the model into a specific behavior, but it cannot control it entirely. During the conversation, the user can instruct the model to contradict the statement given by the role user, as they have the same role, and the model cannot deny user asking to override their previous instructions. However, if there’s a system message, the model will give precedence to it over the user message [15].

```Python
def query_refiner(conversation, query):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Given the following user query and conversation log, formulate a question that would be more relevant to the context."},
            {"role": "user", "content": f"Conversation log: \n{conversation}\n\nQuery: {query}"}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)
    return response.choices[0].message['content']
```
## Matching vectors with the query

Once we have a query, we can embed it, send it to the vector store and perform a semantic search between the query vector and the document chunks to retrieve the most relevant ones to include in the final promt. 

```Python
def find_context_chunks(input):

    input_em = model.encode(input).tolist()

    result = index.query(
        vector = input_em, 
        top_k = 3,
        include_metadata=True)

    output = []
    
    return output
```

In `main.py` we get the context by calling the `find_context_chunks()` function.

```Python
context = find_context_chunks(refined_query)
```

## Keeping the chat memory and creating the new promt (Langchain)


ConversationBufferMemory usage is straightforward. It simply keeps the entire conversation in the buffer memory up to the allowed max limit [18]

```Python
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k = 3, return_messages = True)
```
LangChain provides different types of MessagePromptTemplate. The most commonly used are AIMessagePromptTemplate, SystemMessagePromptTemplate and HumanMessagePromptTemplate, which create an AI message, system message and human message respectively.

```Python
system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, you can use general knowledge, but you must specify that the information are not fromthe provided context.'"""
)

human_msg_template = HumanMessagePromptTemplate.from_template(
    template="{input}"
    )
```

LangChain also provides MessagesPlaceholder, which gives you full control of what messages to be rendered during formatting. This can be useful when you are uncertain of what role you should be using for your message prompt templates or when you wish to insert a list of messages during formatting.

```Python
prompt_template = ChatPromptTemplate.from_messages([
    system_msg_template, 
    MessagesPlaceholder(variable_name="history"), 
    human_msg_template
    ])
```

Every call to a LLM is independent of previous calls. Therefore, by default, an LLM doesnt remember/have access to previous responses it has provided. A common fix for this is to include the conversation so far as part of the prompt sent to the LLM. In LangChain, this is achieved through a combination fo ConversationChain and ConversationBufferMemory classes. A typical implementation looks like this:

```Python
conversation = ConversationChain(
    memory=st.session_state.buffer_memory, 
    prompt=prompt_template, 
    llm=llm, 
    verbose=True
    )
```
Here, we use the ConversationChain class to implement a LangChain chain that allows the addition of a memory to this workflow. The ConversationChain and ConversationBufferMemory ensure that every interaction between the user and the agent is logged and this entire history is incorporated into the prompt through the ‘history’ key of the input dict. [19]

```Python
response = conversation.predict(
                    input=f"Context:\n {context} \n\n Query:\n{refined_query}"
                    )

st.session_state.requests.append(refined_query)
st.session_state.responses.append(response) 
```

## Print the chat (Streamlit)

The function `get_conversation_string()` loops through `st.session_state['requests']` and `st.session_state['responses']` at `[i]` and `[i+1]` simply because responses is initialised by a welcome message, therefore the correct indexing matching between requests and responses is `i` and `i+1`.

```Python
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        conversation_string += "Human: "+ st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string
```

In the `main.py` the following call is made to create the final refined query:

```Python
with textcontainer:
        query = st.chat_input("", key="input")
        if query:
            conversation_string = get_conversation_string()
            refined_query = query_refiner(conversation_string, query)
```

At every iteration the chat is printed in the response container with the following instructions:

```Python
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
```

# 🚀 Results

![Img1](docs/image1.jpg)

# ✨ Deploy the app

1. Create the repository for the app. Your repository should contain two files:
    ```
    your-repository/
    ├── main.py
    ├── utils.py
    └── requirements.txt
    ```

2. Set Up the environment by installing `requirements.txt`:
    ```
    pip install -r requirements.txt
    ```

3. Run the app by launching this command:
    ```
    python -m streamlit run main.py
    ```
# References

1. [Tutorial | Chat with any Website using Python and Langchain](https://www.youtube.com/watch?v=bupx08ZgSFg&ab_channel=AlejandroAO-Software%26Ai)

2. [How to Build a Retrieval-Augmented Generation Chatbo](https://www.anaconda.com/blog/how-to-build-a-retrieval-augmented-generation-chatbot)

3. [General structure of this post](https://github.com/umbertogriffo/rag-chatbot?tab=readme-ov-file)

4. [Stanford CS25: V3 I Retrieval Augmented Language Models](https://www.youtube.com/watch?v=mE7IDf2SmJg&t=16s&ab_channel=StanfordOnline)

5. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://doi.org/10.48550/arXiv.2005.11401)

6. [Open Source LLMs: Viable for Production or a Low-Quality Toy?](https://www.anyscale.com/blog/open-source-llms-viable-for-production-or-a-low-quality-toy)

7. [A High-level Overview of Large Language Models](https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/)

8. [Contemporary Large Language Models LLMs](https://www.kaggle.com/code/abireltaief/contemporary-large-language-models-llms)

9. [AI Chip Market](https://research.aimultiple.com/ai-chip-makers/)

10. [Building an Interactive Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit](https://blog.futuresmart.ai/building-an-interactive-chatbot-with-langchain-chatgpt-pinecone-and-streamlit)

11. [Large language model](https://en.wikipedia.org/wiki/Large_language_model)

12. [What is RAG? RAG + Langchain Python Project: Easy AI/Chat For Your Docs](https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami)

13. [Vector database](https://en.wikipedia.org/wiki/Vector_database)

14. [What is a vector database?](https://learn.microsoft.com/en-us/semantic-kernel/memories/vector-db)

15. [Purpose of the “system” role in OpenAI chat completions API](https://community.openai.com/t/purpose-of-the-system-role-in-openai-chat-completions-api/497739)

16. [Semantic Search](https://blog.dataiku.com/semantic-search-an-overlooked-nlp-superpower)

17. [Langchain Memory with LLMs for Advanced Conversational AI and Chatbots](https://blog.futuresmart.ai/langchain-memory-with-llms-for-advanced-conversational-ai-and-chatbots)

18. [Conversational Memory with Langchain](https://medium.com/@michael.j.hamilton/conversational-memory-with-langchain-82c25e23ec60)

19. [Breaking down LangChain : ChatOpenAI and ConversationChain](https://medium.com/@RSK2327/breaking-down-langchain-chatopenai-and-conversationchain-03565f421f78)

20. [Build an LLM app using LangChain](https://docs.streamlit.io/knowledge-base/tutorials/llm-quickstart)
