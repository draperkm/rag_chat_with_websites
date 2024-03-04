# Chat with Websites using LangChain, Streamlit and Pinecone

- https://www.youtube.com/watch?v=nAKhxQ3hcMA&ab_channel=PradipNichite

## Why this project

LLMs are becoming hugely predominant in the ML 

Interesting question from ? webinar on youtube:


https://www.anyscale.com/blog/open-source-llms-viable-for-production-or-a-low-quality-toy

- Originality: RAGs are applications that are very good for when there is a big amount of text data to interact with. But it is possible even to push it further and interact with other type of files, such as images, audio, video, etc
- Nature of RAGs: RAGs find their true motivation, in delimiting the LLM to act only on a limited set of data, making fine-tuning not strictly necessary, resulting in time saving and cost saving, even if there is a threshold where fine-tuning would be preferable (the bot must be generalistic).

### Why are LLMs more accurate with documents rather than pure hallucination from general context?
This is important because what is happening is that some documents are indexed and retrieved, so the LLM can sum up them. But why is that more accurate than an hallucination? LLMs are only able to hallucinate or also doing other stuff such as summing up coherently an existing text?

## Semantic search

https://blog.dataiku.com/semantic-search-an-overlooked-nlp-superpower

## What is a RAG?

https://github.com/umbertogriffo/rag-chatbot?tab=readme-ov-file

https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami

![RAG Diagram](docs/RAG_diagram.jpg)

## Document indexing

## Chatbot applications with Streamlit

## Embeddings: OpenAIEmbeddings() or SentenceTransformerEmbeddings()?

Embeddings dimension depend from the embedding model, that has to match Pinecone Vector Store dimension

![emb2](docs/emb2.jpg)

![emb1](docs/emb1.jpg)

## Retrieving answers

## Langchain Memory with LLMs for Advanced Conversational AI and Chatbots

https://blog.futuresmart.ai/langchain-memory-with-llms-for-advanced-conversational-ai-and-chatbots

## Environment requirements

## Deploy the app

## References

https://github.com/umbertogriffo/rag-chatbot?tab=readme-ov-file
