# Chat with any website using LangChain, Pinecone and Streamlit via Retrieval-Augmented Generation

Examples:

https://github.com/weaviate/Verba

https://github.com/PavloFesenko/gif_analyzer?tab=readme-ov-file#Introduction

1. [Intro](#problems-of-large-pre-trained-language-models)

2. [What RAG are](#rag-retrieval-augmented-generation)

3. [Code explanation](#code-explanation)

4. [Results]()

5. [Quickstart](#deploy-the-app)

6. [References](#references)

# Introduction

## Large Language Models Inference

A language model is built to process and understand a text input (prompt), and then generate a text output (response) accordingly. These models are usually trained on an extensive corpus of unlabeled text, allowing them to learn general linguistic patterns and acquire a wide knowledge base. The primary distinction between a regular language model and a large language model lies in the number of parameters used.

![Schema3](docs/schema3.jpg)

## Problems of large pre-trained language models

Large pre-trained language models have been shown to store factual knowledge in their parameters, and achieve state-of-the-art results when fine-tuned on downstream NLP tasks. However, their ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures [5].

**Problems**:
- one
- two
- three

#### Why are LLMs more accurate with documents rather than pure hallucination from general context?
This is important because what is happening is that some documents are indexed and retrieved, so the LLM can sum up them. But why is that more accurate than an hallucination? LLMs are only able to hallucinate or also doing other stuff such as summing up coherently an existing text?

RAG can be particularly useful when the pre-trained language model alone may not have the necessary information to generate accurate or sufficiently detailed responses since standard general language models are not capable of accessing post-training/external information directly [2].

In conclusion RAGs find their true motivation, in delimiting the LLM to act only on a limited set of data, making fine-tuning not strictly necessary, resulting in time saving and cost saving, even if there is a threshold where fine-tuning would be preferable (the bot must be generalistic).

### Large Langage Models

# Retrieval-Augmented Generation (RAG)

## RAG in general

A RAG (Retrieval-Augmented Generation) application refers to a class of generative models that enhance their output by incorporating external information. The process involves two main steps: retrieval and generation. First, given a prompt or query, the model retrieves relevant information from a large dataset or knowledge base. This retrieved context is then attached to the original prompt, effectively expanding the model's context window with pertinent information. In the second step, the generative component of the model uses this augmented prompt to generate a response or output [Original ChatGPT].

Paradigm:
- Closed book vs open book [4]
- Grounding means having less hallucinations, and that it's possible to do citacions and attributions by pointing back to the source [4]

![RAG_Architecture](docs/rag_expand.jpg) [7]

## Contextualisation: Frozen RAG

https://www.youtube.com/watch?v=tcqEUSNCn8I&ab_channel=pixegami

![RAG Frozen](docs/frozen_rag.jpg)
![RAG Diagram 1](docs/RAG_diag_1.jpg)

Typical RAG setup:

- **Retrieval**: Given a user query or prompt, the system searches through a knowledge source (a vector store with text embeddings) to find relevant documents or text snippets. The retrieval component typically employs some form of similarity or relevance scoring to determine which portions of the knowledge source are most pertinent to the input query [2].

![Retriever](docs/retrieval.jpg)

- **Generation**: The retrieved documents or snippets are then provided to a large language model, which uses them as additional context for generating a more detailed, factual, and relevant response [2].


## Engineering RAG

Engineering a RAG is a complex task, and a good start is this paper:

![Engineering](docs/engineering_rag.jpg)

## Project diagram

![RAG Diagram](docs/RAG_diagram.jpg)

# Code explanation

https://blog.futuresmart.ai/building-an-interactive-chatbot-with-langchain-chatgpt-pinecone-and-streamlit

https://www.youtube.com/watch?v=nAKhxQ3hcMA&ab_channel=PradipNichite

## Create Graphical User Interface

## Create Chat Component 

## Document indexing

## Semantic search (or relevant introduction for RAGs)

https://blog.dataiku.com/semantic-search-an-overlooked-nlp-superpower

## Chat interface with Streamlit

## Embeddings: OpenAIEmbeddings() or SentenceTransformerEmbeddings()?

Embeddings dimension depend from the embedding model, that has to match Pinecone Vector Store dimension

![emb2](docs/emb2.jpg)

![emb1](docs/emb1.jpg)

## Retrieving answers

## Langchain Memory with LLMs for Advanced Conversational AI and Chatbots

https://blog.futuresmart.ai/langchain-memory-with-llms-for-advanced-conversational-ai-and-chatbots

# Results (the app)

# Deploy the app

## Environment requirements

## Running the app

# References

1. https://www.youtube.com/watch?v=bupx08ZgSFg&ab_channel=AlejandroAO-Software%26Ai

2. https://www.anaconda.com/blog/how-to-build-a-retrieval-augmented-generation-chatbot

3. [General structure of this post](https://github.com/umbertogriffo/rag-chatbot?tab=readme-ov-file)

4. [Stanford CS25: V3 I Retrieval Augmented Language Models](https://www.youtube.com/watch?v=mE7IDf2SmJg&t=16s&ab_channel=StanfordOnline)

5. [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
](https://doi.org/10.48550/arXiv.2005.11401)

6. [Open Source LLMs: Viable for Production or a Low-Quality Toy?](https://www.anyscale.com/blog/open-source-llms-viable-for-production-or-a-low-quality-toy)

7. [A High-level Overview of Large Language Models](https://www.borealisai.com/research-blogs/a-high-level-overview-of-large-language-models/)

8. [Contemporary Large Language Models LLMs](https://www.kaggle.com/code/abireltaief/contemporary-large-language-models-llms)

9. [AI Chip Market](https://research.aimultiple.com/ai-chip-makers/) 

10. [Building an Interactive Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit](https://blog.futuresmart.ai/building-an-interactive-chatbot-with-langchain-chatgpt-pinecone-and-streamlit)