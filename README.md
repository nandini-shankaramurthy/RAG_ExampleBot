# RAG_ExampleBot
This chatbot uses Retrieval-Augmented Generation (RAG). It splits a document into chunks, embeds them with a transformer model, and stores them in a FAISS vector DB. When queried, it retrieves relevant info and uses Flan-T5 to generate accurate responses based on the context.
