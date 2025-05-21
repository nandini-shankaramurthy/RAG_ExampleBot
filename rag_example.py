
!pip install -U langchain langchain-community faiss-cpu sentence-transformers transformers

from google.colab import files
uploaded = files.upload()


from langchain_community.document_loaders import TextLoader
#Loads plain text files (e.g., .txt) into a format LangChain understands.
from langchain.text_splitter import CharacterTextSplitter
#Splits large text into smaller overlapping chunks.
from langchain_community.embeddings import HuggingFaceEmbeddings
#Uses HuggingFace sentence transformer models to turn text chunks into vectors.
from langchain_community.vectorstores import FAISS
#Stores those vectors in FAISS, which allows fast similarity search.
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import os

# Automatically getting the uploaded filename
doc_path = list(uploaded.keys())[0]

#Data Ingestion
# Build vectorstore
def build_vector_store(doc_path):
    loader = TextLoader(doc_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)
    #vector DB : FAISS, Qdrant, Pinecone, chroma

# Setup RAG basically a retrival pipeline
def create_rag_chain(vectorstore):
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Simple text input in Colab
def chat(rag_chain):
    while True:
        question = input("You: ")
        if question.lower() in ["exit", "quit"]:
            print("👋 Exiting chat.")
            break
        response = rag_chain.run(question)
        print("Bot:", response)

# Run
vectorstore = build_vector_store(doc_path)
rag_chain = create_rag_chain(vectorstore)
print("✅ Chatbot is ready. Type your question (or 'exit' to quit).")
chat(rag_chain)

