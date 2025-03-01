import os
import json

from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
)  # extract text content from pdf
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # splits the txt in chunks
from langchain.embeddings.huggingface import (
    HuggingFaceEmbeddings,
)  # transform text in embeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# this will be executed in a different directory this abs path gives absolute path o the rag utitlty file
working_dir = os.path.dirname(os.path.abspath((__file__)))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]
# copying from envirnoment variable and pasting it here to secure it
os.environ["GROQ_API_KEY"] = GROQ_API_KEY


# loading the embedding mdel
embedding = HuggingFaceEmbeddings()

# load the llm from groq
llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0)


def process_document_to_chroma_db(file_name):
    # load the document using unstructured librabry
    loader = UnstructuredPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    # splitting in chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

    texts = text_splitter.split_documents(documents)
    # converted to vectror embeddings
    vectorDb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/docs_vectorStore",
    )
    return 0

def answer_question(user_question):
    #here load the persistent vectordb
    vectorDb=Chroma(persist_directory=f"{working_dir}/doc_vectorStore",embedding_function=embedding)

    #create a retriever
    retriever=vectorDb.as_retriever()

    #create a chain to answer questions using deepseek
    qa_chain=RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    # take the question from the user
    response=qa_chain.invoke({"query":user_question})
    answer=response["result"]

    return answer