import os

import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

DATA_DIR = "./../data"


# Create vector database
def create_vector_database():
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings using HuggingFace,
    and finally persists the embeddings into a Chroma vector database.

    """

    # Initialize loaders for different file types
    """ pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    markdown_loader = DirectoryLoader(
        DATA_DIR, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)

    all_loaders = [pdf_loader, markdown_loader, text_loader]

    # Load documents from all loaders
    loaded_documents = []
    for loader in all_loaders:
        loaded_documents.extend(loader.load()) """

    text_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    loaded_documents = text_loader.load()

    # Split loaded documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    # Initialize HuggingFace embeddings
    huggingface_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # type: ignore
        environment=PINECONE_ENV,  # type: ignore
    )

    # giving index a name
    index_name = "llama2-langchain"

    # delete index if same name already exists
    if index_name in pinecone.list_indexes():
        pinecone.delete_index(index_name)

    # create index
    pinecone.create_index(name=index_name, dimension=384, metric="cosine")

    # Create and store a pinecone vector database from the chunked documents
    vector_database = Pinecone.from_documents(
        documents=chunked_documents,
        embedding=huggingface_embeddings,
        index_name=index_name,
    )


if __name__ == "__main__":
    create_vector_database()
