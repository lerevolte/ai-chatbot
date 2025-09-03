import hashlib
import os
import shutil
from typing import List
from tqdm import tqdm

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings


# Путь к директории для сохранения базы FAISS
FAISS_PATH = "./db_faiss_v1"
DATA_PATH = "./docs"
global_unique_hashes = set()


def walk_through_files(path, file_extension='.txt'):
    for (dir_path, dir_names, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith(file_extension):
                yield os.path.join(dir_path, filename)


def load_documents():
    """
    Load documents from the specified directory
    Returns:
    List of Document objects:
    """
    documents = []
    for f_name in walk_through_files(DATA_PATH):
        document_loader = TextLoader(f_name, encoding="utf-8")
        documents.extend(document_loader.load())

    return documents


def hash_text(text):
    # Generate a hash value for the text using SHA-256
    hash_object = hashlib.sha256(text.encode())
    return hash_object.hexdigest()


def split_text(documents: List[Document]):
    """
    Split the text content of the given list of Document objects into smaller chunks.
    Args:
    documents (list[Document]): List of Document objects containing text content to split.
    Returns:
    list[Document]: List of Document objects representing the split text chunks.
    """
    text_splitter = MarkdownTextSplitter(
        chunk_size=500,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
    )

    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Deduplication mechanism
    unique_chunks = []
    for chunk in chunks:
        chunk_hash = hash_text(chunk.page_content)
        if chunk_hash not in global_unique_hashes:
            unique_chunks.append(chunk)
            global_unique_hashes.add(chunk_hash)

    print(f"Unique chunks equals {len(unique_chunks)}.")

    return unique_chunks  # Return the list of split text chunks


def save_to_faiss(chunks: List[Document]):
    """
    Save the given list of Document objects to a FAISS database with a progress bar.
    """
    # Clear out the existing database directory if it exists
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    # Initialize the embedding function
    embedding_function = OllamaEmbeddings(model="mxbai-embed-large")

    # Set a batch size for processing
    batch_size = 16
    db = None

    # Process chunks in batches with a progress bar
    for i in tqdm(range(0, len(chunks), batch_size), desc="Generating embeddings and saving to FAISS"):
        batch = chunks[i:i+batch_size]
        if not batch:
            continue
        
        if db is None:
            # Create the DB with the first batch
            db = FAISS.from_documents(batch, embedding_function)
        else:
            # Add subsequent batches to the existing DB
            db.add_documents(batch)
            
    # Persist the database to disk if it was created
    if db:
        db.save_local(FAISS_PATH)
        print(f"\nSaved {len(chunks)} chunks to {FAISS_PATH}.")
    else:
        print("No chunks were processed to save.")


def generate_data_store():
    """
    Function to generate vector database in FAISS from documents.
    """
    documents = load_documents()  # Load documents from a source
    chunks = split_text(documents)  # Split documents into manageable chunks
    save_to_faiss(chunks)  # Save the processed data to a data store


if __name__ == "__main__":
    generate_data_store()

