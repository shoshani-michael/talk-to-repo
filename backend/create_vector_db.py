from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

import pandas as pd
import tiktoken


import os
import re
from urllib.request import urlopen
import tempfile
import subprocess

# Import dotenv and load the variables from .env file
from dotenv import load_dotenv
load_dotenv()

embeddings = OpenAIEmbeddings(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    openai_organization=os.environ['OPENAI_ORG_ID'],
)
encoder = tiktoken.get_encoding('cl100k_base')

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['ENVIRONMENT']
)
vector_store = Pinecone(
    index=pinecone.Index(os.environ['PINECONE_INDEX']),
    embedding_function=embeddings.embed_query,
    text_key='text',
    namespace=os.environ['NAMESPACE']
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.environ['CHUNK_SIZE']),
    chunk_overlap=int(os.environ['CHUNK_OVERLAP'])
    )

def clone_from_github(REPO_URL):
    temp_dir = tempfile.mkdtemp()
    repo_url = REPO_URL
    token = os.environ["GITHUB_TOKEN"] if "GITHUB_TOKEN" in os.environ else None

    if token:
        repo_url = repo_url.replace("https://", f"https://x-access-token:{token}@")

    print(f"Cloning {repo_url} into {temp_dir}")
    subprocess.run(["git", "clone", repo_url, temp_dir], check=True)
    return temp_dir

def is_unwanted_file(file_name):
    if (file_name.endswith('/') or 
        any(f in file_name for f in ['.DS_Store', '.gitignore']) or 
        any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.mp3', '.ico'])
    ):
        return True
    return False

def process_file(file):
    file_contents = str(file.read())
    n_tokens = len(encoder.encode(file_contents))
    return file_contents, n_tokens

def process_file_list(temp_dir):
    corpus_summary = []
    file_texts, metadatas = [], []

    for root, _, files in os.walk(temp_dir):
        for filename in files:
            if not is_unwanted_file(filename):
                file_path = os.path.join(root, filename)
                if '.git' in file_path:
                    continue
                with open(file_path, 'rb') as file:
                    print(f'Processing {file_path}')
                    file_contents, n_tokens = process_file(file)
                    file_name_trunc = re.sub(r'^[^/]+/', '', str(filename))
                    corpus_summary.append({'file_name': file_name_trunc, 'n_tokens': n_tokens})
                    file_texts.append(file_contents)
                    metadatas.append({'document_id': file_name_trunc})

    split_documents = splitter.create_documents(file_texts, metadatas=metadatas)
    vector_store.from_documents(
        documents=split_documents,
        embedding=embeddings,
        index_name=os.environ['PINECONE_INDEX'],
        namespace=os.environ['NAMESPACE']
    )

    pd.DataFrame.from_records(corpus_summary).to_csv('data/corpus_summary.csv', index=False)

    return 

def create_vector_db(REPO_URL):
    temp_dir = clone_from_github(REPO_URL)
    process_file_list(temp_dir)

if __name__ == '__main__':
    create_vector_db( os.environ["REPO_URL"] )