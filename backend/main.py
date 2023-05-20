import os
import threading
import queue
import shutil
import patch

from create_vector_db import create_vector_db

from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

import pinecone
import tiktoken
import os
import pandas as pd


# Import dotenv and load the variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://frontend",
    "http://frontend:3000",
]

additional_origins = os.environ["ALLOWED_ORIGINS"].split(',')
origins.extend(additional_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import tempfile

pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment=os.environ['ENVIRONMENT']
)

class RepoInfo(BaseModel):
    hostingPlatform: str
    username: str
    repo: str
    token: Optional[str] = None

class Message(BaseModel):
    text: str
    sender: str

class ContextSystemMessage(BaseModel):
    system_message: str
    
class Chat(BaseModel):
    messages: list[dict]

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

encoder = tiktoken.get_encoding('cl100k_base')

import tempfile
import json


def get_local_repo_path():
    if 'LOCAL_REPO_PATH' in os.environ:
        print("Using LOCAL_REPO_PATH from environment variable")
        LOCAL_REPO_PATH = os.environ['LOCAL_REPO_PATH']
    else:
        print("Using LOCAL_REPO_PATH from cache file")
        if os.path.exists(".talk-to-repo-cache"):
            with open(".talk-to-repo-cache", "r") as f:
                LOCAL_REPO_PATH = f.read()
        else:
            print("Creating new LOCAL_REPO_PATH")
            # Create a new temporary directory
            LOCAL_REPO_PATH = tempfile.mkdtemp()
   
    # keep the path in a cache file and environment variable
    with open(".talk-to-repo-cache", "w") as f:
        f.write(LOCAL_REPO_PATH)
        os.environ['LOCAL_REPO_PATH'] = LOCAL_REPO_PATH
    
    print(f"Using LOCAL_REPO_PATH: {LOCAL_REPO_PATH}")
    return LOCAL_REPO_PATH

LOCAL_REPO_PATH = get_local_repo_path()

def create_tempfile_with_content(content):
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    temp_file.write(content)
    temp_file.close()
    return temp_file.name


def format_context(docs, LOCAL_REPO_PATH):
    # Load corpus_summary.csv
    corpus_summary = pd.read_csv("data/corpus_summary.csv")

    aggregated_docs = {}

    for d in docs:
        document_id = d.metadata["document_id"]
        if document_id in aggregated_docs:
            aggregated_docs[document_id].append(d.page_content)
        else:
            aggregated_docs[document_id] = [d.page_content]

    context_parts = []
    for i, (document_id, content_parts) in enumerate(aggregated_docs.items()):
        # Calculate the token count of the entire file
        entire_file_token_count = corpus_summary.loc[corpus_summary["file_name"] == document_id]["n_tokens"].values[0]

        # Calculate the token count of the content_parts
        content_parts_token_count = sum([len(encoder.encode(cp)) for cp in content_parts])

        if content_parts_token_count / entire_file_token_count > 0.5:
            # Include the entire file contents instead
            fname = LOCAL_REPO_PATH + "/" + document_id
            with open(fname, "r") as f:
                file_contents = f.read()
            context_parts.append(f'[{i}] Full file {document_id}:\n' + file_contents)
        else:
            # Include only the content_parts
            context_parts.append(f'[{i}] From file {document_id}:\n' + "\n---\n".join(content_parts))

    context = "\n\n".join(context_parts)

    return context

def format_query(query, context):
    return f"""Relevant context: {context}
    
    {query}"""

def embedding_search(query, k):
    embeddings = OpenAIEmbeddings(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        openai_organization=os.environ['OPENAI_ORG_ID'],
    )
    docsearch = Pinecone.from_existing_index(
        os.environ['PINECONE_INDEX'], 
        embeddings,
        text_key='text',
        namespace=os.environ['NAMESPACE']
    )

    return docsearch.similarity_search(query, k=k)

import subprocess
import openai

def extract_key_words(query: str) -> str:

    prompt = f"Extract from the following query the key words, \
        which will be used to grep a codebase. \
        Return the key words as a comma-separated list. \
        Query: {query}"
    
    openai.api_key = os.environ['OPENAI_API_KEY']
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.5,
    )

    key_words = response.choices[0].text.strip()
    return key_words

def get_last_commits_messages(repo_path: str, n: int = 20) -> str:
    result = subprocess.run(
        ["git", "-C", repo_path, "log", f"-n {n}", "--pretty=format:%s%n%n%h %cI%n----%n", "--name-status"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise Exception(f"Error getting commit messages: {result.stderr.decode('utf-8')}")

    commit_messages = result.stdout.decode("utf-8").split("\n")

    return "\n".join(commit_messages)

def get_last_commit(repo_path):
    result = subprocess.run(
        ["git", "-C", repo_path, "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    if result.returncode != 0:
        raise Exception(f"Error getting the last commit hash: {result.stderr.decode('utf-8')}")
    
    last_commit_hash = result.stdout.decode("utf-8").strip()
    return last_commit_hash

@app.get("/health")
def health():
    return "OK"

@app.post("/system_message", response_model=ContextSystemMessage)
def system_message(query: Message):
    numdocs = int(os.environ['CONTEXT_NUM'])
    docs = embedding_search(query.text, k=numdocs)
    context = format_context(docs, LOCAL_REPO_PATH)
    grep_context = grep_more_context(query)
    last_commits = get_last_commits_messages(LOCAL_REPO_PATH, 5)
 
    prompt = """Given the following context and code, answer the following question. Do not use outside context, and do not assume the user can see the provided context. \
    Try to be as detailed as possible and reference the components that you are looking at. Keep in mind that these are only code snippets, and more snippets may be added during the conversation. \
    When generating code, if it's a code change, produce a concise diff. \
    Please pay attention to produce the correct indentation so that the diffs apply correctly. \
    The markup should be of type `diff` (after the three backticks), and then start with `diff --git`. \
    Don't include an index line. When counting the number of lines in the patch, make sure not to add a line for the line count itself. \
    Also don't break lines inside html tags, it would create bad diffs.\
    The generated diff will later be applied to the code, as is. 

    For example, here's a correctly generated diff:

    ```diff
    diff --git a/relative/path/file.py b/relative/path/file.py
    --- a/relative/path/file.py
    +++ b/relative/path/file.py
    @@ -1234,5 +1234,6 @@ def say_hello():
         a = "hello"
         b = " everyone"
    -    print(a, b)
    +    print(a, b, "!")
    +    print("hello again")
         c = 1 + 2
         return c
    ```

    When writing existing or new code (not diffs), make sure to specify the language of the code. \
    For example, if you were generating Python, you would write the following:

    ```python
    # relative/path/file.py line: 1234 (the line number is optional, no leading slash in file path)
    # a line of context right before the relevant code
    <python code goes here>
    # a line of context right after the relevant code
    ```
    
    Now, here is the relevant context, each context piece is seperated by a line of dashes: 

    Context: {context}

    Grep Context: {grep_context}

    Commit messages: {last_commits}
    """

    return {'system_message': prompt.format(
        context=context,
        grep_context=grep_context,
        last_commits=last_commits
    )}

def clear_local_repo_path():
    global LOCAL_REPO_PATH

    # Remove the existing temporary directory
    shutil.rmtree(LOCAL_REPO_PATH)

    # Create a new temporary directory
    LOCAL_REPO_PATH = tempfile.mkdtemp()

    # Update the cache file and environment variable
    with open(".talk-to-repo-cache", "w") as f:
        f.write(LOCAL_REPO_PATH)
        os.environ['LOCAL_REPO_PATH'] = LOCAL_REPO_PATH

def grep_more_context(query):
    key_words = extract_key_words(query)
    print(f"Key words: {key_words}")
    context_from_key_words = ""
    for keyword in key_words.split(','):
        keyword = keyword.strip()
        print(f"Working in directory: {LOCAL_REPO_PATH}")
        output = subprocess.run(['git', 'grep', '-C 5', '-h', '-e', keyword, '--', '.' ], 
            cwd=LOCAL_REPO_PATH,  
            stdout=subprocess.PIPE,         
            stderr=subprocess.PIPE,
        ).stdout

        # Add output from git grep command to the overall context
        context_from_key_words += output.decode('utf-8') + "\n\n"
        print(f"Context from key word: {context_from_key_words}")
    
    # limit context to 1000 characters
    context_from_key_words = context_from_key_words[:1000]

    return context_from_key_words

@app.post("/chat_stream")
async def chat_stream(chat: List[Message]):
    model_name = os.environ['MODEL_NAME']
    encoding_name = 'cl100k_base'

    def llm_thread(g, prompt):
        try:
            llm = ChatOpenAI(
                model_name=model_name,
                verbose=True,
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                temperature=os.environ['TEMPERATURE'],
                openai_api_key=os.environ['OPENAI_API_KEY'],
                openai_organization=os.environ['OPENAI_ORG_ID']
            )

            encoding = tiktoken.get_encoding(encoding_name)

            # the system message gets 10 new docs. Only include 2 more for new queries
            if len(chat) > 2:
                system_message, latest_query = [chat[0].text, chat[-1].text]
                keep_messages = [system_message, latest_query]
                new_messages = []

                token_count = sum([len(encoding.encode(m)) for m in keep_messages])
                # fit in as many of the previous human messages as possible
                for message in chat[1:-1:2]:
                    token_count += len(encoding.encode(message.text))

                    if token_count > 750:
                        break

                    new_messages.append(message.text)
                    
                query_messages = [system_message] + new_messages + [latest_query]
                query_text = '\n'.join(query_messages)

                context_from_key_words = grep_more_context(latest_query)

                # add some more context
                docs = embedding_search(query_text, k=5)
                context = format_context(docs, LOCAL_REPO_PATH)
                formatted_query = format_query(latest_query, context + context_from_key_words)
            else:
                formatted_query = chat[-1].text

            # always include the system message and the latest query in the prompt
            system_message = SystemMessage(content=chat[0].text)
            latest_query = HumanMessage(content=formatted_query)
            messages = [latest_query]

            # for all the rest of the messages, iterate over them in reverse and fit as many in as possible
            token_limit = int(os.environ['TOKEN_LIMIT'])
            num_tokens = len(encoding.encode(chat[0].text)) + len(encoding.encode(formatted_query))
            for message in reversed(chat[1:-1]):
                # count the number of new tokens
                num_tokens += 4
                num_tokens += len(encoding.encode(message.text))

                if num_tokens > token_limit:
                    # if we're over the token limit, stick with what we've got
                    break
                else:
                    # otherwise, add the new message in after the system prompt, but before the rest of the messages we've added
                    new_message = HumanMessage(content=message.text) if message.sender == 'user' else AIMessage(content=message.text)
                    messages = [new_message] + messages

            # add the system message to the beginning of the prompt
            messages = [system_message] + messages

            llm(messages)

        finally:
            g.close()

    def chat_fn(prompt):
        g = ThreadedGenerator()
        threading.Thread(target=llm_thread, args=(g, prompt)).start()
        return g

    return StreamingResponse(chat_fn(chat), media_type='text/event-stream')

@app.post("/load_repo")
def load_repo(repo_info: RepoInfo):
    clear_local_repo_path()

    print(f"Loading repo: {repo_info.repo}")
    if repo_info.hostingPlatform == "github":
        if repo_info.token:
            REPO_URL = f"https://{repo_info.token}@github.com/{repo_info.username}/{repo_info.repo}.git"
        else:
            REPO_URL = f"https://github.com/{repo_info.username}/{repo_info.repo}.git"
    elif repo_info.hostingPlatform == "gitlab":
        if repo_info.token:
            REPO_URL = f"https://oauth2:{repo_info.token}@gitlab.com/{repo_info.username}/{repo_info.repo}.git"
        else:
            REPO_URL = f"https://gitlab.com/{repo_info.username}/{repo_info.repo}.git"
    elif repo_info.hostingPlatform == "bitbucket":
        if repo_info.token:
            REPO_URL = f"https://x-token-auth:{repo_info.token}@bitbucket.org/{repo_info.username}/{repo_info.repo}.git"
        else:
            REPO_URL = f"https://bitbucket.org/{repo_info.username}/{repo_info.repo}.git"
    else:
        return {"status": "error", "message": "Invalid hosting platform"}

    os.environ['GITHUB_TOKEN'] = repo_info.token

    pinecone_index = os.environ['PINECONE_INDEX']
    namespace = os.environ['NAMESPACE']

    index = pinecone.Index(pinecone_index)
    delete_response = index.delete(delete_all=True, namespace=namespace)
    
    create_vector_db(REPO_URL, LOCAL_REPO_PATH)

    last_commit = get_last_commit(LOCAL_REPO_PATH)
    return {"status": "success", "message": "Repo loaded successfully", "last_commit": last_commit}

def call_gpt3(query, max_tokens, n=1, temperature=0.0) -> str:
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens=max_tokens,
        n=n,
        stop=None,
        temperature=temperature,
    )

    return response.choices[0].text.strip()

def grep_file_from_snippet(snippet: str) -> str:
    # keep first line of snippet:
    snippet = snippet.split("\n")[0]
    query = f"In this code snippet:\n\n{snippet}\n\n What's the full file name? Please only provide the file path and nothing else."
    file_path = call_gpt3(query, 50)
    
    return file_path


def generate_commit_message() -> str:
    query = "Generate a commit message for changes in these(code snippets) files."
    commit_message = call_gpt3(query, 20)

    return commit_message

def apply_diff_to_file(diff: str, file_path: str) -> None:
    # save the diff to temp file
    with open(LOCAL_REPO_PATH + "/temp.diff", "w") as file:
        file.write(diff)
        file.write("\n")

    # debug
    print(f"Applying diff to file: {file_path}, CWD: {LOCAL_REPO_PATH}")

    # call git apply with the diff, and check if it was successful
    result = subprocess.run(
        ["git", "apply", "temp.diff", "--unidiff-zero", "--inaccurate-eof", "--allow-empty", "--ignore-whitespace"], 
        cwd=LOCAL_REPO_PATH,
        capture_output=True
    )
    if not result.returncode == 0:
        raise Exception(f"Failed to apply diff to file: {file_path}")

class CodeDiffs(BaseModel):
    diff: str

def create_commit_from_diffs(diffs: List[CodeDiffs]):
    # Iterate through the diffs
    for code_diff in diffs:
        diff = code_diff.diff
        # Use grep to figure out which file to change
        file_path = grep_file_from_snippet(diff)
        
        # Apply the diff
        apply_diff_to_file(diff, file_path)

    # Use GPT-3 to generate a commit message
    commit_message = generate_commit_message()

    # Create a commit with all the changes
    subprocess.run(["git", "add", "."], cwd=LOCAL_REPO_PATH)
    subprocess.run(["git", "commit", "-m", commit_message], cwd=LOCAL_REPO_PATH)

    return True


@app.post("/create_commit")
def create_commit(diffs: List[CodeDiffs]):
    success = create_commit_from_diffs(diffs)
    return {'status': 'success' if success else 'error'}
