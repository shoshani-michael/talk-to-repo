# Talk to Repo

Talk to any GitHub repository and ask questions about the codebase. This app provides an interactive way to explore and understand open-source projects.

The app is publicly hosted here: https://talk-to-repo.vercel.app/. Instructions for hosting it yourself are provided below.

## Features

- Load any GitHub repository
- Input a GitHub API token for authentication and increased rate limits
- Docker support for easy deployment

## Basic architecture

The app is a NextJS/Tailwind CSS frontend with a FastAPI backend. The frontend is hosted on Vercel, and the backend is hosted on a small node on [fly.io](https://fly.io/). The backend uses a Pinecone vector DB on the free tier.

Given any GitHub repository, this app allows users to chat, fetch information from the repository, and display it in a user-friendly conversation interface.

## Architecture Overview

The Talk to Repo app consists of two main components:

1. **Frontend**: A web interface built using Next.js and Tailwind CSS for a modern, responsive design. Users interact with this interface to input repository information, authenticate using their GitHub API tokens, and ask questions about the content of the repository.

   Key frontend components include:

   - `Header.js`: Displays the header with project title and repository link.
   - `ChatMessages.js`: Renders chat messages, including code snippets from the repository.
   - `GitHubInput.js`: Provides input fields for users to enter GitHub repository and API token information.

2. **Backend**: A FastAPI server responsible for processing user input, fetching data from the GitHub repository, and returning relevant repository information as chat messages.

   Key backend components include:

   - Endpoints for loading repositories
   - Endpoints for processing chat requests
   - Endpoints for fetching CSV data

### Communication

The frontend communicates with the backend via RESTful APIs. User input and repository details are sent as API requests to the backend, which processes the requests, fetches relevant data from the GitHub repository, and returns the responses as chat messages to be displayed by the frontend.

### Deployment

The frontend is hosted on Vercel, and the backend is hosted on a small node on [fly.io](https://fly.io/). Instructions for deploying the app locally, as well as launching it with Docker, are provided in the README file.


## Running locally

Follow these steps to set up and run the Talk to Repo app locally.

1. Set up environment variables:

  Copy the .env.template file to create a .env file:

```bash
cp .env.template .env
cd backend
cp .env.template .env
```

 Replace the placeholder values with your actual API keys, organization ID, and other configuration values as needed. Save the file when you're done.

2. Install Node dependencies

```
npm i
```

3. Run the Node server

```
npm run dev
```

4. In another terminal, install the Python dependencies

```
# in the backend/ directory
cd backend 
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
```

6. Set up a Pinecone index. Give it a vector dimension of 1536 and name it `pinecone-index`. You can change this in `backend/main.py` if you want.

7. Run the backend server

```
uvicorn main:app --reload
```

Now, you can access the app at `http://localhost:3000`.

## Launching with Docker

To run the Talk to Repo app using Docker, build the Docker image using the provided Dockerfile and then run a container with the built image. Make sure you have Docker installed on your system before proceeding.

## Potential improvements

- The dependency on Pinecone could be removed and replaced with a simple NumPy array. I just wanted to try Pinecone.
- Replace the `chat_stream` endpoint with a WebSocket implementation to improve real-time communication.
- Ask the model not to generatively reference its sources. Instead, simply copy the code snippet directly.
- The splitter could be improved. Right now, it's a character splitter that favors newlines, but OpenAI has implemented a similar one that splits on tokens instead.
- Get inspired by Replit's Ghostwriter for the embeddings and retrieval mechanisms, to account for the hierarchy of code structure
