#!/bin/bash

npm run dev &
cd backend
source .venv/bin/activate
uvicorn main:app --reload  &
