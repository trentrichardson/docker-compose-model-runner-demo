import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI
import uvicorn
import logging

base_dir = os.path.abspath('./app')

load_dotenv()

logging.basicConfig(
    level=logging.INFO if os.environ.get('ENV') == 'lcl' else logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from routers import (
    documents,
    conversations,
)


app = FastAPI(
    title="Docker Compose Demo API",
    version="0.0.1",
)


@app.get("/")
def read_root():
    return "Connected to Docker Compose Demo API"

app.include_router(documents.router)
app.include_router(conversations.router)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ['API_PORT']),
        reload=True
    )