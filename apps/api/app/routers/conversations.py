import os
import re
import json
import requests
from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import TypeAdapter
from pgvector.sqlalchemy import Vector
from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, select, func
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from typing import List, Optional
from datetime import datetime
import uuid
import httpx
import os
from utils.embeddings import generate_embedding_from_text
from models.document import Document
from models.conversation import Conversation, ConversationCreateRequest


# ─── CONFIG ─────────────────────────────────────────────────────

# These environment variables should be set automatically by docker model runner
# https://docs.docker.com/compose/how-tos/model-runner/
AI_RUNNER_URL = os.getenv("AI_RUNNER_URL", "")
AI_RUNNER_MODEL = os.getenv("AI_RUNNER_MODEL", "")

DATABASE_URL = os.getenv("POSTGRES_CONNSTR", "postgresql+asyncpg://user:password@localhost/dbname")

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# ─── DEPENDENCY ─────────────────────────────────────────────────

async def get_session() -> AsyncSession:
    async with async_session() as session:
        yield session



async def find_closest_documents(session: AsyncSession, embedding: list[float], limit: int = 5):
    stmt = (
        select(Document)
        .order_by(Document.embedding.l2_distance(embedding))
        .limit(limit)
    )
    result = await session.execute(stmt)
    return result.scalars().all()

def call_llm_api(user_message):
    print("------------------------")
    print(f"Calling LLM API with message: {user_message}")
    print(f"AI_RUNNER_URL: {AI_RUNNER_URL}")
    print(f"AI_RUNNER_MODEL: {AI_RUNNER_MODEL}")

    """Calls the LLM API and returns the response"""
    chat_request = {
        "model": AI_RUNNER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        #"max_tokens": 512,
        "temperature": 0.7
    }
    
    headers = {"Content-Type": "application/json"}
    
    # Send request to LLM API
    print(f"Sending request to: {f'{AI_RUNNER_URL}chat/completions'}")
    response = requests.post(
        f'{AI_RUNNER_URL}chat/completions',
        headers=headers,
        json=chat_request,
        timeout=60
    )
    
    # Check if the status code is not 200 OK
    if response.status_code != 200:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")
    
    # Parse the response
    print(response.text)

    print("------------------------")
    print("Parsing response...")
    chat_response = response.json()
    print(chat_response)
    print("------------------------")

    # Extract the assistant's message
    if chat_response.get('choices') and len(chat_response['choices']) > 0:
        return chat_response['choices'][0]['message']['content'].strip()
    
    raise Exception("No response choices returned from API")


def extract_md_code(response_text: str) -> str:
    match = re.search(r"```\w*\s*\n(.*?)\n```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

# ─── ROUTER ─────────────────────────────────────────────────────

router = APIRouter(prefix="/conversations", tags=["Conversations"])

@router.get("/")
async def get_root() -> dict:
    return {'status': 'OK'}


@router.post("/", response_model=str)
async def create_or_update_conversation(
    response: Response,
    body: ConversationCreateRequest,
    session: AsyncSession = Depends(get_session),
):
    # 1. Compute embedding for user message
    user_embedding = generate_embedding_from_text(body.message)

    # 2. Find relevant documents via embedding similarity
    relevant_docs = await find_closest_documents(session, user_embedding, 5)

    # 3. Construct context from relevant docs (e.g., concatenate content)
    context_texts = []
    for doc in relevant_docs:
        content_snippet = doc.content.decode("utf-8", errors="ignore")#[:500]  # first 500 chars
        context_texts.append(f"Document '{doc.name}': {content_snippet}")
    context = "\n\n".join(context_texts)

    # 4. Build prompt with context + user message    
    prompt = f"""You are an expert coding assistant named Milton. You are not allowed to use bad language. You may use jokes and humor in your responses.

- Your response should be in valid markdown format.
- Code blocks should be enclosed in triple backticks with the language specified.
- No more than one paragraph explanation is allowed before the code block.

Use the following documents as context to generate the answer:
{context}

Question: {body.message}
Answer:"""
    
    # 5. Query LLM
    #llm_response = await query_llm(prompt)
    llm_response = call_llm_api(prompt)

    # 6. Save conversation to database
    new_conv = Conversation(
        request={'prompt': prompt},
        response={'content': llm_response},
    )

    session.add(new_conv)
    await session.commit()

    return PlainTextResponse(llm_response)
