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
    embedding_literal = f"[{', '.join(map(str, embedding))}]"

    result = await session.execute(
        text("""
            SELECT id, name, content, content_type, embedding
            FROM documents
            ORDER BY embedding <=> :embedding
            LIMIT :limit
        """),
        {"embedding": embedding_literal, "limit": limit}
    )
    return result.fetchall()

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
        content_snippet = doc.content.decode("utf-8", errors="ignore")[:500]  # first 500 chars
        context_texts.append(f"Document '{doc.name}': {content_snippet}")
    context = "\n\n".join(context_texts)

    # 4. Build prompt with context + user message
    prompt = f"""You are an expert SVG graphic designer.

- The SVG should be valid and well-formed.
- Use simple <rect>, <circle>, <line>, <polygon>, and <path> elements.
- Include inline comments explaining the major parts.
- Make sure the entire SVG is enclosed inside a single <svg> tag.
- Do not explain your output — only return the SVG code inside a markdown code block like this:

```svg
<your SVG code here>
```

Use the following documents as context to generate the svg design:
{context}

Question: {body.message}
Answer:"""
    
#     prompt = f"""You are an expert SVG graphic designer named Rob Villa who excels at woodworking designs.

# - The SVG should be valid and well-formed.
# - If xlink is used, you must include the xlink namespace in the <svg> tag. `xmlns:xlink="http://www.w3.org/1999/xlink"`
# - Use simple <rect>, <circle>, <line>, <polygon>, <text>, and <path> elements.
# - Make sure the entire SVG is enclosed inside a single <svg> tag.
# - Do not explain your output — only return the SVG code inside a markdown code block like this:

# ```svg
# <your SVG code here>
# ```

# Use the following documents as context to generate the svg design:
# {context}
# - Full sheets of wood that can be cut are 4 feet tall by 8 feet wide and should be scaled to 400px tall by 800px wide as the svg canvas.
# - The canvas has a white #ffffff background.
# - All requested cuts are represented as shapes and must be scaled appropriately to the canvas size representing the full sheet of wood.
# - Each cut shape must have be .125 inches space between shapes. Scale this to the canvas size and represent this as a red #ff0000 1px line.
# - Requested cut shapes should be grouped together to optimize and preserve as much unused wood space as possible.
# - Shapes must not overlap each other.
# - Shapes must be completely inside the 400px tall by 800px wide canvas area.
# - Shapes may be rotated to find the optimal fit within the sheet of wood.
# - Measurements requested are all imperial (inches and feet) and must be scaled to the 400px tall by 800px wide canvas.
# - In the center of each cut shape should be the a text label of the imperial size of the cut shape in red #ff0000 san-serif font 10px.

# Question: {body.message}
# Answer:"""

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

    md_code = extract_md_code(llm_response)
    print(llm_response)

    if md_code == "":
        return PlainTextResponse(llm_response)
    else:
        return Response(content=md_code, 
                        media_type="image/svg+xml",
                        headers={"Content-Disposition": 'inline; filename="generated_image.svg"'})
    
    return {
        "status": "success",
        "conversation_id": str(new_conv.id),
        "response": llm_response,
        "md_code": extract_md_code(llm_response)
    }