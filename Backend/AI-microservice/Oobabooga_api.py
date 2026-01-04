# =========================
# Imports and Dependencies
# =========================
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uvicorn
import aiohttp
import asyncio
from functools import lru_cache
import threading


# ===================
# Configuration
# ===================
OOBABOOGA_API_URL = "http://127.0.0.1:5000/v1/completions"
DATA_PATH = r"C:\Users\Lenovo\Documents\programing\miniProject\Backend\AI-microservice\software_career_knowledge.json"

# ==========================
# Role-based Prompt Templates
# ==========================
roles = {
    "default": "### Instruction:\n{context}\n{user}\n\n### Response:",
    "Career_mentor": (
        "### Instruction:\nYou are a friendly mentor. Who helps software engineering students, answer their doubts and guide them. "
        "Don't assume anything until they ask and don't mention any fields until they ask. "
        "Try asking a question at the end for your information if needed. Try keeping it short and to the point.\n\n{context}\n{user}\n\n### Response:"
    ),
    "Career_mentor-v2": (
        "### Instruction:\nYou are a friendly software mentor. Only answer what the student asks. "
        "Do not assume the field or topic. Keep answers short and clear. Ask questions only if you need clarification.\n\n{context}\n{user}\n\n### Response:"
    ),
    "USER": (
        "### Instruction:\nYou are a friendly mentor. Who helps software engineering students, answer their doubts and guide them. "
        "Don't assume anything until they ask and don't mention any fields until they ask. "
        "Try asking a question at the end for your information if needed. Try keeping it short and to the point.\n\n{context}\n{user}\n\n### Response:"
    )
}
active_role = "Career_mentor"

# ==========================
# Load Knowledge Base (RAG)
# ==========================
print("📚 Loading knowledge base...")

docs = []
if DATA_PATH.endswith(".json"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            docs = list(data.values())
        elif isinstance(data, list):
            docs = data
elif DATA_PATH.endswith(".txt"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

# Load embedder - try GPU first, fallback to CPU
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')
    print("✅ Using GPU for embeddings")
except:
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
    print("⚠️ Using CPU for embeddings (slower)")

doc_embeddings = embedder.encode(docs, show_progress_bar=True, batch_size=64)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings).astype('float32'))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ================
# Shared HTTP Session (CRITICAL for performance)
# ================
http_session = None

async def get_http_session():
    global http_session
    if http_session is None:
        http_session = aiohttp.ClientSession()
    return http_session

# ================
# Threading for CPU-bound encoding
# ================
executor = None

def encode_query(text):
    """CPU-bound encoding in thread pool"""
    return embedder.encode([text])

# ================
# Greeting detection (faster)
# ================
GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening", "sup", "yo"}

def is_greeting(text):
    """Fast greeting detection"""
    words = text.strip().lower().split()
    return len(words) <= 2 and words[0] in GREETINGS

# ================
# Query cache for repeated questions
# ================
query_cache = {}
MAX_CACHE_SIZE = 100

def get_cached_context(user_input):
    """Cache RAG results for identical queries"""
    cache_key = user_input.lower().strip()
    if cache_key in query_cache:
        return query_cache[cache_key]
    return None

def cache_context(user_input, context):
    """Store RAG result in cache"""
    if len(query_cache) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        query_cache.pop(next(iter(query_cache)))
    query_cache[user_input.lower().strip()] = context

# ================
# FastAPI App
# ================
app = FastAPI(title="Optimized Oobabooga + RAG Chat API")

@app.on_event("startup")
async def startup_event():
    global executor
    executor = asyncio.get_event_loop().run_in_executor
    await get_http_session()
    print("✅ HTTP session initialized")

@app.on_event("shutdown")
async def shutdown_event():
    global http_session
    if http_session:
        await http_session.close()
        print("✅ HTTP session closed")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role

@app.post("/chat")
async def chat(request: Request):
    """Optimized non-streaming chat endpoint"""
    data = await request.json()
    user_input = data.get("userInput")
    role = data.get("role", active_role)

    if not user_input:
        return JSONResponse(
            status_code=400,
            content={"error": "Missing user input"}
        )

    import time
    start_time = time.time()

    # ===== Fast RAG retrieval with caching =====
    if is_greeting(user_input):
        context = ""
    else:
        # Check cache first
        context = get_cached_context(user_input)
        
        if context is None:
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            query_vec = await loop.run_in_executor(None, encode_query, user_input)
            
            D, I = index.search(np.array(query_vec).astype('float32'), k=3)
            retrieved_docs = []
            for i in I[0]:
                if isinstance(docs[i], dict) and "text" in docs[i]:
                    retrieved_docs.append(docs[i]["text"])
                else:
                    retrieved_docs.append(str(docs[i]))
            context = "\n".join(retrieved_docs)
            
            # Cache the result
            cache_context(user_input, context)

    # ===== Build prompt =====
    full_prompt = roles.get(role, roles["Career_mentor"]).format(
        context=context, 
        user=user_input
    )

    # ===== Call Oobabooga with reused session =====
    try:
        payload = {
            "prompt": full_prompt,
            "max_tokens": 150,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.2,
            "stream": False
        }

        session = await get_http_session()
        async with session.post(OOBABOOGA_API_URL, json=payload) as response:
            if response.status != 200:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Oobabooga API error"}
                )
            
            result = await response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                generated_text = result['choices'][0].get('text', '')
                
                end_time = time.time()
                elapsed = end_time - start_time
                
                print(f"✅ Generated response in {elapsed:.2f}s")
                
                return JSONResponse(content={
                    "response": generated_text.strip(),
                    "generation_time": elapsed
                })
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": "No response from model"}
                )
                
    except aiohttp.ClientError as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error connecting to Oobabooga: {str(e)}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error: {str(e)}"}
        )

@app.get("/health")
async def health_check():
    try:
        session = await get_http_session()
        async with session.get("http://127.0.0.1:5000/v1/models") as response:
            if response.status == 200:
                return {
                    "status": "healthy",
                    "backend": "oobabooga-text-generation-webui",
                    "knowledge_base_entries": len(docs),
                    "mode": "non-streaming",
                    "cache_size": len(query_cache),
                    "optimizations": "enabled"
                }
    except:
        return {
            "status": "error",
            "message": "Oobabooga API not reachable. Make sure it's running with API enabled!"
        }

@app.post("/clear-cache")
async def clear_cache():
    """Clear the query cache"""
    query_cache.clear()
    return {"message": "Cache cleared", "size": 0}

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    print("\n🎯 Using Oobabooga (text-generation-webui) as backend")
    print("⚠️  Make sure Oobabooga is running with API enabled")
    print("\n📡 Your API will be available at: http://127.0.0.1:8000")
    print("📖 Docs available at: http://127.0.0.1:8000/docs")
    print("\n⚡ Optimizations enabled:")
    print("   • Persistent HTTP connection pooling")
    print("   • Query result caching")
    print("   • Async embedding encoding")
    print("   • Fast greeting detection")
    uvicorn.run(app, host="127.0.0.1", port=8000)