# =========================
# Imports and Dependencies
# =========================
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, TextIteratorStreamer, AutoConfig
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import uvicorn
import threading
import warnings
warnings.filterwarnings('ignore')


# ===================
# Model Configuration
# ===================
MODEL_PATH = r"E:\text-generation-webui-main\text-generation-webui-main\user_data\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"
DATA_PATH = r"C:\Users\Lenovo\Documents\programing\miniProject\Backend\AI-microservice\software_career_knowledge.json"

# ======================
# Load Tokenizer & Model
# ======================
print("🚀 Loading optimized GPTQ model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    use_fast=True,
    local_files_only=True
)

# Load with AutoGPTQ but simpler settings
model = AutoGPTQForCausalLM.from_quantized(
    MODEL_PATH,
    device="cuda:0",
    use_triton=False,
    use_safetensors=True,
    warmup_triton=False,
    disable_exllama=False,  # Try to use exllama if available
    disable_exllamav2=False,  # Try to use exllamav2 if available  
    inject_fused_attention=False,
    inject_fused_mlp=False,
    use_cuda_fp16=True,
    quantize_config=None,
    local_files_only=True,
    low_cpu_mem_usage=True,  # Important for your RAM
)

print("✅ Model loaded successfully!")

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

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True)

dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings))

print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ================
# FastAPI App
# ================
app = FastAPI(title="Optimized GPTQ + RAG Chat API")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role

@app.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    user_input = data.get("userInput")
    role = data.get("role", active_role)

    if not user_input:
        return StreamingResponse(
            iter(["data: Error: Missing user input\n\n"]), 
            media_type="text/event-stream"
        )

    # ===== RAG retrieval =====
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening"]
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""
    else:
        query_vec = embedder.encode([user_input])
        D, I = index.search(np.array(query_vec), k=3)
        retrieved_docs = []
        for i in I[0]:
            if isinstance(docs[i], dict) and "text" in docs[i]:
                retrieved_docs.append(docs[i]["text"])
            else:
                retrieved_docs.append(str(docs[i]))
        context = "\n".join(retrieved_docs)

    # ===== Build prompt =====
    full_prompt = roles.get(role, roles["Career_mentor"]).format(
        context=context, 
        user=user_input
    )
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Background thread for generation
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        streamer=streamer,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # ===== Stream tokens as Server-Sent Events =====
    def sse_generator():
        for token in streamer:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "AutoGPTQ",
        "model_path": MODEL_PATH,
        "knowledge_base_entries": len(docs)
    }

# ===============================
# Run server
# ===============================
if __name__ == "__main__":
    print("\n🎯 Starting optimized FastAPI server...")
    print("📡 API will be available at: http://127.0.0.1:8000")
    print("📖 Docs available at: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)