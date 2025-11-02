# gptq_rag_server_fixed.py
"""
Notes:
- This script loads GPTQ models using auto-gptq.
- It includes robust fallbacks when CUDA kernels / exllama kernels are missing.
- If you want full GPU speed, follow the "How to get full GPU speed" printed at startup.

Recommended installs (if not done):
pip install auto-gptq safetensors transformers sentence-transformers faiss-cpu fastapi uvicorn
On Windows, if you later want bitsandbytes: pip install bitsandbytes-windows
"""

import os
import sys
import json
import threading
import numpy as np
import torch
import faiss
import uvicorn

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

# AutoGPTQ import
try:
    from auto_gptq import AutoGPTQForCausalLM
    AUTO_GPTQ_AVAILABLE = True
except Exception as e:
    AUTO_GPTQ_AVAILABLE = False
    AUTO_GPTQ_IMPORT_ERROR = e

from sentence_transformers import SentenceTransformer

# ===================
# User config (edit paths)
# ===================
MODEL_PATH = r"E:\text-generation-webui-main\text-generation-webui-main\user_data\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"
DATA_PATH = r"C:\Users\Lenovo\Documents\programing\miniProject\Backend\AI-microservice\software_career_knowledge.json"

# ===================
# Environment checks
# ===================
print("=== Environment check ===")
cuda_available = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {cuda_available}")

if not AUTO_GPTQ_AVAILABLE:
    print("ERROR: auto_gptq is not importable.")
    print("Import error:", AUTO_GPTQ_IMPORT_ERROR)
    print("Install auto-gptq and safetensors: pip install auto-gptq safetensors")
    sys.exit(1)

# Helpful messages about CUDA/kernel warnings
print("If you previously saw messages like 'CUDA extension not installed' or 'exllamav2 kernel is not installed',\n"
      "this script will attempt to load the model safely but you may see slower inference or warnings.\n")

# Decide device_map fallback strategy:
# - If CUDA available, try device_map='auto' (best) but also set safe flags to avoid calling missing kernels.
# - If CUDA unavailable, force device_map='cpu'.
device_map = "auto" if cuda_available else "cpu"

# ======================
# Load Tokenizer
# ======================
print("🔁 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True, local_files_only=True)

# ======================
# Load GPTQ model with safe options
# ======================
print("🔁 Loading GPTQ model (safe defaults)...")

# We pass a few conservative kwargs to avoid invoking missing kernels that cause runtime errors.
# If you want maximum speed and have a proper CUDA+nvcc+compiled exllama setup, you can re-run with:
#   disable_exllamav2=False, use_triton=True, inject_fused_attention=True
# But those options may not exist on all auto_gptq builds; the defaults below are conservative.

gptq_kwargs = dict(
    model_name_or_path=MODEL_PATH,     # required positional argument for some auto_gptq versions
    device_map=device_map,
    use_safetensors=True,
    trust_remote_code=True,
    # conservative flags:
    inject_fused_attention=False,      # avoid relying on fused CUDA kernels
    disable_exllamav2=True,            # avoid exllama kernels when not installed
    # You may optionally add use_triton=False here if your install complains about triton
)

# Some auto_gptq vintages accept the model name as first positional arg: try both styles safely.
model = None
load_errors = []
try:
    # First try: positional first argument (most typical)
    model = AutoGPTQForCausalLM.from_quantized(MODEL_PATH, **{k: v for k, v in gptq_kwargs.items() if k != "model_name_or_path"})
except TypeError as te:
    load_errors.append(("positional attempt", te))
    try:
        # Second try: keyword style
        model = AutoGPTQForCausalLM.from_quantized(**gptq_kwargs)
    except Exception as e:
        load_errors.append(("keyword attempt", e))
        # Fall back to raising with a helpful message
        print("Failed to load model via AutoGPTQForCausalLM.from_quantized with attempts:")
        for tag, err in load_errors:
            print(f" - {tag}: {repr(err)}")
        print("\nPlease check your auto-gptq version and its from_quantized signature.")
        print("Recommended: pip install --upgrade auto-gptq safetensors")
        raise

print("✅ Model object created (auto_gptq).")
# Ensure pad token id
if getattr(model.config, "pad_token_id", None) is None:
    model.config.pad_token_id = model.config.eos_token_id

# Inform user about possible performance limitations
print("\n=== Loader notes ===")
print("auto_gptq printed status above at load time. If you saw warnings about missing CUDA kernels or exllamav2,")
print("you are running in a degraded mode (model will still run but inference may be slow).")
print("To get full GPU speed, follow the steps printed at the end of this script.")

# ============================
# Custom Stopping Criteria
# ============================
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        super().__init__()
        self.stop_token_ids = set(stop_token_ids)

    def __call__(self, input_ids, scores, **kwargs):
        return int(input_ids[0, -1].item() in self.stop_token_ids)

stop_ids = [model.config.eos_token_id]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])

# ==========================
# Prompt roles
# ==========================
roles = {
    "default": "### Instruction:\n{context}\n{user}\n\n### Response:",
    "Career_mentor": (
        "### Instruction:\nYou are a friendly mentor who helps software engineering students. "
        "Answer their doubts and guide them. Do not assume anything until they ask, "
        "and do not mention fields unless asked. Ask a clarifying question if needed, "
        "but keep responses short and to the point.\n\n{context}\n{user}\n\n### Response:"
    ),
}
active_role = "Career_mentor"

# ==========================
# Load knowledge base (RAG)
# ==========================
print("📚 Loading knowledge base...")
docs = []
if DATA_PATH.endswith(".json"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
        docs = list(data.values()) if isinstance(data, dict) else data
elif DATA_PATH.endswith(".txt"):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
else:
    raise RuntimeError("DATA_PATH must point to a .json or .txt knowledge file.")

docs = [d if isinstance(d, str) else json.dumps(d, ensure_ascii=False) for d in docs]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs, show_progress_bar=True, convert_to_numpy=True)
dim = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(doc_embeddings, dtype=np.float32))
print(f"✅ Knowledge base loaded with {len(docs)} entries.")

# ================
# FastAPI App
# ================
app = FastAPI(title="GPTQ + RAG Chat API (safe loader)")

class ChatRequest(BaseModel):
    user_input: str
    role: str = active_role

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    user_input = request.user_input
    role = request.role if request.role in roles else active_role

    greetings = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
    if user_input.strip().lower() in greetings or len(user_input.strip().split()) <= 2:
        context = ""
    else:
        query_vec = embedder.encode([user_input], convert_to_numpy=True)
        D, I = index.search(np.array(query_vec, dtype=np.float32), k=3)
        retrieved_docs = [docs[idx] for idx in I[0] if 0 <= idx < len(docs)]
        context = "\n".join(retrieved_docs)

    full_prompt = roles[role].format(context=context, user=user_input)

    inputs = tokenizer(full_prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except StopIteration:
        # model may not have parameters (unlikely) — keep tensors on CPU
        pass

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.2,
        stopping_criteria=stopping_criteria,
        streamer=streamer,
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    def token_generator():
        try:
            for token in streamer:
                yield token
        except GeneratorExit:
            pass

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")

# ===============
# Helpful final instructions printed at runtime
# ===============
def print_fix_instructions():
    print("\n=== How to get full GPU speed (if you want it) ===")
    print("1) Ensure CUDA toolkit & nvcc are installed and visible in PATH.")
    print("   - Linux: install CUDA from NVIDIA and run 'nvcc --version' to verify.")
    print("   - Windows: install CUDA toolkit and make sure nvcc is on PATH.")
    print("2) Re-install auto-gptq from source so CUDA kernels and exllama kernels are compiled:")
    print("   git clone https://github.com/PanQiWei/AutoGPTQ.git")
    print("   cd AutoGPTQ")
    print("   pip install -r requirements.txt")
    print("   pip install .")
    print("3) (Optional) If you want bitsandbytes 4-bit fallback on Windows, use:")
    print("   pip install bitsandbytes-windows")
    print("4) Restart Python and re-run this script. You should see fewer warnings and much faster inference.")
    print("\nIf after following the steps you still see warnings, paste the full error output in the AutoGPTQ or bitsandbytes GitHub issues.\n")

print_fix_instructions()

if __name__ == "__main__":
    # On Windows, avoid tokenizers parallelism if issues:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    uvicorn.run(app, host="127.0.0.1", port=8000)
