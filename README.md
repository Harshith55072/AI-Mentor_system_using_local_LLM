# AI Career Mentor System for Software Students

## Overview
The **AI Career Mentor System** is designed to act as a personal mentor for software engineering students.  
It provides **structured roadmaps**, **career guidance**, and an **AI-powered mentor chatbot** that works even offline.  

This tool helps students:
- Follow clear, pre-built learning paths.
- Ask doubts and questions to a local AI (LLM).
- Get career guidance tailored to their interests.
- Overcome the overwhelm of scattered internet advice.
- Access mentorship even if they are shy, introverted, or have limited internet.

---

## Technologies Used

### Frontend
- **HTML, CSS, JavaScript**
- (Future: React for improved UI)

### Backend
- **Spring Boot (Java)**  
  Dependencies used (already installed in backend, no action needed):  
  - Spring Web (REST APIs)  
  - Spring Data JPA (Database access)  
  - PostgreSQL Driver (DB connection)  
  - Spring Security (authentication / JWT later)  
  - Spring Boot DevTools (hot reload, optional)  
  - Validation (for request DTOs)  
  - Springdoc OpenAPI (auto-generate API docs)  

### AI Microservice
- **Python (FastAPI + Transformers)**  
- Libraries:  
  - `fastapi`  
  - `uvicorn`  
  - `pydantic`  
  - `transformers`  
  - `torch`  
  - `sentence-transformers`  
  - `faiss`  
  - `numpy`  

---

## Installation & Setup

### 1. Install Java (for Backend)
- Download and install [Java JDK](https://www.oracle.com/java/technologies/downloads/).
- Verify installation:
```bash
java -version
```

### 2. Install Python (for AI Microservice)
- Download Python from [python.org](https://www.python.org/downloads/).
- Verify installation:
```bash
python --version
```

### 3. Install Required Python Libraries
Run these commands in your terminal/command prompt:
```bash
pip install fastapi
pip install uvicorn
pip install pydantic
pip install transformers
pip install torch
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
```
(Note: If you have CUDA/GPU, you may want to install a GPU-enabled version of torch.)

### 4. Install a GPTQ Model from Hugging Face
The AI mentor requires a **quantized GPTQ model** for efficient local inference.  
Follow these steps:

1. Visit the [Hugging Face GPTQ models page](https://huggingface.co/TheBloke) (TheBloke maintains many pre-quantized models).  
2. Choose a model appropriate for your system (e.g., `TheBloke/Llama-2-7B-GPTQ`).  
3. Download the model into your backend `ai-service/models/` directory. You can use `git lfs` for this:
   ```bash
   git lfs install
   git clone https://huggingface.co/TheBloke/Llama-2-7B-GPTQ
   ```
   This will create a folder with model weights inside `models/`.

4. Update your Python script (e.g., `ai-service/main.py`) to point to the downloaded model:
   ```python
   model_name_or_path = "./models/Llama-2-7B-GPTQ"
   tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
   model = AutoModelForCausalLM.from_pretrained(
       model_name_or_path,
       device_map="auto",
       trust_remote_code=True
   )
   ```

---

## How It Works
1. *Frontend (Web UI)*: Provides students with roadmaps and an easy-to-use interface.
2. *Backend (Spring Boot)*: Handles APIs, authentication, and database logic.
3. *AI Microservice (Python)*: Runs the local language model for answering student queries.

---

## Future Plans
- Move frontend to *React* for a smoother experience.
- Improve AI accuracy with more fine-tuned datasets.
- Add advanced career analytics and personalized recommendations.
