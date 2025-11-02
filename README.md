# AI Career Mentor System for Software Students

## 🎯 Overview
The **AI Career Mentor System** is a comprehensive personal mentorship platform designed for software engineering students. It combines structured learning paths, career guidance, and an AI-powered chatbot to provide personalized mentorship accessible to everyone.

### Key Features
✅ **Structured Learning Roadmaps** - Pre-built paths for Software Developer, Data Scientist, and AI/ML Engineer careers  
✅ **DSA Progress Tracking** - Track completion of Data Structures & Algorithms modules  
✅ **AI Mentor Chatbot** - Local LLM-powered assistant for answering questions and providing guidance  
✅ **Chat History** - Persistent conversation history per user account  
✅ **Progress Dashboard** - Visual tracking of learning achievements and next steps  
✅ **User Authentication** - Secure JWT-based authentication system  
✅ **Offline Capable** - AI runs locally without requiring constant internet connection

### Why This System?
- **Structured Guidance**: Clear roadmaps prevent overwhelm from scattered internet advice
- **24/7 Availability**: AI mentor available anytime without scheduling
- **Privacy-First**: Local AI ensures your questions remain private
- **Personalized**: Tracks individual progress and suggests next steps
- **Accessible**: Perfect for shy, introverted students or those with limited internet

---

## 🛠️ Technologies Used

### Frontend
- **HTML5, CSS3, JavaScript (ES6+)**
- **Responsive Design** - Works on desktop and mobile
- **Future**: Migration to React for enhanced UX

### Backend (Spring Boot)
- **Java 17+**
- **Spring Boot 3.x**
- **Dependencies**:
  - Spring Web (REST APIs)
  - Spring Data JPA (Database ORM)
  - Spring Security (JWT Authentication)
  - PostgreSQL Driver
  - Spring Boot DevTools
  - Validation API
  - Springdoc OpenAPI (API Documentation)

### AI Microservice
- **Python 3.9+**
- **FastAPI** (High-performance web framework)
- **Libraries**:
  - `fastapi` - Web framework
  - `uvicorn` - ASGI server
  - `pydantic` - Data validation
  - `transformers` - Hugging Face models
  - `torch` - PyTorch for model inference
  - `sentence-transformers` - Embeddings for RAG
  - `faiss-cpu` - Vector similarity search
  - `numpy` - Numerical operations

### Database
- **PostgreSQL 14+**
- Tables: `users`, `chat_history`, `user_progress`, `roadmap_progress`

---

## 📦 Prerequisites

### System Requirements
- **RAM**: Minimum 8GB (16GB recommended for AI model)
- **Storage**: 10GB+ free space (for model files)
- **OS**: Windows, macOS, or Linux

### Required Software
1. **Java JDK 17 or higher**
2. **Python 3.9 or higher**
3. **PostgreSQL 14 or higher**
4. **Git** (for cloning repository)
5. **Maven** (usually comes with IDE)

---

## 🚀 Installation & Setup

### Step 1: Install Java
1. Download from [Oracle](https://www.oracle.com/java/technologies/downloads/) or [Adoptium](https://adoptium.net/)
2. Verify installation:
```bash
java -version
# Should show: java version "17.x.x" or higher
```

### Step 2: Install Python
1. Download from [python.org](https://www.python.org/downloads/)
2. **Important**: Check "Add Python to PATH" during installation
3. Verify installation:
```bash
python --version
# Should show: Python 3.9.x or higher
```

### Step 3: Install PostgreSQL
1. Download from [postgresql.org](https://www.postgresql.org/download/)
2. During installation, set a password for the `postgres` user
3. Create the database:
```bash
# Login to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE ai_mentor_db;
CREATE USER aiuser WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE ai_mentor_db TO aiuser;
\q
```

### Step 4: Clone the Repository
```bash
git clone https://github.com/yourusername/ai-career-mentor.git
cd ai-career-mentor
```

### Step 5: Install Python Dependencies
```bash
cd AI-microservice
pip install fastapi uvicorn pydantic transformers torch sentence-transformers faiss-cpu numpy
```

**For GPU Support** (if you have NVIDIA GPU):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Step 6: Download AI Model

#### Option 1: Using Text-Generation-WebUI (Recommended for Windows)
1. Download [text-generation-webui](https://github.com/oobabooga/text-generation-webui)
2. Use the built-in model downloader to get a GPTQ model:
   - **Recommended**: `TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ` (balanced performance)
   - **Alternative**: `TheBloke/StellarX-4B-V0.2-GPTQ` (lighter, faster)
3. Note the model path (e.g., `E:\text-generation-webui\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ`)

#### Option 2: Direct Download from Hugging Face
```bash
# Install git-lfs first
git lfs install

# Clone model (choose one)
cd AI-microservice/models
git clone https://huggingface.co/TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ
```

### Step 7: Configure AI Microservice
Edit `AI-microservice/main.py` (or your AI service file):
```python
# Update MODEL_PATH to your actual model location
MODEL_PATH = r"E:\text-generation-webui\models\TheBloke_CapybaraHermes-2.5-Mistral-7B-GPTQ"

# Update DATA_PATH if needed
DATA_PATH = r"C:\path\to\software_career_knowledge.json"
```

### Step 8: Configure Spring Boot Backend
Edit `Backend/ai-mentor-backend/src/main/resources/application.properties`:
```properties
# Database Configuration
spring.datasource.url=jdbc:postgresql://localhost:5432/ai_mentor_db
spring.datasource.username=aiuser
spring.datasource.password=your_password

# JWT Configuration
app.jwt.secret=your-very-long-random-secret-key-here-min-256-bits
app.jwt.expiration=86400000

# Server Port
server.port=8081
```

---

## ▶️ Running the Application

### Terminal 1: Start AI Microservice
```bash
cd AI-microservice
python main.py
```
Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
📚 Loading knowledge base...
✅ Knowledge base loaded with XXX entries.
```

### Terminal 2: Start Spring Boot Backend
```bash
cd Backend/ai-mentor-backend
./mvnw spring-boot:run
# Windows: mvnw.cmd spring-boot:run
```
Expected output:
```
Started AiMentorBackendApplication in X.XXX seconds
```

### Terminal 3: Start Frontend
```bash
cd Frontend
# Using Python's built-in server
python -m http.server 5500

# Or use Live Server extension in VS Code
```

### Access the Application
Open your browser and go to:
```
http://localhost:5500
# Or http://127.0.0.1:5500
```

---

## 📖 Usage Guide

### 1. Registration & Login
- Create an account on the registration page
- Login with your credentials
- JWT token is stored automatically

### 2. Dashboard
- View your learning progress
- See completed DSA parts and roadmap phases
- Check "Latest Achievements" for recent completions
- Review "Your Next Steps" for upcoming tasks

### 3. AI Mentor Chat
- Ask questions about programming, career advice, or concepts
- Chat history is automatically saved per user
- Conversations persist across sessions
- Use natural language - the AI understands context

### 4. DSA Learning Path
- Complete 4 parts: Basics & Arrays, Linked Lists, Trees, Advanced Topics
- Mark sections as complete to track progress
- Progress bar updates automatically on dashboard

### 5. Career Roadmaps
- Choose from 3 career paths:
  - **Software Developer**: Full-stack web development
  - **Data Scientist**: ML, statistics, and data analysis
  - **AI/ML Engineer**: Deep learning and AI systems
- Each roadmap has 5 phases
- Track status: Not Started → In Progress → Completed

---

## 🗂️ Project Structure

```
ai-career-mentor/
├── Frontend/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── ai-mentor.html         # Chat interface
│   ├── roadmap.html           # Career roadmaps
│   ├── dsa.html               # DSA overview
│   ├── dsa1.html - dsa4.html  # DSA parts
│   ├── phase1-software.html   # Roadmap phases (x15 files)
│   └── resources.html
│
├── Backend/ai-mentor-backend/
│   └── src/main/java/com/ai/mentor/backend/
│       ├── config/
│       │   └── SecurityConfig.java
│       ├── controller/
│       │   ├── AuthController.java
│       │   ├── ChatController.java
│       │   ├── ProgressController.java
│       │   └── RoadmapController.java
│       ├── filter/
│       │   └── JwtAuthFilter.java
│       ├── model/
│       │   ├── User.java
│       │   ├── ChatHistory.java
│       │   ├── UserProgress.java
│       │   └── RoadmapProgress.java
│       ├── repository/
│       │   ├── UserRepository.java
│       │   ├── ChatHistoryRepository.java
│       │   ├── UserProgressRepository.java
│       │   └── RoadmapProgressRepository.java
│       └── service/
│           ├── ChatService.java
│           ├── JwtService.java
│           └── CustomUserDetailsService.java
│
└── AI-microservice/
    ├── main.py                    # FastAPI app
    ├── software_career_knowledge.json
    └── models/                    # Downloaded AI models
```

---

## 🔧 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - Login and get JWT token

### Chat
- `POST /api/chat/stream` - Stream AI responses (SSE)
- `GET /api/chat/history` - Get chat history (requires auth)

### DSA Progress
- `GET /api/progress` - Get DSA completion status
- `POST /api/progress/complete/{partNumber}` - Mark DSA part complete

### Roadmap Progress
- `GET /api/roadmap/progress` - Get all roadmap progress
- `POST /api/roadmap/start/{roadmapId}/{phaseNumber}` - Mark phase as in-progress
- `POST /api/roadmap/complete/{roadmapId}/{phaseNumber}` - Mark phase complete

### AI Microservice
- `POST /chat/stream` - Get streaming AI response

---

## 🐛 Troubleshooting

### AI Microservice Issues
**Problem**: Model not loading
```bash
# Solution: Check MODEL_PATH is correct
# Verify model files exist in the directory
```

**Problem**: Out of memory
```bash
# Solution: Use a smaller model (4B instead of 7B)
# Or increase system RAM/swap space
```

### Backend Issues
**Problem**: Database connection refused
```bash
# Solution: Ensure PostgreSQL is running
sudo systemctl start postgresql  # Linux
# Windows: Start PostgreSQL service from Services
```

**Problem**: Port 8081 already in use
```bash
# Solution: Change port in application.properties
server.port=8082
```

### Frontend Issues
**Problem**: 403 Forbidden errors
```bash
# Solution: Check JWT token is being sent
# Open DevTools → Application → Local Storage
# Verify 'token' exists
```

**Problem**: CORS errors
```bash
# Solution: Ensure backend CORS allows your frontend origin
# Check @CrossOrigin annotation in controllers
```

---

## 🔒 Security Notes

- **Never commit** `application.properties` with real passwords
- **Change JWT secret** before deployment
- **Use environment variables** for sensitive data in production
- **Enable HTTPS** for production deployment
- **Validate all user inputs** (already implemented with Spring Validation)

---

## 🎨 Future Enhancements

### Short Term
- [ ] Add more career roadmaps (DevOps, Mobile Dev, etc.)
- [ ] Implement quiz system for knowledge testing
- [ ] Add resource recommendations based on progress
- [ ] Email notifications for milestones

### Long Term
- [ ] Migrate frontend to React/Next.js
- [ ] Implement real-time collaboration features
- [ ] Add video tutorial integration
- [ ] Create mobile app (React Native)
- [ ] Fine-tune AI model on software engineering Q&A
- [ ] Add mentor matching with senior developers
- [ ] Implement gamification (badges, leaderboards)

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Authors

- **Harshith B** - backend and AI integration - [GitHub Profile](https://github.com/Harshith55072)
- **aayush** - Frontend - [GitHub Profile]()

---

## 🙏 Acknowledgments

- Hugging Face for transformer models and hosting
- TheBloke for GPTQ quantized models
- Spring Boot and FastAPI communities
- All contributors and testers

---

## 📊 Project Status

**Current Version**: 1.0.0  
**Status**: Active Development  
**Last Updated**: November 2025

---

**Made with ❤️ for software engineering students worldwide**