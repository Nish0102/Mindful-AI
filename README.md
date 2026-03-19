# 🌿 MindfulAI — Mental Health Support Chatbot

> An empathetic AI-powered mental wellness companion with mood tracking, crisis detection, and resource recommendations.

![Tech Stack](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![React](https://img.shields.io/badge/React-18.3-61DAFB?style=flat-square&logo=react)
![LLaMA 3](https://img.shields.io/badge/LLaMA_3-Groq-FF6B35?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🤖 **LLM-Powered Chat** | LLaMA 3 via Groq API — empathetic, context-aware responses |
| 📈 **Mood Tracking** | Sentiment analysis on every message, visualized as a live chart |
| 🆘 **Crisis Detection** | Keyword-based trigger surfaces Indian helpline numbers instantly |
| 📚 **Resource Recommendations** | Articles & exercises matched to conversation topics |
| 💬 **Session Memory** | Last 10 messages sent as context for coherent conversations |

---

## 🗂 Project Structure

```
mindful-ai/
├── backend/
│   ├── main.py              # FastAPI app, Groq LLM, sentiment analysis
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── App.jsx           # Main chat UI
    │   ├── hooks/useChat.js  # API state management
    │   └── components/
    │       ├── ChatMessage.jsx
    │       ├── Sidebar.jsx
    │       ├── MoodChart.jsx
    │       └── ResourceCard.jsx
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## 🚀 Setup & Running

### 1. Get a Free Groq API Key
- Go to [console.groq.com](https://console.groq.com)
- Sign up (free) → Create API Key
- Copy your key

### 2. Backend Setup
```bash
cd backend
pip install -r requirements.txt
python -m textblob.download_corpora   # Download NLP data
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
uvicorn main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit **http://localhost:5173** 🎉

---

## 🧠 How It Works

```
User message
    │
    ▼
Crisis Detection (keyword scan)
    │
    ├── Crisis detected → Helpline response + resources
    │
    └── Normal → Groq LLaMA 3 (with therapeutic system prompt)
                    │
                    ├── TextBlob sentiment analysis
                    ├── Topic-matched resource lookup
                    └── Response streamed to UI
```

---

## 🌐 Deployment

**Backend → Railway / Render:**
```bash
# Set GROQ_API_KEY as environment variable in dashboard
# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Frontend → Vercel:**
```bash
# Update API_BASE in useChat.js to your deployed backend URL
vercel --prod
```

---

## 📌 Crisis Resources (India)
- **iCall**: 9152987821
- **Vandrevala Foundation**: 1860-2662-345 (24/7)
- **NIMHANS**: 080-46110007

---

## 🔮 Future Improvements
- [ ] User authentication + persistent mood history
- [ ] Voice input / speech-to-text
- [ ] Multi-language support (Telugu, Hindi)
- [ ] Weekly mood report PDF export
- [ ] Integration with journaling feature
