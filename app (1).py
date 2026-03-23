from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from groq import Groq
from textblob import TextBlob
from datetime import datetime
import logging

load_dotenv()

app = FastAPI(title="MindfulAI Backend")

# -------------------- CORS --------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Logging --------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------- Groq Client --------------------
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# -------------------- Memory (Session-based) --------------------
sessions = {}

def get_session_history(session_id: str):
    return sessions.get(session_id, [])

def update_session(session_id: str, messages):
    sessions[session_id] = messages[-20:]

# -------------------- Prompt --------------------
SYSTEM_PROMPT = """You are MindfulAI, a warm, empathetic mental health support companion.

- Validate emotions first
- Use gentle, supportive tone
- Suggest 1–2 practical coping strategies
- Ask a soft follow-up question
- Keep responses concise (2–4 paragraphs)

Important:
- Not a replacement for therapy
- If crisis detected → provide emergency resources immediately
- Never diagnose conditions
"""

# -------------------- Data --------------------
RESOURCES = {
    "anxiety": [
        {"title": "4-7-8 Breathing", "link": "https://www.healthline.com/health/4-7-8-breathing"},
    ],
    "sadness": [
        {"title": "Behavioral Activation", "link": "https://www.psychologytools.com/resource/behavioral-activation/"},
    ],
    "stress": [
        {"title": "5-4-3-2-1 Grounding", "link": "https://www.verywellmind.com/grounding-techniques-for-anxiety-4692574"},
    ],
    "general": [
        {"title": "Mindfulness Meditation", "link": "https://www.headspace.com/meditation"},
    ],
}

CRISIS_KEYWORDS = [
    "kill myself", "end it all", "don't want to live",
    "hurt myself", "suicide", "self harm", "want to die"
]

CRISIS_RESPONSE = """I’m really sorry you're feeling this way. Your safety matters.

Please reach out immediately:
🆘 iCall: 9152987821  
🆘 Vandrevala Foundation: 1860-2662-345  
🆘 NIMHANS: 080-46110007  

You don’t have to go through this alone. Would you be willing to call one of them?"""

# -------------------- Models --------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None

class SentimentRequest(BaseModel):
    text: str

# -------------------- NLP --------------------
def analyze_sentiment(text: str):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.3:
        mood = "positive"
    elif polarity > 0:
        mood = "slightly positive"
    elif polarity == 0:
        mood = "neutral"
    elif polarity > -0.3:
        mood = "slightly negative"
    else:
        mood = "negative"

    return {
        "polarity": round(polarity, 3),
        "mood": mood,
        "score": round((polarity + 1) / 2 * 100)
    }

def detect_emotion(text: str):
    text = text.lower()
    if "anxious" in text or "nervous" in text:
        return "anxiety"
    elif "sad" in text or "down" in text:
        return "sadness"
    elif "stress" in text or "overwhelmed" in text:
        return "stress"
    return "general"

def detect_crisis(text: str):
    text_lower = text.lower()
    sentiment = analyze_sentiment(text)

    keyword_flag = any(kw in text_lower for kw in CRISIS_KEYWORDS)
    return keyword_flag or sentiment["polarity"] < -0.6

def get_resources(emotion: str):
    return RESOURCES.get(emotion, RESOURCES["general"])

# -------------------- Routes --------------------
@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    last_message = request.messages[-1].content

    # Session memory
    if request.session_id:
        history = get_session_history(request.session_id)
        messages = history + request.messages
        update_session(request.session_id, messages)
    else:
        messages = request.messages

    # Crisis detection
    is_crisis = detect_crisis(last_message)
    sentiment = analyze_sentiment(last_message)
    emotion = detect_emotion(last_message)
    resources = get_resources(emotion)

    if is_crisis:
        return {
            "response": CRISIS_RESPONSE,
            "sentiment": sentiment,
            "emotion": emotion,
            "resources": resources,
            "is_crisis": True
        }

    # Build LLM input
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages[-10:]:
        groq_messages.append({"role": msg.role, "content": msg.content})

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=groq_messages,
            temperature=0.7,
            max_tokens=500,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Logging
    logging.info({
        "time": datetime.now().isoformat(),
        "sentiment": sentiment,
        "emotion": emotion,
        "crisis": is_crisis
    })

    return {
        "response": response_text,
        "sentiment": sentiment,
        "emotion": emotion,
        "resources": resources,
        "is_crisis": False
    }

@app.post("/analyze-sentiment")
async def analyze(req: SentimentRequest):
    return analyze_sentiment(req.text)

@app.get("/daily-checkin")
async def daily_checkin():
    return {
        "prompt": "How are you feeling today? One word is enough 🌱"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
