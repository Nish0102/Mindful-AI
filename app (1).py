from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from groq import Groq
from textblob import TextBlob
from datetime import datetime

load_dotenv()

app = FastAPI(title="MindfulAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are MindfulAI, a warm, empathetic mental health support companion. 

Your role:
- Listen actively and validate feelings without judgment
- Use therapeutic techniques like CBT reframing, mindfulness prompts, and grounding exercises
- Ask gentle clarifying questions to understand the user better
- Suggest practical coping strategies tailored to what they share
- Use calming, caring language — never clinical or cold
- Keep responses concise (2-4 paragraphs max) unless the user needs detailed guidance

Important rules:
- You are NOT a replacement for professional therapy — gently mention this when appropriate
- If you detect crisis signals (suicidal ideation, self-harm), IMMEDIATELY respond with crisis resources and urge professional help
- Never diagnose conditions
- Always end with a question or gentle prompt to keep the conversation going

Crisis keywords to watch: "kill myself", "end it all", "don't want to live", "hurt myself", "suicide", "self harm"
"""

RESOURCES = {
    "anxiety": [
        {"title": "4-7-8 Breathing Exercise", "type": "exercise", "desc": "Inhale 4s, hold 7s, exhale 8s. Repeat 4 times.", "link": "https://www.healthline.com/health/4-7-8-breathing"},
        {"title": "Understanding Anxiety", "type": "article", "desc": "How anxiety works and evidence-based coping strategies.", "link": "https://www.anxietycanada.com/articles/what-is-anxiety/"},
    ],
    "depression": [
        {"title": "Behavioral Activation Guide", "type": "exercise", "desc": "Schedule small enjoyable activities to lift mood.", "link": "https://www.psychologytools.com/resource/behavioral-activation/"},
        {"title": "Understanding Depression", "type": "article", "desc": "Signs, causes, and paths to recovery.", "link": "https://www.nimh.nih.gov/health/topics/depression"},
    ],
    "stress": [
        {"title": "Progressive Muscle Relaxation", "type": "exercise", "desc": "Tense and release each muscle group to release tension.", "link": "https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/relaxation-technique/art-20045368"},
        {"title": "5-4-3-2-1 Grounding", "type": "exercise", "desc": "Name 5 things you see, 4 you hear, 3 you can touch, 2 you smell, 1 you taste.", "link": "https://www.verywellmind.com/grounding-techniques-for-anxiety-4692574"},
    ],
    "sleep": [
        {"title": "Sleep Hygiene Tips", "type": "article", "desc": "Evidence-based habits for better sleep quality.", "link": "https://www.sleepfoundation.org/sleep-hygiene"},
        {"title": "Body Scan Meditation", "type": "exercise", "desc": "Guided relaxation to help you fall asleep.", "link": "https://www.mindful.org/body-scan-meditation/"},
    ],
    "general": [
        {"title": "Mindfulness Meditation", "type": "exercise", "desc": "A 5-minute breathing meditation to center yourself.", "link": "https://www.headspace.com/meditation/breathing-exercises"},
        {"title": "Journaling for Mental Health", "type": "article", "desc": "How writing helps process emotions and reduce stress.", "link": "https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentID=4552&ContentTypeID=1"},
    ],
}

CRISIS_KEYWORDS = ["kill myself", "end it all", "don't want to live", "hurt myself", "suicide", "self harm", "selfharm", "want to die", "no reason to live"]

CRISIS_RESPONSE = """I hear you, and I'm deeply concerned about your safety right now. 

Please reach out immediately to:
🆘 **iCall (India)**: 9152987821
🆘 **Vandrevala Foundation**: 1860-2662-345 (24/7)
🆘 **NIMHANS**: 080-46110007

You don't have to face this alone. A trained counselor can help right now. Would you be willing to call one of these numbers?"""

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    session_id: Optional[str] = None

class SentimentRequest(BaseModel):
    text: str

def detect_crisis(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CRISIS_KEYWORDS)

def analyze_sentiment(text: str) -> dict:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # -1 to 1
    subjectivity = blob.sentiment.subjectivity  # 0 to 1

    if polarity > 0.3:
        mood = "positive"
        emoji = "😊"
    elif polarity > 0.0:
        mood = "slightly positive"
        emoji = "🙂"
    elif polarity == 0.0:
        mood = "neutral"
        emoji = "😐"
    elif polarity > -0.3:
        mood = "slightly negative"
        emoji = "😔"
    else:
        mood = "negative"
        emoji = "😢"

    return {
        "polarity": round(polarity, 3),
        "subjectivity": round(subjectivity, 3),
        "mood": mood,
        "emoji": emoji,
        "score": round((polarity + 1) / 2 * 100)  # 0-100 scale
    }

def get_resources(messages: List[Message]) -> List[dict]:
    all_text = " ".join([m.content.lower() for m in messages])
    matched = []
    for topic, resources in RESOURCES.items():
        if topic != "general" and topic in all_text:
            matched.extend(resources)
    if not matched:
        matched = RESOURCES["general"]
    return matched[:3]

@app.post("/chat")
async def chat(request: ChatRequest):
    last_message = request.messages[-1].content if request.messages else ""

    # Crisis detection
    is_crisis = detect_crisis(last_message)
    if is_crisis:
        sentiment = analyze_sentiment(last_message)
        resources = get_resources(request.messages)
        return {
            "response": CRISIS_RESPONSE,
            "sentiment": sentiment,
            "resources": resources,
            "is_crisis": True
        }

    # Build messages for Groq
    groq_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages[-10:]:  # last 10 messages for context
        groq_messages.append({"role": msg.role, "content": msg.content})

    try:
        completion = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=groq_messages,
            temperature=0.75,
            max_tokens=600,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    sentiment = analyze_sentiment(last_message)
    resources = get_resources(request.messages)

    return {
        "response": response_text,
        "sentiment": sentiment,
        "resources": resources,
        "is_crisis": False
    }

@app.post("/analyze-sentiment")
async def analyze(req: SentimentRequest):
    return analyze_sentiment(req.text)

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}
