"""
LanguageBuddy: A Streamlit app for Norwegian reading & speaking practice

Features
- Choose user (Anh / Giang / Hannah)
- Choose topic & CEFR level (A1–C2)
- Generate a ~5-minute Norwegian reading text
- Record speech in-browser (WebRTC) or upload audio
- Transcribe audio (OpenAI Whisper API)
- Compare transcript vs original (WER/CER, diff), score out of 100
- Feedback: strengths & weaknesses
- Save results to MongoDB (DB: languagebuddy, Collection: readingskill)

Setup
1) Python 3.10+
2) `pip install -r requirements.txt` with:
   streamlit==1.38.0
   streamlit-webrtc==0.47.1
   openai>=1.35.0
   jiwer==3.0.3
   pymongo==4.8.0
   python-dotenv==1.0.1

3) Environment variables (e.g., in .env):
   OPENAI_API_KEY=sk-...
   MONGODB_URI=mongodb+srv://user:pass@cluster/db

Run
   streamlit run languagebuddy_streamlit_app.py
"""
from __future__ import annotations
import os
import io
import json
import time
import uuid
import datetime as dt
from dataclasses import dataclass

import numpy as np
import asyncio

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from jiwer import wer, cer
from difflib import ndiff
from pymongo import MongoClient
from dotenv import load_dotenv

# OpenAI (>= v1.0 SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# ---------------------- Config & Helpers ----------------------
load_dotenv()
OPENAI_API_KEY = st.secrets["api"]["key"]
MONGODB_URI = st.secrets["mongo"]["uri"]

st.set_page_config(page_title="LanguageBuddy — Norwegian Practice", page_icon="🇳🇴", layout="wide")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Best-effort: ignore noisy asyncio datagram transport errors from ICE teardown
def _ignore_asyncio_noise(loop, context):
    msg = context.get("message", "")
    exc = context.get("exception")
    text = f"{msg} {exc}" if exc else msg
    noisy = (
        "Fatal write error on datagram transport" in text
        or "Transaction.__retry" in text
        or "sendto" in text
    )
    if noisy:
        return  # swallow known ICE teardown noise
    # Fallback to default handler
    try:
        loop.default_exception_handler(context)
    except Exception:
        pass

try:
    asyncio.get_running_loop().set_exception_handler(_ignore_asyncio_noise)
except RuntimeError:
    # No running loop yet; Streamlit will create one later.
    pass

USERS = ["anh", "giang", "hannah"]
CEFR_LEVELS = ["A1", "A2", "B1", "B2", "C1", "C2"]
DEFAULT_TOPICS = [
    "Utdanning", "Arbeid og karriere", "Helse og livsstil", "Familie og venner",
    "Norsk kultur og tradisjoner", "Reise og natur", "Teknologi i hverdagen",
]

# Estimated words per minute per CEFR (rough heuristic)
WPM = {
    "A1": 95,
    "A2": 110,
    "B1": 130,
    "B2": 150,
    "C1": 165,
    "C2": 180,
}

@dataclass
class SessionState:
    original_text: str = ""
    transcript_text: str = ""
    analysis: dict | None = None
    audio_bytes: bytes | None = None
    record_id: str = ""


if "state" not in st.session_state:
    st.session_state.state = SessionState()

state: SessionState = st.session_state.state

# ---------------------- OpenAI Client ----------------------
@st.cache_resource(show_spinner=False)
def get_openai_client():
    if not OPENAI_API_KEY or OpenAI is None:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None

client = get_openai_client()

# ---------------------- Mongo Client ----------------------
@st.cache_resource(show_spinner=False)
def get_mongo():
    if not MONGODB_URI:
        return None
    try:
        mongo = MongoClient(MONGODB_URI)
        db = mongo["languagebuddy"]
        col = db["readingskill"]
        return col
    except Exception as e:
        st.sidebar.warning(f"MongoDB-tilkobling feilet: {e}")
        return None

collection = get_mongo()

# ---------------------- Text Generation ----------------------
SYSTEM_PROMPT = (
    "Du er en norsk språklærer. Skriv en faktatekst på norsk (bokmål) som er klar,\n"
    "strukturert og naturlig å lese høyt. Match CEFR-nivået strengt i ordvalg og\n"
    "syntaks. Unngå punktlister, skriv sammenhengende avsnitt med god flyt."
)

GEN_TEMPLATE = (
    "Emne: {topic}. Målgruppe: bruker = {user}, nivå = {level}.\n"
    "Skriv en sammenhengende tekst for høytlesning i ca. 5 minutter.\n"
    "Sikt mot omtrent {target_words} ord (±10%).\n\n"
    "Krav:\n"
    "- Bruk klart bokmål, naturlig prosodi.\n"
    "- Hold deg konsekvent til nivå {level} (vokabular og setningslengde).\n"
    "- Gi en tydelig innledning, hoveddel (2–4 avsnitt) og kort avslutning.\n"
    "- Unngå for mange tall og forkortelser.\n"
    "- Ingen dialog, ingen listepunkter.\n"
)


def estimate_target_words(level: str, minutes: int = 5) -> int:
    wpm = WPM.get(level, 140)
    return int(wpm * minutes)


def generate_text(topic: str, user: str, level: str) -> str:
    target_words = estimate_target_words(level)
    if not client:
        # Fallback deterministic template if OpenAI not set
        return (
            f"Tema: {topic}. Dette er en plassholdertekst fordi OPENAI_API_KEY ikke er satt. "
            f"For en ekte tekst, sett nøkkelen. Nivå {level}, mål ~{target_words} ord.\n\n"
            "Utdrag: Norge har et sterkt fokus på {topic.lower()}, og mange opplever at læring gjennom livet "
            "gir mening og retning. (…)")

    prompt = GEN_TEMPLATE.format(topic=topic, user=user, level=level, target_words=target_words)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        st.error(f"Tekstgenerering feilet: {e}")
        return ""

# ---------------------- Audio Utils ----------------------

# Real-time audio level callback for streamlit-webrtc

def audio_level_callback(frame):
    try:
        pcm = frame.to_ndarray()
        if pcm.ndim == 2:
            pcm = pcm.mean(axis=1)
        if pcm.dtype == np.int16:
            data = pcm.astype(np.float32) / 32768.0
        else:
            data = pcm.astype(np.float32)
        rms = float(np.sqrt(np.mean(np.square(data))))
        prev = st.session_state.get("_rms", 0.0)
        smoothed = 0.8 * prev + 0.2 * rms
        st.session_state["_rms"] = max(0.0, min(1.0, smoothed))
        st.session_state["_speaking"] = st.session_state["_rms"] > 0.08
    except Exception:
        pass
    return frame


def transcribe_audio(audio_bytes: bytes, filename: str = "speech.webm") -> str:
    if not client:
        st.warning("Ingen OPENAI_API_KEY. Kan ikke transkribere.")
        return ""
    try:
        # Write in-memory to a file-like object for upload
        b = io.BytesIO(audio_bytes)
        b.name = filename
        b.seek(0)
        tr = client.audio.transcriptions.create(
            model="whisper-1",
            file=b,
            language="no",
            response_format="verbose_json",
            temperature=0.0,
        )
        return tr.text.strip()
    except Exception as e:
        st.error(f"Transkripsjon feilet: {e}")
        return ""

# ---------------------- Analysis ----------------------

def normalize_text(s: str) -> str:
    import re
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def diff_markup(a: str, b: str) -> str:
    """Return simple HTML marking differences (insertions/deletions)."""
    out = []
    for token in ndiff(a.split(), b.split()):
        if token.startswith("- "):
            out.append(f"<span style='background:#ffe0e0;text-decoration:line-through'>{token[2:]}</span>")
        elif token.startswith("+ "):
            out.append(f"<span style='background:#e0ffe0'>{token[2:]}</span>")
        elif token.startswith("? "):
            # detail markers skipped
            pass
        else:
            out.append(token[2:])
    return " ".join(out)


def analyze(original: str, transcript: str) -> dict:
    o = normalize_text(original)
    t = normalize_text(transcript)
    metrics = {
        "wer": float(wer(o, t)) if o and t else 1.0,
        "cer": float(cer(o, t)) if o and t else 1.0,
        "orig_words": len(o.split()),
        "trans_words": len(t.split()),
    }

    # Score (simple): 100 * (1 - WER), clipped to [0,100]
    score = max(0.0, 100.0 * (1.0 - metrics["wer"]))

    # Heuristic feedback
    strengths = []
    weaknesses = []

    # Length adherence
    length_ratio = metrics["trans_words"] / (metrics["orig_words"] + 1e-9)
    if 0.9 <= length_ratio <= 1.1:
        strengths.append("Godt tempo og komplett lesing (lengde ~ original).")
    elif length_ratio < 0.85:
        weaknesses.append("Du hoppet over flere ord/setninger. Forsøk å fullføre hele teksten.")
    else:
        weaknesses.append("Du la til mange ekstra ord. Hold deg nær teksten.")

    # Error-based feedback
    if metrics["wer"] <= 0.15:
        strengths.append("Svært presis uttale og ordgjenkjenning (lav WER).")
    elif metrics["wer"] <= 0.30:
        strengths.append("Greit nivå på nøyaktighet. Litt variasjon i uttale/ordvalg.")
    else:
        weaknesses.append("Høy WER: jobb med tydelig uttale, rytme og vanskelige ord.")

    if metrics["cer"] <= 0.10:
        strengths.append("Stabil artikulasjon på ordnivå (lav CER).")
    elif metrics["cer"] > 0.25:
        weaknesses.append("Mange tegnfeil tyder på uklare endelser/konsonanter.")

    # Diff
    html_diff = diff_markup(o, t)

    return {
        "metrics": metrics,
        "score": round(score, 2),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "diff_html": html_diff,
    }

# ---------------------- Persistence ----------------------

def save_to_mongo(doc: dict) -> str | None:
    if not collection:
        st.warning("Ingen MongoDB-tilkobling. Kan ikke lagre.")
        return None
    try:
        res = collection.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        st.error(f"Lagring feilet: {e}")
        return None

# ---------------------- UI ----------------------

st.title("🇳🇴 LanguageBuddy — Les & Snakk (NO)")
st.caption("Generer tekst, les høyt, få transkripsjon og analyse — lagre fremdrift i MongoDB.")

with st.sidebar:
    st.header("Oppsett")
    user = st.selectbox("Bruker", USERS, index=0)
    level = st.selectbox("Nivå (CEFR)", CEFR_LEVELS, index=2)
    topic = st.selectbox("Tema", DEFAULT_TOPICS, index=0)
    topic_custom = st.text_input("Eller eget tema", placeholder="f.eks. Klima, skole, reise …")
    final_topic = topic_custom.strip() if topic_custom.strip() else topic

    st.divider()
    st.subheader("Tilkobling")
    use_host_only = st.toggle(
        "Bruk kun lokale ICE-kandidater (deaktiver STUN/TURN)",
        value=True,
        help="Kan hjelpe bak brannmurer eller i miljøer uten UDP. Skru av hvis du har TURN/STUN.",
    )
    if use_host_only:
        active_rtc_conf = RTCConfiguration({"iceServers": []})
        st.caption("Host-only: ingen STUN/TURN. Kan feile over internett, men stabilt lokalt.")
    else:
        active_rtc_conf = RTC_CONFIGURATION

    st.divider()
    st.subheader("Tekst")
    if st.button("✍️ Generer ~5 min tekst", type="primary"):
        with st.spinner("Genererer tekst …"):
            state.original_text = generate_text(final_topic, user, level)
            state.analysis = None
            state.transcript_text = ""
            state.record_id = str(uuid.uuid4())
