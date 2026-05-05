import re
import json
import ast
import os
import requests
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import gradio as gr

torch.set_num_threads(2)

DATA_FILE = "dataset.json"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# =========================
# LOAD / SAVE
# =========================
def load_data():
    try:
        with open(DATA_FILE) as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(DATA_FILE, "w") as f:
        json.dump(data, f)

dataset = load_data()

# =========================
# EMBEDDINGS (LIGHT)
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = [embed_model.encode(d["input"]) for d in dataset]

def search_memory(q):
    if not embeddings:
        return None

    qv = embed_model.encode(q)
    scores = [
        np.dot(qv, e) / (np.linalg.norm(qv) * np.linalg.norm(e))
        for e in embeddings
    ]

    i = int(np.argmax(scores))
    if scores[i] > 0.75:
        return dataset[i]["output"]

# =========================
# SEARCH (FAST)
# =========================
def google_search(q):
    if not SERPER_API_KEY:
        return None

    try:
        r = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={"q": q},
            timeout=5
        ).json()

        if "answerBox" in r:
            return r["answerBox"].get("snippet")

        if "organic" in r:
            return r["organic"][0].get("snippet")

    except:
        pass

# =========================
# SAFE MATH
# =========================
def solve_math(t):
    try:
        t = t.replace("x", "*").replace(" ", "")
        return str(eval(compile(ast.parse(t, mode="eval"), "", "eval")))
    except:
        return None

# =========================
# MODEL (FAST LOAD)
# =========================
model_name = "distilgpt2"  # 🔥 MUCH FASTER for Render

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def generate(p):
    inp = tokenizer(p, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)

# =========================
# CHAT
# =========================
def chat(msg, history):
    text = msg.strip()
    low = text.lower()

    if low in ["hi","hello","hey"]:
        yield "Hey! 😄 What can I help you with?"
        return

    m = solve_math(low)
    if m:
        yield f"{m} (Math ✅)"
        return

    mem = search_memory(low)
    if mem:
        yield mem
        return

    web = google_search(low)
    if web:
        yield web
        return

    yield generate(f"User: {text}\nAssistant:")

# =========================
# UI (IMPORTANT)
# =========================
port = int(os.environ.get("PORT", 10000))

demo = gr.ChatInterface(
    fn=chat,
    title="⚡ Smart AI (Render)",
    description="Fast • Search • Memory • Math"
)

demo.launch(server_name="0.0.0.0", server_port=port)
