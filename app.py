import re
import json
import time
import ast
import os
import requests
import torch
import numpy as np
from difflib import SequenceMatcher
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
        json.dump(data, f, indent=2)

dataset = load_data()

# =========================
# EMBEDDINGS
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = []

def build_embeddings():
    global embeddings
    embeddings = [embed_model.encode(item["input"]) for item in dataset]

def search_embedding(query):
    if not embeddings:
        return None

    q = embed_model.encode(query)

    scores = [
        np.dot(q, e) / (np.linalg.norm(q) * np.linalg.norm(e))
        for e in embeddings
    ]

    idx = int(np.argmax(scores))
    if scores[idx] > 0.75:
        return dataset[idx]["output"]

    return None

build_embeddings()

# =========================
# SEARCH ENGINES
# =========================
def wiki_search(q):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q.replace(' ','_')}"
        r = requests.get(url)
        if r.status_code == 200:
            return r.json().get("extract")
    except:
        pass

def google_search(q):
    if not SERPER_API_KEY:
        return None
    try:
        res = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },
            json={"q": q}
        ).json()

        if "answerBox" in res:
            return res["answerBox"].get("snippet")

        if "organic" in res:
            return res["organic"][0].get("snippet")
    except:
        pass

def duck_search(q):
    try:
        r = requests.get(f"https://api.duckduckgo.com/?q={q}&format=json").json()
        return r.get("AbstractText")
    except:
        pass

def backup_search(q):
    try:
        html = requests.get(
            f"https://duckduckgo.com/html/?q={q}",
            headers={"User-Agent": "Mozilla/5.0"}
        ).text

        m = re.findall(r'class="result__a".*?>(.*?)</a>', html)
        if m:
            clean = re.sub("<.*?>", "", m[0])
            if len(clean) > 20:
                return clean
    except:
        pass

def smart_search(q):
    for fn in [wiki_search, google_search, duck_search, backup_search]:
        res = fn(q)
        if res:
            return res
    return None

# =========================
# MATH
# =========================
def solve_math(text):
    try:
        text = text.replace("x", "*").replace(" ", "")
        return str(eval(text))
    except:
        return None

# =========================
# MODEL (FIXED)
# =========================
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32   # SAFE for CPU
)

tokenizer.pad_token = tokenizer.eos_token

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False
        )

    return tokenizer.decode(out[0], skip_special_tokens=True).split("Assistant:")[-1].strip()

# =========================
# CHAT
# =========================
def chat(user_input, history):
    text = user_input.strip()
    lower = text.lower()

    # greeting
    if lower in ["hi","hello","hey"]:
        yield "Hey! 😄 What can I help you with?"
        return

    # math
    m = solve_math(lower)
    if m:
        yield f"{m} (Math ✅)"
        return

    # embeddings memory
    mem = search_embedding(lower)
    if mem:
        yield mem
        return

    # web search
    res = smart_search(lower)
    if res:
        if len(res) < 300:
            dataset.append({"input": lower, "output": res})
            if len(dataset) > 1000:
                dataset.pop(0)
            save_data(dataset)
            build_embeddings()

        yield res
        return

    # coding mode
    if any(k in lower for k in ["code","python","html","js"]):
        yield generate(f"Write clean code:\n{text}")
        return

    # fallback
    yield generate(f"User: {text}\nAssistant:")

# =========================
# UI
# =========================
demo = gr.ChatInterface(
    fn=chat,
    title="⚡ Smart AI Ultra",
    description="Search + Memory + Math + Code"
)

demo.launch(server_name="0.0.0.0", server_port=7860)
