"""
app.py - the StudyMate web app.
Students type a doubt, and the bot finds the best matching passage in the
CBSE Class XII AI Student Handbook (Subject Code 843).

Run it with:  streamlit run app.py
"""

import base64
import html
import json
import os
import re

import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

# ---- Easy switches (change True / False, save, and refresh the page) ----
SHOW_TOOLBAR = False   # True brings back the Deploy button and the top-right menu (with Settings > Theme)
SIDEBAR_OPEN = True    # True keeps the left sidebar (unit names, search filter) open when the app starts

# These must match the numbers in train.py
MIN_SCORE = 0.10
MIN_COVERAGE = 0.6
TITLE_WEIGHT = 0.3   # how much the section title counts (same as train.py)

HINGLISH_STOP_WORDS = {
    "kya", "hai", "hota", "hote", "kaise", "kyun", "kyu", "batao", "bataiye", "ka", "ki", "ke",
    "me", "mein", "se", "ko", "aur", "hain", "kaun", "kaunsa", "samjhao", "matlab", "explain",
    "tell", "define", "please", "plz", "kare", "karo", "kar", "karta", "karte", "karna", "chahiye",
    "banate", "banata", "banaye", "banana", "hoti", "hoto", "bolo", "bata", "dikhao",
}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | HINGLISH_STOP_WORDS

UNITS = {
    1: "Python Programming-II",
    2: "Data Science Methodology",
    3: "Making Machines See",
    4: "AI with Orange Data Mining Tool",
    5: "Big Data and Data Analytics",
    6: "Understanding Neural Networks",
    7: "Generative AI",
    8: "Data Storytelling",
}

# One colour per unit, like the coloured tabs of a notebook. Bright enough to glow on dark glass.
UNIT_COLORS = {
    1: "#34C77B",
    2: "#A27BFF",
    3: "#2BC4DB",
    4: "#FF8A3D",
    5: "#F2C230",
    6: "#6C8CFF",
    7: "#FF6B9A",
    8: "#C79A78",
}

# Example doubts shown on the first screen: (question, unit it belongs to)
SUGGESTIONS = [
    ("What are the types of neural networks?", 6),
    ("What are the characteristics of big data?", 5),
    ("How do machines see images?", 3),
    ("What is K-fold cross validation?", 2),
    ("What is a large language model?", 7),
]

ACRONYMS = {"AI", "NN", "CNN", "RNN", "GAN", "FFNN", "LLM", "CSV", "MSE", "RMSE", "NLP", "ML", "API"}
GREETINGS = {"hi", "hello", "hey", "hii", "namaste", "yo", "hola"}
THANKS = {"thanks", "thank", "thx", "shukriya", "dhanyavad"}
GOODBYES = {"bye", "goodbye", "tata", "alvida"}

st.set_page_config(
    page_title="StudyMate", page_icon="📚", layout="centered",
    initial_sidebar_state="expanded" if SIDEBAR_OPEN else "auto",
)

# ---------- Look and feel ----------
def background_css():
    """If background.jpg / .png is in the folder, use it as the page background."""
    for name in ("background.jpg", "background.jpeg", "background.png"):
        if os.path.exists(name):
            mime = "image/png" if name.endswith("png") else "image/jpeg"
            with open(name, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            return f"background-image: url('data:{mime};base64,{data}'); background-size: cover; background-position: center top; background-repeat: no-repeat;"
    return None


def build_css():
    bg_image = background_css()
    hide_toolbar = "" if SHOW_TOOLBAR else '#MainMenu, [data-testid="stMainMenu"], [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"] { display: none; }'
    key_colors = "\n".join(
        f"div.st-key-sugg{i} {{ --c: {UNIT_COLORS[unit]}; }}" for i, (_, unit) in enumerate(SUGGESTIONS)
    )
    bg_rule = bg_image or """background-image:
    radial-gradient(50rem 36rem at 4% 0%, rgba(96, 62, 214, .55) 0%, transparent 62%),
    radial-gradient(46rem 34rem at 100% 4%, rgba(24, 96, 214, .50) 0%, transparent 62%),
    radial-gradient(46rem 36rem at 94% 100%, rgba(14, 140, 128, .42) 0%, transparent 60%),
    radial-gradient(40rem 32rem at 0% 98%, rgba(190, 52, 130, .36) 0%, transparent 60%);"""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,800&family=Atkinson+Hyperlegible:wght@400;700&display=swap');

:root {{
  --ink: #ECF0FF;
  --muted: #A9B4DC;
  --blue: #9DB4FF;
  --edge: rgba(255, 255, 255, .20);
  --glass: rgba(255, 255, 255, .075);
  --glass-strong: rgba(255, 255, 255, .15);
  --glass-shadow: 0 16px 44px rgba(0, 0, 0, .45), inset 0 1px 0 rgba(255,255,255,.26), inset 0 -1px 0 rgba(255,255,255,.04);
}}

/* deep night colour behind the glass, so the blur has something to bend */
.stApp {{
  font-family: 'Atkinson Hyperlegible', 'Segoe UI', system-ui, sans-serif;
  color: var(--ink);
  background-color: #060A1C;
  {bg_rule}
  background-attachment: fixed;
}}
footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
{hide_toolbar}
.block-container {{ max-width: 780px; padding-top: 2.2rem; padding-bottom: 7rem; }}

/* one glass recipe, used everywhere */
.answer, .plain, .note, .st-key-hero, [data-testid="stExpander"] details,
div[class*="st-key-sugg"] button, [data-testid="stChatInput"] > div {{
  background: var(--glass);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  backdrop-filter: blur(24px) saturate(180%);
  border: 1px solid var(--edge);
  box-shadow: var(--glass-shadow);
}}
@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {{
  .answer, .plain, .note, .st-key-hero, [data-testid="stExpander"] details,
  div[class*="st-key-sugg"] button, [data-testid="stChatInput"] > div {{ background: rgba(24,30,66,.92); }}
}}

/* welcome sheet */
.st-key-hero {{ border-radius: 32px; padding: 2.1rem 2rem 1.7rem; }}
.brand {{
  font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif;
  font-weight: 800; font-size: 3.2rem; line-height: 1.02; letter-spacing: -0.03em;
  color: var(--ink); margin: 0;
}}
.brand.small {{ font-size: 1.8rem; margin-bottom: .4rem; }}
.brand-sub {{ color: var(--muted); font-size: 1.12rem; line-height: 1.5; max-width: 34rem; margin: .6rem 0 1.5rem; }}

/* chat */
[data-testid="stChatMessage"] {{ background: transparent; padding: .4rem 0; }}
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"] {{ display: none; }}
.ask {{
  margin-left: auto; width: fit-content; max-width: 88%;
  background: linear-gradient(135deg, rgba(60,90,236,.95), rgba(124,92,255,.92));
  color: #fff; font-size: 1.05rem; line-height: 1.45;
  padding: .65rem 1rem; border-radius: 22px 22px 6px 22px;
  border: 1px solid rgba(255,255,255,.35);
  box-shadow: 0 12px 30px rgba(60,70,230,.4), inset 0 1px 0 rgba(255,255,255,.45);
}}

/* an answer: tinted glass, the glowing edge colour tells the unit */
.answer {{
  position: relative; overflow: hidden; border-radius: 24px;
  padding: 1.1rem 1.4rem 1.25rem 1.6rem;
  background: linear-gradient(135deg, color-mix(in srgb, var(--unit) 24%, rgba(255,255,255,.07)) 0%, rgba(255,255,255,.06) 60%);
  box-shadow: inset 6px 0 0 var(--unit), var(--glass-shadow);
}}
.answer::before {{
  content: ""; position: absolute; inset: 0; border-radius: inherit; pointer-events: none;
  background: linear-gradient(158deg, rgba(255,255,255,.16) 0%, rgba(255,255,255,0) 34%);
}}
.answer > * {{ position: relative; }}
.answer.small {{ padding: .85rem 1.1rem .95rem 1.35rem; border-radius: 20px; }}
.answer-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: .55rem; }}
.badge {{
  color: #0A0F2B; font-weight: 700; font-size: .85rem; padding: .12rem .7rem; border-radius: 999px;
  background: linear-gradient(180deg, color-mix(in srgb, var(--unit) 70%, white), var(--unit));
  box-shadow: inset 0 1px 0 rgba(255,255,255,.6), 0 2px 10px color-mix(in srgb, var(--unit) 55%, transparent);
}}
.unit-name {{
  font-family: 'Bricolage Grotesque', 'Segoe UI', system-ui, sans-serif;
  font-weight: 600; font-size: 1.1rem; color: var(--ink);
}}
.section {{ color: var(--muted); font-size: .95rem; margin: .25rem 0 .7rem; }}
.passage {{ color: var(--ink); font-size: 1.06rem; line-height: 1.7; max-width: 66ch; margin: 0; }}
.answer.small .passage {{ font-size: .98rem; line-height: 1.6; }}
.plain, .note {{ border-radius: 20px; padding: .9rem 1.2rem; color: var(--ink); font-size: 1.05rem; line-height: 1.55; }}
.note {{ border-style: dashed; border-color: rgba(255,255,255,.4); }}
.score {{ color: var(--muted); font-size: .88rem; margin-top: .5rem; }}

/* example doubts */
div[class*="st-key-sugg"], div[class*="st-key-sugg"] > div {{ width: 100% !important; }}
{key_colors}
div[class*="st-key-sugg"] button {{
  width: 100%; justify-content: flex-start; text-align: left; border-radius: 20px;
  color: var(--ink); font-weight: 700; font-size: 1.02rem; padding: .75rem 1.1rem .75rem 1.3rem;
  box-shadow: inset 6px 0 0 var(--c), var(--glass-shadow);
  transition: transform .15s ease, background .15s ease;
}}
div[class*="st-key-sugg"] button > div, div[class*="st-key-sugg"] button p {{
  width: 100%; text-align: left; justify-content: flex-start;
}}
div[class*="st-key-sugg"] button:hover {{ background: var(--glass-strong); color: #fff; border-color: rgba(255,255,255,.5); transform: translateY(-1px); }}
div[class*="st-key-sugg"] button:focus-visible {{ outline: 3px solid var(--c); outline-offset: 2px; }}

/* floating glass chat bar */
[data-testid="stBottom"], [data-testid="stBottom"] > div, [data-testid="stBottomBlockContainer"] {{ background: transparent !important; }}
[data-testid="stChatInput"] {{ background: transparent; }}
[data-testid="stChatInput"] > div {{ border-radius: 30px; }}
[data-testid="stChatInput"] textarea {{ background: transparent; color: var(--ink); }}
[data-testid="stChatInput"] textarea::placeholder {{ color: rgba(236,240,255,.55); }}

/* expanders */
[data-testid="stExpander"] details {{ border-radius: 18px; margin-top: .65rem; }}
[data-testid="stExpander"] summary {{ color: var(--blue); font-weight: 700; }}

/* sidebar */
[data-testid="stSidebar"] {{
  background: rgba(12, 16, 44, .5) !important;
  -webkit-backdrop-filter: blur(28px) saturate(180%); backdrop-filter: blur(28px) saturate(180%);
  border-right: 1px solid var(--edge);
}}
[data-testid="stSidebar"] > div:first-child {{ background: transparent; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{
  background: rgba(255,255,255,.09); border: 1px solid var(--edge); border-radius: 14px;
}}
.legend-row {{ display: flex; align-items: center; gap: .6rem; margin: .5rem 0; color: var(--ink); font-size: .96rem; line-height: 1.3; }}
.dot {{ flex: none; width: 13px; height: 13px; border-radius: 50%; box-shadow: 0 0 10px currentColor, inset 0 1px 0 rgba(255,255,255,.5); }}

@media (max-width: 640px) {{
  .brand {{ font-size: 2.4rem; }}
  .st-key-hero {{ padding: 1.4rem 1.1rem 1.2rem; border-radius: 26px; }}
  .answer {{ padding: .95rem 1rem 1.05rem 1.3rem; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
</style>
"""


st.markdown(build_css(), unsafe_allow_html=True)


# ---------- Model ----------
def build_index(chunks):
    """Build the TF-IDF search model from the passages (same recipe as train.py)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    stop = list(STOP_WORDS)
    vec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
    matrix = vec.fit_transform([c["section"] + ". " + c["text"] for c in chunks])
    tvec = TfidfVectorizer(stop_words=stop, ngram_range=(1, 2), sublinear_tf=True)
    tmatrix = tvec.fit_transform([c["section"].split(":")[0] for c in chunks])
    return {"vectorizer": vec, "matrix": matrix, "title_vectorizer": tvec, "title_matrix": tmatrix}


@st.cache_resource
def load_everything():
    """Load the passages and the search model once.
    If index.pkl is missing (for example on the online server), the model is built on the spot."""
    if not os.path.exists("book_chunks.json"):
        st.error("book_chunks.json not found. Put it in the same folder as app.py.")
        st.stop()
    with open("book_chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    index = None
    if os.path.exists("index.pkl"):
        try:
            index = joblib.load("index.pkl")
        except Exception:
            index = None
    if index is None:
        index = build_index(chunks)
    return index, chunks


index, chunks = load_everything()
vectorizer = index["vectorizer"]


def coverage(question):
    """Share of the question's meaningful words that appear somewhere in the book."""
    words = [w for w in re.findall(r"[a-z0-9]+", question.lower())
             if len(w) > 1 and w not in STOP_WORDS]
    if not words:
        return 0.0
    return sum(w in vectorizer.vocabulary_ for w in words) / len(words)


def search(question, unit_filter):
    """Return (list of (passage, score) best first, answerable?)."""
    q = question.lower()
    scores = linear_kernel(vectorizer.transform([q]), index["matrix"])[0]
    scores = scores + TITLE_WEIGHT * linear_kernel(index["title_vectorizer"].transform([q]), index["title_matrix"])[0]
    for i, c in enumerate(chunks):
        if c["section"] == "Overview":      # learning-objective lists are not answers
            scores[i] = 0
    order = scores.argsort()[::-1]
    if unit_filter:
        order = [i for i in order if chunks[i]["unit"] == unit_filter]
    results = [(chunks[i], float(scores[i])) for i in order[:3]]
    ok = bool(results) and results[0][1] >= MIN_SCORE and coverage(question) >= MIN_COVERAGE
    return results, ok


def following_passages(chunk, how_many=2):
    """The next passages from the same section of the book."""
    out = []
    for c in chunks[chunk["id"] + 1:]:
        if c["unit"] != chunk["unit"] or c["section"] != chunk["section"]:
            break
        out.append(c)
        if len(out) == how_many:
            break
    return out


def make_reply(question, unit_filter):
    """Build the bot's reply as a dictionary the chat can display."""
    words = set(re.findall(r"[a-z]+", question.lower()))
    if words and words <= GREETINGS | {"there", "studymate", "bot"}:
        return {"kind": "plain", "text": "Hi! Ask me any doubt from your Class XII AI handbook."}
    if words & THANKS and len(words) <= 4:
        return {"kind": "plain", "text": "You're welcome! Ask me anything else from the handbook."}
    if words & GOODBYES and len(words) <= 3:
        return {"kind": "plain", "text": "Bye! All the best for your studies and exams."}

    results, ok = search(question, unit_filter)
    if not ok:
        return {"kind": "note"}

    best, best_score = results[0]
    return {
        "kind": "answer",
        "best": best,
        "more": following_passages(best),
        "related": [c for c, s in results[1:] if s >= 0.05],
        "details": f"Match score {best_score:.2f}. Question words found in the book: {coverage(question) * 100:.0f}%.",
    }


# ---------- Drawing the chat ----------
def pretty_section(name):
    """Tidy a section name from the book: cut attached sentences, fix ALL CAPS."""
    name = name.split(":")[0].strip(" -:")
    letters = [ch for ch in name if ch.isalpha()]
    if letters and all(ch.isupper() for ch in letters):
        words = name.split()
        name = " ".join(w if w.strip("?,.") in ACRONYMS else w.lower() for w in words)
        name = re.sub(r"^((?:\d+\.)*\d*\s*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), name)
    return name


def passage_html(text):
    text = re.sub(r"(Image Source:\s*)?https?://\S+", "", text)      # web links are not useful in the chat
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s*[●•]\s*", "<br>• ", html.escape(text))
    return text[len("<br>"):] if text.startswith("<br>") else text


def answer_card(chunk, small=False):
    color = UNIT_COLORS[chunk["unit"]]
    return (
        f'<div class="answer{" small" if small else ""}" style="--unit:{color}">'
        f'<div class="answer-head"><span class="badge">Unit {chunk["unit"]}</span>'
        f'<span class="unit-name">{html.escape(chunk["unit_title"])}</span></div>'
        f'<div class="section">Section {html.escape(pretty_section(chunk["section"]))}</div>'
        f'<p class="passage">{passage_html(chunk["text"])}</p></div>'
    )


def show_reply(reply, show_details):
    kind = reply["kind"]
    if kind == "plain":
        st.markdown(f'<div class="plain">{html.escape(reply["text"])}</div>', unsafe_allow_html=True)
    elif kind == "note":
        st.markdown(
            '<div class="note"><b>This is not in the handbook, or I could not match it.</b><br>'
            "Try the main keyword of your doubt, like <i>neural network</i>, <i>confusion matrix</i>, "
            "<i>Orange widgets</i> or <i>data storytelling</i>.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(answer_card(reply["best"]), unsafe_allow_html=True)
        if reply["more"]:
            with st.expander("Continue reading this section"):
                for chunk in reply["more"]:
                    st.markdown(answer_card(chunk, small=True), unsafe_allow_html=True)
        if reply["related"]:
            with st.expander("Other places in the handbook that match"):
                for chunk in reply["related"]:
                    st.markdown(answer_card(chunk, small=True), unsafe_allow_html=True)
        if show_details:
            st.markdown(f'<div class="score">{html.escape(reply["details"])}</div>', unsafe_allow_html=True)


# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("Search in")
    choice = st.selectbox(
        "Search in",
        ["All units"] + [f"Unit {n}: {t}" for n, t in UNITS.items()],
        label_visibility="collapsed",
    )
    unit_filter = None if choice == "All units" else int(choice.split(":")[0].replace("Unit ", ""))
    show_details = st.checkbox("Show match score", value=False)
    st.markdown("**Unit colours**")
    st.markdown(
        "".join(
            f'<div class="legend-row"><span class="dot" style="background:{UNIT_COLORS[n]};color:{UNIT_COLORS[n]}"></span>'
            f"<span>Unit {n}: {html.escape(t)}</span></div>"
            for n, t in UNITS.items()
        ),
        unsafe_allow_html=True,
    )

# ---------- Main page ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

question = st.chat_input("Type your doubt here...") or st.session_state.pop("pending", None)
if question:
    st.session_state.messages.append({"role": "user", "text": question})
    st.session_state.messages.append({"role": "assistant", "reply": make_reply(question, unit_filter)})

LOGO = next((f for f in ("logo.png", "logo.jpg", "logo.jpeg", "logo.svg") if os.path.exists(f)), None)


def brand(small=False):
    """Your own logo if you put logo.png in this folder, otherwise the StudyMate wordmark."""
    if LOGO:
        st.image(LOGO, width=150 if small else 230)
    else:
        st.markdown(f'<h1 class="brand{" small" if small else ""}">StudyMate</h1>', unsafe_allow_html=True)


if not st.session_state.messages:
    with st.container(key="hero"):
        brand()
        st.markdown(
            '<p class="brand-sub">Ask a doubt from your Class XII AI handbook. '
            "Every answer shows the unit and section it came from.</p>",
            unsafe_allow_html=True,
        )
        for i, (text, _unit) in enumerate(SUGGESTIONS):
            if st.button(text, key=f"sugg{i}"):
                st.session_state.pending = text
                st.rerun()
else:
    brand(small=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f'<div class="ask">{html.escape(message["text"])}</div>', unsafe_allow_html=True)
            else:
                show_reply(message["reply"], show_details)
