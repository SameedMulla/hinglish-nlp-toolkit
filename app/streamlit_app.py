"""
Streamlit Web Application — Unified Code-Mixed NLP Demonstration Interface.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.inference.predictor import UnifiedPredictor
from src.preprocessing.normalizer import CodeMixedNormalizer
from src.utils.lang_utils import detect_script

# ---------------------------------------------------------------------------
# Page Config & CSS
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Code-Mixed NLP Toolkit (Hinglish)",
    page_icon="🔠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-title {
    font-size: 2.3rem; font-weight: 700;
    background: linear-gradient(90deg, #3B82F6, #8B5CF6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-title { font-size: 1.05rem; color: #6B7280; margin-bottom: 1.5rem; }
.metric-card {
    background: #F3F4F6; border-radius: 10px; padding: 15px;
    border-left: 5px solid #3B82F6; margin-bottom: 10px;
}
.entity-tag {
    display: inline-block; padding: 2px 8px; border-radius: 5px;
    font-weight: 600; font-size: 0.85rem; margin: 2px;
}
.tag-per { background: #FDE68A; color: #92400E; }
.tag-loc { background: #BFDBFE; color: #1E40AF; }
.tag-org { background: #BBF7D0; color: #166534; }
.tag-misc { background: #E9D5FF; color: #6B21A8; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuration")

lang_pair = st.sidebar.selectbox(
    "Language Pair",
    ["hi-en (Hindi-English)", "ta-en (Tamil-English) [Future]"],
    index=0,
)
lang_code = "hi-en" if "hi-en" in lang_pair else "ta-en"

task_choice = st.sidebar.radio(
    "Select NLP Task",
    [
        "🔄 Translation",
        "😊 Sentiment Analysis",
        "🏷️ Named Entity Recognition (NER)",
        "❓ Question Answering (QA)",
        "🎯 Intent Detection",
    ],
    index=0,
)

# Translation direction sub-option
translation_dir = "cm_to_en"
if "Translation" in task_choice:
    translation_dir = st.sidebar.radio(
        "Translation Direction",
        ["Code-Mixed → English", "English → Code-Mixed Hindi"],
        index=0,
    )
    translation_dir = "cm_to_en" if "English" in translation_dir.split("→")[1] else "en_to_cm"

st.sidebar.markdown("---")
st.sidebar.info(
    "**Code-Mixed Language Understanding**\n\n"
    "Unified framework supporting 5 NLP tasks for Hindi-English (Hinglish) code-mixed text.\n\n"
    "Uses Helsinki-NLP translation models, TF-IDF/CRF baselines, and Transformer fine-tuning."
)

TASK_MAP = {
    "🔄 Translation": "translation",
    "😊 Sentiment Analysis": "sentiment",
    "🏷️ Named Entity Recognition (NER)": "ner",
    "❓ Question Answering (QA)": "qa",
    "🎯 Intent Detection": "intent",
}

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">Code-Mixed NLP Multitask Toolkit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Unified Language Understanding for Hindi-English (Hinglish) Code-Mixed Text</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
@st.cache_resource
def get_predictor(lang: str) -> UnifiedPredictor:
    return UnifiedPredictor(lang_pair=lang)

predictor = get_predictor(lang_code)
normalizer = CodeMixedNormalizer(lang_pair=lang_code)

# ---------------------------------------------------------------------------
# Sample inputs
# ---------------------------------------------------------------------------
SAMPLES = {
    "🔄 Translation": {
        "cm_to_en": [
            "Aaj shaam ko hum sab friends milke party karenge.",
            "Delhi me weather today kafi pleasant hai.",
            "Yeh movie bohot acchi thi, direction mind blowing tha!",
            "IPL ka match Delhi Capitals aur Rajasthan Royals ke beech khela gaya.",
            "Mujhe Delhi se Mumbai ki flight ticket book karni hai.",
        ],
        "en_to_cm": [
            "The weather in Delhi is very pleasant today.",
            "I want to book a flight ticket from Delhi to Mumbai.",
            "This movie was really good, the direction was amazing!",
            "The IPL match was played between Delhi Capitals and Rajasthan Royals.",
        ],
    },
    "😊 Sentiment Analysis": [
        "Yeh movie bohot acchi thi, casting aur direction mind blowing tha!",
        "Khaana ekdum bekar tha, waste of money and time.",
        "Flight timing change ho gaya, okayish service.",
        "Product ki quality bahut achhi hai, highly recommended!",
        "Worst customer care ever, phone bhi attend nahi karte.",
    ],
    "🏷️ Named Entity Recognition (NER)": [
        "Shah Rukh Khan Mumbai me Mannat me rehte hain.",
        "Google ne Bangalore me naya R&D office open kiya hai.",
        "Narendra Modi Prime Minister of India ne Delhi me speech di.",
        "Virat Kohli ne BCCI ko letter likha hai.",
    ],
    "❓ Question Answering (QA)": [
        "Shah Rukh Khan kahan rehte hain?",
        "Google ne naya office kahan khola hai?",
        "Virat Kohli ne kisko letter likha?",
    ],
    "🎯 Intent Detection": [
        "Mujhe Delhi se Mumbai ki flight ticket book karni hai.",
        "Aaj Bangalore me rain hogi kya, weather report batao.",
        "Mere account ka balance kitna hai check karo.",
        "Ek pizza order karo Dominos se.",
        "Kal subah 6 baje ka alarm set karo.",
    ],
}

selected_task = TASK_MAP[task_choice]

# ---------------------------------------------------------------------------
# Input Area
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("📥 Input Text")

    if selected_task == "translation":
        sample_list = SAMPLES[task_choice].get(translation_dir, [])
    else:
        sample_list = SAMPLES.get(task_choice, [])

    preset = st.selectbox(
        "Choose a sample or type your own:",
        ["Custom..."] + sample_list,
    )
    default_val = sample_list[0] if preset == "Custom..." and sample_list else preset

    if selected_task == "translation" and translation_dir == "en_to_cm":
        input_label = "Enter English Text:"
        placeholder = "Type English text here..."
    else:
        input_label = "Enter Code-Mixed (Hinglish) Text:"
        placeholder = "Type Hinglish text here..."

    user_text = st.text_area(input_label, value=default_val, height=100, placeholder=placeholder)

    context_text = ""
    if selected_task == "qa":
        context_text = st.text_area(
            "Context Paragraph (Required for QA):",
            value="Shah Rukh Khan Indian film actor hain. Woh Mumbai me Mannat house me rehte hain. Unhone bohot saari superhit movies me acting ki hai.",
            height=120,
        )

    submit = st.button("🚀 Process", type="primary", use_container_width=True)

with col_right:
    st.subheader("🔍 Preprocessing Inspection")
    if user_text:
        cleaned = normalizer.normalize(user_text)
        script = detect_script(cleaned)
        c1, c2, c3 = st.columns(3)
        c1.metric("Dominant Script", script.dominant_script.capitalize())
        c2.metric("Mixed Script?", "Yes 🔀" if script.is_mixed_script else "No 🔤")
        c3.metric("Mix Ratio", f"{script.mix_ratio:.1%}")
        with st.expander("Normalized Text", expanded=False):
            st.code(f"Original  : {user_text}\nNormalized: {cleaned}", language="text")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if submit and user_text:
    st.markdown("---")
    st.subheader("🎯 Model Prediction Output")

    with st.spinner("Running inference..."):
        result = predictor.predict(
            text=user_text,
            task=selected_task,
            context=context_text if selected_task == "qa" else None,
            translation_direction=translation_dir,
        )

    st.caption(f"Model used: `{result.get('model_type', 'heuristic')}`")

    # ---------- Translation ----------
    if selected_task == "translation":
        direction_label = result.get("direction", "")
        st.info(f"**Direction:** {direction_label}")
        st.success(f"**Translation Output:**\n\n{result['translation']}")

    # ---------- Sentiment ----------
    elif selected_task == "sentiment":
        s_val = result["sentiment"].capitalize()
        conf = result["confidence"]
        color_map = {"Positive": "green", "Negative": "red", "Neutral": "orange"}
        badge = color_map.get(s_val, "blue")
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Sentiment: <span style="color:{badge};">{s_val}</span></h3>
            <p>Confidence Score: <b>{conf:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        st.progress(conf)

    # ---------- NER ----------
    elif selected_task == "ner":
        entities = result.get("entities", [])
        if not entities:
            st.info("No named entities detected.")
        else:
            st.write("**Extracted Entities:**")
            for ent in entities:
                tag_class = f"tag-{ent['entity'].lower()}"
                st.markdown(
                    f"<span class='entity-tag {tag_class}'>{ent['entity']}</span> "
                    f"<b>{ent['word']}</b> (Confidence: {ent['score']:.2%})",
                    unsafe_allow_html=True,
                )

    # ---------- QA ----------
    elif selected_task == "qa":
        st.markdown(f"""
        <div class="metric-card">
            <h3>Extracted Answer: <span style="color:#1E40AF;">{result['answer']}</span></h3>
            <p>Confidence Score: <b>{result['confidence']:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)

    # ---------- Intent ----------
    elif selected_task == "intent":
        intent_val = result["intent"]
        conf = result["confidence"]
        st.markdown(f"""
        <div class="metric-card">
            <h3>Detected Intent: <span style="color:#6B21A8;">{intent_val}</span></h3>
            <p>Confidence Score: <b>{conf:.2%}</b></p>
        </div>
        """, unsafe_allow_html=True)
        top_k = result.get("top_k", [])
        if top_k:
            st.write("**Top Candidates:**")
            for ik, sc in top_k:
                st.write(f"- `{ik}`: {sc:.2%}")
