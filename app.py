"""
Food Verse AI — Multimodal Food Intelligence Agent
Skill Wallet Capstone Project 3

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Safety note: image-based freshness and hygiene checks are advisory, not a substitute
for food-safety procedures, temperature logs, or qualified inspection.
"""
from __future__ import annotations

import base64
import json
import math
import os
import random
import re
from datetime import date
from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image

APP_TITLE = "Food Verse AI"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

st.set_page_config(
    page_title="Food Verse AI",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Design ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { --fv-ink:#172033; --fv-muted:#64748b; --fv-violet:#6d28d9; --fv-coral:#fb7185; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.035em; }
.stApp { background: linear-gradient(145deg,#fff9f3 0%,#f7f5ff 42%,#effaff 100%); }
.block-container { padding-top:1.5rem; padding-bottom:3rem; max-width:1400px; }
.hero { padding:1.6rem 1.8rem; border-radius:24px; color:white;
 background:linear-gradient(115deg,#4c1d95 0%,#7c3aed 48%,#ec4899 100%);
 box-shadow:0 18px 45px rgba(109,40,217,.18); margin-bottom:1.1rem; }
.hero h1 { color:white; margin:0; font-size:2.6rem; }
.hero p { color:#f3e8ff; margin:.45rem 0 0; font-size:1.04rem; }
.pill { display:inline-block; background:#ffffff24; border:1px solid #ffffff4a; border-radius:99px;
 padding:.3rem .7rem; margin-bottom:.7rem; font-size:.78rem; letter-spacing:.05em; }
.metric-card { background:rgba(255,255,255,.82); border:1px solid #e7e5f4; border-radius:18px;
 padding:1rem 1.1rem; box-shadow:0 5px 18px rgba(38,38,70,.04); }
.small-muted { color:#64748b; font-size:.88rem; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.82); border:1px solid #e7e5f4;
 padding:14px 16px; border-radius:16px; }
.stButton>button { border-radius:12px; font-weight:700; }
section[data-testid="stSidebar"] { background:linear-gradient(180deg,#f5f0ff,#fff9f3); }
div[data-testid="stTabs"] button { font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------- Session and helpers ----------
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "provider" not in st.session_state:
    st.session_state.provider = "Demo mode"
if "model" not in st.session_state:
    st.session_state.model = DEFAULT_GEMINI_MODEL

def safe_json(text: str) -> dict[str, Any]:
    """Parse JSON from model output, including fenced JSON."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {"result": data}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
                return data if isinstance(data, dict) else {"result": data}
            except json.JSONDecodeError:
                pass
        return {"summary": cleaned[:6000]}

def image_to_data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    from io import BytesIO
    buf = BytesIO()
    image.convert("RGB").save(buf, format=fmt, quality=88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

def call_vision_or_text(prompt: str, image: Image.Image | None = None, max_tokens: int = 1800) -> dict[str, Any]:
    """Call Gemini or Groq if configured; use clearly labelled local demo otherwise."""
    provider = st.session_state.get("provider", "Demo mode")
    key = st.session_state.get("api_key", "").strip()
    if provider == "Demo mode" or not key:
        return {"_demo": True, "summary": "Demonstration mode is active. Results below are illustrative and were not inferred from your image."}
    try:
        if provider == "Gemini":
            model = st.session_state.get("model") or DEFAULT_GEMINI_MODEL
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            parts: list[dict[str, Any]] = [{"text": prompt}]
            if image is not None:
                from io import BytesIO
                buf = BytesIO()
                image.convert("RGB").save(buf, format="JPEG", quality=88)
                parts.append({"inline_data": {"mime_type": "image/jpeg", "data": base64.b64encode(buf.getvalue()).decode("ascii")}})
            response = requests.post(
                url, params={"key": key},
                json={"contents": [{"parts": parts}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}},
                timeout=75,
            )
            response.raise_for_status()
            payload = response.json()
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return safe_json(text)
        if provider == "Groq":
            model = st.session_state.get("model") or DEFAULT_GROQ_MODEL
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            if image is not None:
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.2, "max_tokens": max_tokens, "response_format": {"type": "json_object"}},
                timeout=75,
            )
            response.raise_for_status()
            return safe_json(response.json()["choices"][0]["message"]["content"])
        return {"error": "Unknown provider. Switch to Demo mode or select Gemini/Groq."}
    except requests.HTTPError as exc:
        body = ""
        try:
            body = exc.response.text[:800]
        except Exception:
            pass
        return {"error": f"API request failed ({exc.response.status_code if exc.response is not None else 'HTTP'}). Check the key, model name, quota, and provider access.", "details": body}
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        return {"error": f"Could not complete model request: {type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"error": f"Unexpected model error: {type(exc).__name__}: {exc}"}

def display_result(data: dict[str, Any]) -> None:
    if data.get("_demo"):
        st.info(data.get("summary", "Demo mode is active."))
    if data.get("error"):
        st.error(data["error"])
        if data.get("details"):
            with st.expander("Technical details"):
                st.code(data["details"])
    if data.get("summary") and not data.get("_demo"):
        st.write(data["summary"])

def model_json(data: dict[str, Any], title: str = "Model response") -> None:
    with st.expander(title, expanded=False):
        st.json(data)

def score_color(score: int) -> str:
    if score >= 80:
        return "🟢"
    if score >= 60:
        return "🟡"
    return "🔴"

def demo_food_result() -> dict[str, Any]:
    return {
        "dish_name": "Vegetable poha",
        "cuisine": "Indian / Madhya Pradesh",
        "ingredients": ["flattened rice", "onion", "potato", "peas", "mustard seeds", "oil", "lemon", "coriander"],
        "estimated_portion": "1 medium bowl (~280 g)",
        "nutrition": {"calories_kcal": 340, "protein_g": 7.5, "carbohydrates_g": 57, "fat_g": 9.5, "fiber_g": 5.2},
        "allergen_alerts": ["May contain peanuts depending on preparation", "Check packaged ingredients for gluten cross-contact"],
        "health_score": 76,
        "health_notes": ["Add a protein side such as curd or sprouts", "Oil and portion size can materially change the estimate"],
        "_demo": True,
        "summary": "Illustrative example only. The uploaded photo was not analysed in demo mode."
    }

def demo_freshness_result() -> dict[str, Any]:
    return {
        "item": "Tomato (illustrative)",
        "freshness_stage": "Moderately Fresh",
        "visual_observations": ["Slight surface wrinkling", "No obvious visible mold in this example"],
        "recommended_action": "Consume within 24h",
        "storage_tips": ["Keep dry and remove damaged items", "Refrigerate cut tomato in a clean covered container"],
        "caution": "A photo cannot reliably detect pathogens, internal spoilage, or unsafe storage temperature.",
        "_demo": True,
        "summary": "Illustrative result only. Demo mode does not inspect your uploaded image."
    }

def demo_hygiene_result() -> dict[str, Any]:
    return {
        "compliance_score_percent": 72,
        "observations": ["Example only: hair restraint not clearly visible", "Example only: raw and ready-to-eat items should be separated"],
        "violations": [
            {"severity": "High", "issue": "Potential raw/ready-to-eat cross-contamination", "action": "Separate by time and space; use dedicated boards and utensils"},
            {"severity": "Medium", "issue": "Hair restraint cannot be confirmed", "action": "Require a clean hat or hairnet covering all hair"},
            {"severity": "Low", "issue": "Sanitizer concentration/log not verifiable from an image", "action": "Verify approved sanitizer concentration and contact time"}
        ],
        "limitations": "Images cannot verify handwashing, sanitizer concentration, temperatures, or activity outside the frame.",
        "_demo": True,
        "summary": "Illustrative checklist only. Demo mode does not inspect uploaded media."
    }

def format_grocery(meals: list[dict[str, str]]) -> list[str]:
    counts: dict[str, int] = {}
    for meal in meals:
        for item in meal.get("ingredients", "").split(","):
            item = item.strip().lower()
            if item:
                counts[item] = counts.get(item, 0) + 1
    return sorted(counts)

# ---------- Sidebar / configuration ----------
with st.sidebar:
    st.markdown("## 🍜 Food Verse AI")
    st.caption("Multimodal Food Intelligence Agent")
    page = st.radio(
        "Explore intelligence pillars",
        [
            "Overview",
            "1 · Food Recognition & Nutrition",
            "2 · Freshness & Spoilage",
            "3 · Kitchen Hygiene",
            "4 · Smart Recipe Assistant",
            "5 · Personalized Meal Planner",
        ],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("### ⚙️ AI configuration")
    provider = st.selectbox("Inference provider", ["Demo mode", "Gemini", "Groq"], index=["Demo mode", "Gemini", "Groq"].index(st.session_state.provider))
    st.session_state.provider = provider
    if provider != "Demo mode":
        entered_key = st.text_input(
            f"{provider} API key",
            type="password",
            value=st.session_state.api_key,
            help="Key is held in this Streamlit session only. Do not paste it into source code or commit it.",
            placeholder="Paste API key",
        )
        st.session_state.api_key = entered_key.strip()
        if provider == "Gemini":
            st.session_state.model = st.text_input("Model ID", value=st.session_state.model if "gemini" in st.session_state.model else DEFAULT_GEMINI_MODEL)
        else:
            st.session_state.model = st.text_input("Model ID", value=st.session_state.model if "llama" in st.session_state.model else DEFAULT_GROQ_MODEL)
        if not st.session_state.api_key:
            st.warning("Add an API key to enable live inference. Demo mode remains available.")
    else:
        st.session_state.api_key = ""
        st.info("No API key required. Demo outputs are clearly labelled illustrative.")
    st.divider()
    st.caption("Food safety note: AI output is advisory. When in doubt, follow official food-safety guidance and discard questionable food.")

# ---------- Header ----------
st.markdown("""
<div class="hero">
  <span class="pill">SKILL WALLET · CAPSTONE PROJECT 3</span>
  <h1>Food Verse AI</h1>
  <p>See your food differently — nutrition intelligence, freshness diagnostics, kitchen safety, recipes, and adaptive meal planning.</p>
</div>
""", unsafe_allow_html=True)

if page == "Overview":
    st.subheader("Your kitchen, powered by food intelligence.")
    st.write("A modular multimodal assistant designed to turn food photos, pantry ingredients, and personal goals into practical decisions.")
    cols = st.columns(5)
    cards = [
        ("📸", "Nutrition audit", "Dish, macros, allergens"),
        ("🍓", "Freshness scan", "Visual spoilage cues"),
        ("🧤", "Kitchen hygiene", "Compliance observations"),
        ("🥘", "Recipe studio", "Cook from your pantry"),
        ("🗓️", "Meal planner", "Targets, 7-day plan"),
    ]
    for col, (emoji, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:1.8rem">{emoji}</div><b>{title}</b><p class="small-muted">{desc}</p></div>', unsafe_allow_html=True)
    st.markdown("### Quick start")
    st.markdown("1. Choose an intelligence pillar from the sidebar.\n2. Use **Demo mode** to explore the interface without credentials, or add a Gemini/Groq API key for live model inference.\n3. Review model outputs critically; nutrition values are estimates and photos cannot establish microbiological safety.")
    st.info("Deployment-ready structure: one Streamlit app, no external database, and API keys entered at runtime.")
    st.markdown("### System status")
    a, b, c = st.columns(3)
    a.metric("Inference mode", st.session_state.provider)
    b.metric("Database", "Not required")
    c.metric("Session key", "Configured" if st.session_state.api_key else "Not set")
    st.stop()

# ---------- Tab 1: Food Recognition ----------
if page == "1 · Food Recognition & Nutrition":
    st.subheader("📸 Food Recognition & Nutrition Audit")
    st.caption("Epic 2 · Identify the meal, estimate nutrition, and surface possible allergen concerns.")
    uploaded = st.file_uploader("Upload a meal or dish photo", type=["jpg", "jpeg", "png", "webp"], key="food_image")
    portion = st.selectbox("Portion context", ["Single serving", "Half serving", "Large serving", "Shared dish"], index=0)
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded meal image", use_container_width=True)
            if st.button("✨ Analyse meal", type="primary", key="analyse_food"):
                prompt = f"""
You are a cautious food-science assistant. Inspect the supplied meal image and return ONLY valid JSON with:
dish_name, cuisine, ingredients (array), estimated_portion, nutrition object with calories_kcal, protein_g, carbohydrates_g, fat_g, fiber_g (numbers), allergen_alerts (array), health_score (integer 1-100), health_notes (array), confidence_notes.
User portion context: {portion}.
Estimate visible portion and nutrition transparently. Do not claim exact measurements from pixels. Mention uncertainty and hidden oil/sauces. Allergen alerts must say possible/verify rather than asserting allergens that cannot be seen. Health score is a heuristic, not a medical rating. If the image is ambiguous, say so.
"""
                result = call_vision_or_text(prompt, image)
                if result.get("_demo"):
                    result = demo_food_result()
                st.session_state.food_result = result
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")
    elif st.session_state.provider == "Demo mode":
        st.caption("Try the demo result below without uploading a photo.")
        if st.button("Show sample nutrition audit"):
            st.session_state.food_result = demo_food_result()

    result = st.session_state.get("food_result")
    if result:
        display_result(result)
        nutrition = result.get("nutrition", {})
        if nutrition:
            st.markdown("### Nutrition snapshot")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Calories", f"{nutrition.get('calories_kcal', '—')} kcal")
            m2.metric("Protein", f"{nutrition.get('protein_g', '—')} g")
            m3.metric("Carbs", f"{nutrition.get('carbohydrates_g', '—')} g")
            m4.metric("Fat", f"{nutrition.get('fat_g', '—')} g")
            m5.metric("Fiber", f"{nutrition.get('fiber_g', '—')} g")
            macros = pd.DataFrame([
                {"Macro": "Protein", "Grams": float(nutrition.get("protein_g", 0) or 0)},
                {"Macro": "Carbohydrates", "Grams": float(nutrition.get("carbohydrates_g", 0) or 0)},
                {"Macro": "Fat", "Grams": float(nutrition.get("fat_g", 0) or 0)},
                {"Macro": "Fiber", "Grams": float(nutrition.get("fiber_g", 0) or 0)},
            ])
            left, right = st.columns([1.2, 1])
            with left:
                st.plotly_chart(px.bar(macros, x="Macro", y="Grams", color="Macro", title="Estimated macronutrients (g)", text_auto=".1f"), use_container_width=True)
            with right:
                score = result.get("health_score")
                if isinstance(score, (int, float)):
                    st.markdown("### Health score")
                    st.markdown(f"## {score_color(int(score))} {int(score)}/100")
                    st.progress(max(0, min(100, int(score))) / 100)
                    st.caption("Heuristic score; not a clinical assessment.")
                st.markdown("### Allergen alerts")
                for alert in result.get("allergen_alerts", []):
                    st.warning(str(alert))
            st.markdown("### Dish details")
            st.write("**Dish:**", result.get("dish_name", "Not identified"))
            st.write("**Cuisine:**", result.get("cuisine", "Uncertain"))
            st.write("**Estimated portion:**", result.get("estimated_portion", "Not estimated"))
            st.write("**Ingredients:**", ", ".join(map(str, result.get("ingredients", []))) or "Not confidently identified")
            for note in result.get("health_notes", []):
                st.markdown(f"- {note}")
        model_json(result, "Full structured analysis")
        st.download_button("Download nutrition report (JSON)", json.dumps(result, indent=2), file_name="food_verse_nutrition_report.json", mime="application/json")

# ---------- Tab 2: Freshness ----------
elif page == "2 · Freshness & Spoilage":
    st.subheader("🍓 Food Freshness & Spoilage Diagnostic")
    st.caption("Epic 3 · Review visible quality cues and choose a conservative next step.")
    uploaded = st.file_uploader("Upload fruit, vegetable, meat, fish, or dairy photo", type=["jpg", "jpeg", "png", "webp"], key="fresh_image")
    item_type = st.selectbox("Food category", ["Fruit", "Vegetable", "Raw meat or poultry", "Fish or seafood", "Dairy", "Other"])
    storage = st.selectbox("Storage history (if known)", ["Unknown", "Refrigerated continuously", "Room temperature", "Frozen", "Left out for more than 2 hours"])
    if uploaded:
        try:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded food item", use_container_width=True)
            if st.button("🔎 Assess visible freshness", type="primary", key="analyse_fresh"):
                prompt = f"""
Act as a conservative food-safety assistant. Food category: {item_type}. Reported storage history: {storage}.
Inspect the image for visible texture, discoloration, shriveling, bruising, liquid, and visible mold. Return ONLY JSON:
item, freshness_stage (exactly one of Fresh | Moderately Fresh | Spoiled), visual_observations (array), recommended_action (exactly one of Safe to consume | Consume within 24h | Discard immediately), storage_tips (array), caution, confidence.
Do not infer pathogen presence or safety from appearance alone. For raw meat/seafood/dairy with uncertain history, visible normality does not prove safety. If visible mold/putrid appearance is suspected or history is unsafe, recommend discard. Never recommend tasting to check.
"""
                result = call_vision_or_text(prompt, image)
                if result.get("_demo"):
                    result = demo_freshness_result()
                st.session_state.fresh_result = result
        except Exception as exc:
            st.error(f"Could not open this image: {exc}")
    elif st.session_state.provider == "Demo mode" and st.button("Show sample freshness assessment"):
        st.session_state.fresh_result = demo_freshness_result()

    result = st.session_state.get("fresh_result")
    if result:
        display_result(result)
        stage = result.get("freshness_stage", "Unknown")
        action = result.get("recommended_action", "Review carefully")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Freshness stage")
            st.subheader(stage)
            if stage == "Fresh":
                st.success("No obvious visual warning detected — not a guarantee of safety.")
            elif stage == "Moderately Fresh":
                st.warning("Quality may be declining. Consider using promptly if storage history is safe.")
            elif stage == "Spoiled":
                st.error("Do not consume food suspected to be spoiled.")
            st.write("**Visible observations**")
            for item in result.get("visual_observations", []):
                st.markdown(f"- {item}")
        with col2:
            st.markdown("### Recommended action")
            if action == "Discard immediately":
                st.error(f"**{action}**")
            elif action == "Consume within 24h":
                st.warning(f"**{action}**")
            else:
                st.info(f"**{action}**")
            st.write("**Storage tips**")
            for tip in result.get("storage_tips", []):
                st.markdown(f"- {tip}")
        st.warning(result.get("caution", "Appearance alone cannot establish food safety."))
        model_json(result, "Full freshness report")
        st.download_button("Download freshness report (JSON)", json.dumps(result, indent=2), file_name="food_verse_freshness_report.json", mime="application/json")

# ---------- Tab 3: Kitchen hygiene ----------
elif page == "3 · Kitchen Hygiene":
    st.subheader("🧤 Kitchen Hygiene & Food Safety Compliance")
    st.caption("Epic 4 · A visual checklist for training and internal review, not formal certification.")
    uploaded = st.file_uploader("Upload kitchen image or short video", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "avi"], key="hygiene_media")
    media = None
    is_video = False
    if uploaded:
        ext = uploaded.name.lower().rsplit(".", 1)[-1]
        is_video = ext in {"mp4", "mov", "avi"}
        if is_video:
            st.video(uploaded)
            st.info("Video analysis samples a small number of frames. It may miss events between frames.")
        else:
            try:
                media = Image.open(uploaded).convert("RGB")
                st.image(media, caption="Kitchen environment", use_container_width=True)
            except Exception as exc:
                st.error(f"Could not open image: {exc}")
    if uploaded and st.button("🧼 Run hygiene checklist", type="primary", key="analyse_hygiene"):
        if is_video:
            # OpenCV is optional at runtime; fail gracefully if not installed or video decode fails.
            try:
                import cv2
                import tempfile
                from io import BytesIO
                video_bytes = uploaded.getvalue()
                with tempfile.NamedTemporaryFile(suffix="." + ext) as temp:
                    temp.write(video_bytes)
                    temp.flush()
                    cap = cv2.VideoCapture(temp.name)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                    frames = []
                    for fraction in (0.15, 0.5, 0.85):
                        if total <= 0:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, min(total - 1, int(total * fraction)))
                        ok, frame = cap.read()
                        if ok:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(Image.fromarray(frame))
                    cap.release()
                if frames:
                    media = frames[1] if len(frames) > 1 else frames[0]
                    st.caption(f"Sampled {len(frames)} frame(s) from video ({total} frames, approx. {fps:.1f} FPS). The model sees one representative frame in this version.")
                else:
                    st.error("Could not extract frames. Try a still image or a shorter MP4.")
            except ImportError:
                st.error("Video frame extraction requires opencv-python-headless. Install requirements.txt, or upload a still image.")
            except Exception as exc:
                st.error(f"Video processing failed: {exc}")
        if media is not None:
            prompt = """
Act as a commercial kitchen hygiene visual-audit assistant. Return ONLY JSON with:
compliance_score_percent (integer 0-100), observations (array), violations (array of objects with severity High|Medium|Low, issue, action), strengths (array), limitations.
Inspect only visible evidence for hair covering/chef hat, glove use where appropriate, clean work surfaces, sanitization evidence, hand-contact risks, raw poultry/meat near vegetables or ready-to-eat food, and separation of boards/utensils.
Do not claim that glove use proves clean hands or that an image verifies sanitization concentration, handwashing, time/temperature controls, or full compliance. If an indicator is obscured, say it cannot be confirmed. Score is a heuristic visual checklist, not regulatory certification.
"""
            result = call_vision_or_text(prompt, media)
            if result.get("_demo"):
                result = demo_hygiene_result()
            st.session_state.hygiene_result = result
    elif not uploaded and st.session_state.provider == "Demo mode" and st.button("Show sample hygiene checklist"):
        st.session_state.hygiene_result = demo_hygiene_result()

    result = st.session_state.get("hygiene_result")
    if result:
        display_result(result)
        score = result.get("compliance_score_percent")
        if isinstance(score, (int, float)):
            st.markdown("### Visual compliance snapshot")
            st.metric("Kitchen hygiene compliance score", f"{int(score)}%")
            st.progress(max(0, min(100, int(score))) / 100)
            st.caption("Heuristic visual score only; not an inspection, certification, or legal compliance determination.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Observed indicators**")
            for obs in result.get("observations", []):
                st.markdown(f"- {obs}")
            st.markdown("**Strengths**")
            for item in result.get("strengths", []):
                st.markdown(f"- {item}")
        with col2:
            st.markdown("**Violation log**")
            violations = result.get("violations", [])
            if violations:
                for v in violations:
                    severity = str(v.get("severity", "Review"))
                    if severity == "High":
                        st.error(f"**{severity} · {v.get('issue', 'Issue')}**\n\n{v.get('action', '')}")
                    elif severity == "Medium":
                        st.warning(f"**{severity} · {v.get('issue', 'Issue')}**\n\n{v.get('action', '')}")
                    else:
                        st.info(f"**{severity} · {v.get('issue', 'Issue')}**\n\n{v.get('action', '')}")
            else:
                st.success("No violations were returned. This does not prove the kitchen is compliant.")
        st.warning(result.get("limitations", "A visual review cannot verify every food-safety control."))
        model_json(result, "Full hygiene report")
        st.download_button("Download hygiene report (JSON)", json.dumps(result, indent=2), file_name="food_verse_hygiene_report.json", mime="application/json")

# ---------- Tab 4: Recipe assistant ----------
elif page == "4 · Smart Recipe Assistant":
    st.subheader("🥘 Smart Kitchen Recipe Assistant")
    st.caption("Epic 4 · Turn your pantry into practical meal ideas.")
    pantry = st.text_area("Available ingredients", placeholder="e.g. rice, chickpeas, tomatoes, spinach, onion, yogurt, cumin", height=100)
    preferences_col, time_col, servings_col = st.columns(3)
    with preferences_col:
        recipe_style = st.selectbox("Diet preference", ["No preference", "Vegetarian", "Vegan", "High protein", "Keto", "Gluten-free"])
    with time_col:
        max_time = st.slider("Maximum cooking time (minutes)", 10, 120, 35, step=5)
    with servings_col:
        servings = st.selectbox("Servings", [1, 2, 3, 4, 5, 6], index=1)
    extra = st.text_input("Optional goal or restriction", placeholder="e.g. low sodium, no peanuts, use leftovers")
    if st.button("🍳 Generate recipes", type="primary", disabled=not pantry.strip()):
        prompt = f"""
Create 3 realistic recipes primarily using these pantry ingredients: {pantry}.
Diet preference: {recipe_style}. Maximum cooking time: {max_time} minutes. Servings: {servings}. Additional goal/restriction: {extra or 'none'}.
Return ONLY JSON: recipes array of objects with name, description, cooking_time_minutes, difficulty (Easy|Medium|Hard), servings, ingredients (array of strings with quantities), steps (array of numbered step strings), substitutions (array of strings), nutrition_estimate (object with calories_kcal_per_serving, protein_g, carbohydrates_g, fat_g), allergen_notes (array).
Do not invent unavailable ingredients as if already in pantry; mark extras clearly. Give safe cooking advice for animal products and flag cross-contact risks for allergies. Nutrition is approximate.
"""
        result = call_vision_or_text(prompt)
        if result.get("_demo"):
            ingredients = [x.strip() for x in pantry.split(",") if x.strip()]
            result = {
                "_demo": True,
                "summary": "Demo recipes are generated from a generic template and are not tailored by an AI model.",
                "recipes": [
                    {"name": "Quick pantry bowl", "description": "A flexible one-pan meal using pantry staples.", "cooking_time_minutes": min(max_time, 25), "difficulty": "Easy", "servings": servings,
                     "ingredients": ingredients[:8] + ["salt, pepper, and oil as needed"],
                     "steps": ["Wash and prepare the vegetables or other fresh ingredients.", "Heat a little oil in a pan and cook aromatics until fragrant.", "Add the remaining ingredients and cook until tender and safely heated.", "Taste, season, and serve."],
                     "substitutions": ["Use lemon juice or vinegar for brightness.", "Use beans, lentils, tofu, or yogurt for a protein boost where compatible."],
                     "nutrition_estimate": {"calories_kcal_per_serving": 350, "protein_g": 14, "carbohydrates_g": 48, "fat_g": 10}, "allergen_notes": ["Check labels and avoid cross-contact with known allergens."]},
                    {"name": "Warm spiced skillet", "description": "A simple skillet with spices adjusted to taste.", "cooking_time_minutes": min(max_time, 30), "difficulty": "Easy", "servings": servings,
                     "ingredients": ingredients[:7] + ["cumin or paprika (optional)"],
                     "steps": ["Prep and chop ingredients evenly.", "Cook firm ingredients first, adding softer ingredients later.", "Add spices and a splash of water; simmer until cooked through.", "Serve hot and refrigerate leftovers promptly."],
                     "substitutions": ["Swap vegetables based on what is available.", "Use herbs instead of extra salt."],
                     "nutrition_estimate": {"calories_kcal_per_serving": 320, "protein_g": 12, "carbohydrates_g": 42, "fat_g": 9}, "allergen_notes": ["Confirm packaged ingredient labels."]},
                    {"name": "No-waste leftover plate", "description": "A mix-and-match plate for ingredients that need using soon.", "cooking_time_minutes": min(max_time, 20), "difficulty": "Easy", "servings": servings,
                     "ingredients": ingredients[:8],
                     "steps": ["Check leftovers were stored safely; discard anything doubtful.", "Reheat leftovers thoroughly where applicable.", "Combine with fresh ingredients and a balanced seasoning.", "Serve immediately."],
                     "substitutions": ["Replace grains with potatoes or another available staple.", "Add seeds or legumes if suitable."],
                     "nutrition_estimate": {"calories_kcal_per_serving": 300, "protein_g": 10, "carbohydrates_g": 40, "fat_g": 8}, "allergen_notes": ["Do not use ingredients with uncertain storage history."]}
                ]
            }
        st.session_state.recipe_result = result
    result = st.session_state.get("recipe_result")
    if result:
        display_result(result)
        recipes = result.get("recipes", [])
        if recipes:
            for idx, recipe in enumerate(recipes, 1):
                with st.expander(f"{idx}. {recipe.get('name', 'Recipe')} · {recipe.get('cooking_time_minutes', '—')} min", expanded=(idx == 1)):
                    st.write(recipe.get("description", ""))
                    a, b, c = st.columns(3)
                    a.metric("Cook time", f"{recipe.get('cooking_time_minutes', '—')} min")
                    b.metric("Difficulty", recipe.get("difficulty", "—"))
                    c.metric("Servings", recipe.get("servings", servings if 'servings' in locals() else 2))
                    st.markdown("**Ingredients**")
                    for item in recipe.get("ingredients", []):
                        st.markdown(f"- {item}")
                    st.markdown("**Method**")
                    for step_i, step in enumerate(recipe.get("steps", []), 1):
                        st.markdown(f"{step_i}. {step}")
                    st.markdown("**Substitutions**")
                    for item in recipe.get("substitutions", []):
                        st.markdown(f"- {item}")
                    nutrition = recipe.get("nutrition_estimate", {})
                    if nutrition:
                        st.caption("Approximate nutrition per serving: " + " · ".join(f"{k.replace('_per_serving','').replace('_',' ')}: {v}" for k, v in nutrition.items()))
                    if recipe.get("allergen_notes"):
                        st.warning(" ".join(map(str, recipe["allergen_notes"])))
        model_json(result, "Full recipe response")
        st.download_button("Download recipes (JSON)", json.dumps(result, indent=2), file_name="food_verse_recipes.json", mime="application/json")

# ---------- Tab 5: Meal planner ----------
elif page == "5 · Personalized Meal Planner":
    st.subheader("🗓️ Personalized Meal Planner")
    st.caption("Epic 5 · Estimate daily energy needs and draft an adaptive 7-day schedule.")
    st.info("Energy estimates are approximations for generally healthy adults. This planner is not medical nutrition therapy. If you have diabetes, are pregnant, have an eating disorder history, or need therapeutic nutrition, consult a qualified clinician/dietitian.")
    with st.form("meal_profile"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=25)
            gender = st.selectbox("Sex used for BMR equation", ["Female", "Male", "Prefer not to specify"])
            height_cm = st.number_input("Height (cm)", min_value=120.0, max_value=230.0, value=170.0, step=0.5)
        with col2:
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, value=70.0, step=0.5)
            activity = st.selectbox("Activity level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])
            goal = st.selectbox("Fitness goal", ["Maintenance", "Fat Loss", "Muscle Gain"])
        with col3:
            diet = st.selectbox("Dietary pattern", ["Balanced", "Vegan", "Vegetarian", "Keto", "Diabetic-Friendly"])
            allergies = st.text_input("Allergies / exclusions", placeholder="e.g. peanuts, shellfish")
            cuisine = st.selectbox("Preferred cuisine", ["Indian", "Mediterranean", "Mixed", "East Asian"])
        submitted = st.form_submit_button("Calculate targets & build 7-day plan", type="primary")

    if submitted:
        # Mifflin-St Jeor equation. For unspecified sex, use the mean of female/male constants.
        base = 10 * float(weight_kg) + 6.25 * float(height_cm) - 5 * int(age)
        if gender == "Male":
            bmr = base + 5
        elif gender == "Female":
            bmr = base - 161
        else:
            bmr = base - 78
        activity_factor = {"Sedentary": 1.2, "Lightly active": 1.375, "Moderately active": 1.55, "Very active": 1.725, "Extra active": 1.9}[activity]
        tdee = bmr * activity_factor
        if goal == "Fat Loss":
            target = max(1200, tdee - min(500, tdee * 0.20))
        elif goal == "Muscle Gain":
            target = tdee + min(300, tdee * 0.10)
        else:
            target = tdee
        # Avoid creating a falsely precise clinical prescription.
        target = int(round(target / 50) * 50)
        allergies_list = [x.strip().lower() for x in allergies.split(",") if x.strip()]
        # Basic dietary template, clearly labelled as a draft. Model can personalize the menu if available.
        menu = {
            "Balanced": [
                ("Oats with milk, banana and seeds", "Dal, roti, mixed vegetables and curd", "Paneer/tofu stir-fry with rice", "Fruit and roasted chana"),
                ("Vegetable poha with curd", "Rajma rice with salad", "Egg/tofu bhurji with roti", "Yogurt and fruit"),
                ("Besan chilla and curd", "Chickpea salad bowl with roti", "Vegetable khichdi and raita", "Nuts and fruit"),
                ("Idli, sambar and chutney", "Paneer/tofu wrap with salad", "Dal, rice and sautéed vegetables", "Buttermilk and roasted makhana"),
                ("Peanut-free overnight oats", "Chole, roti and salad", "Vegetable pulao with raita", "Sprouts chaat"),
                ("Upma with vegetables and curd", "Moong dal khichdi and salad", "Grilled paneer/tofu with potatoes", "Milk or fortified soy drink"),
                ("Dosa with sambar", "Dal, roti and seasonal vegetables", "Chickpea and vegetable bowl", "Fruit with seeds"),
            ],
            "Vegan": [
                ("Oats with fortified soy milk, banana and seeds", "Dal, roti and vegetables", "Tofu stir-fry with rice", "Fruit and roasted chana"),
                ("Vegetable poha", "Rajma rice with salad", "Chickpea curry with roti", "Soy yogurt and fruit"),
                ("Besan chilla with tomato chutney", "Chickpea salad bowl", "Vegetable khichdi with tofu", "Nuts or seeds and fruit"),
                ("Idli and sambar", "Tofu wrap with salad", "Dal, rice and vegetables", "Roasted makhana"),
                ("Overnight oats with soy milk", "Chole, roti and salad", "Vegetable pulao with chickpeas", "Sprouts chaat"),
                ("Vegetable upma", "Moong dal khichdi", "Tofu with potatoes and greens", "Fortified soy drink"),
                ("Dosa with sambar", "Dal, roti and vegetables", "Chickpea and vegetable bowl", "Fruit with seeds"),
            ],
            "Vegetarian": [
                ("Oats with milk, banana and seeds", "Dal, roti, vegetables and curd", "Paneer stir-fry with rice", "Fruit and roasted chana"),
                ("Vegetable poha with curd", "Rajma rice and salad", "Paneer bhurji with roti", "Yogurt and fruit"),
                ("Besan chilla and curd", "Chickpea salad bowl", "Khichdi and raita", "Nuts and fruit"),
                ("Idli and sambar", "Paneer wrap with salad", "Dal, rice and vegetables", "Buttermilk and makhana"),
                ("Overnight oats", "Chole, roti and salad", "Vegetable pulao with raita", "Sprouts chaat"),
                ("Upma with curd", "Moong dal khichdi", "Paneer with potatoes and greens", "Milk and fruit"),
                ("Dosa with sambar", "Dal, roti and vegetables", "Chickpea and vegetable bowl", "Fruit with seeds"),
            ],
            "Keto": [
                ("Paneer/tofu scramble with spinach", "Cauliflower rice with paneer/tofu", "Zucchini and mushroom skillet", "Cucumber with yogurt dip"),
                ("Unsweetened yogurt or soy alternative with seeds", "Egg/tofu salad bowl", "Paneer tikka with greens", "Cheese/tofu cubes"),
                ("Avocado-free seed bowl with curd", "Mushroom paneer stir-fry", "Cauliflower pulao with tofu", "Nuts/seeds"),
                ("Besan is not low-carb; choose tofu scramble", "Grilled paneer/tofu salad", "Creamy spinach and paneer/tofu", "Cucumber and yogurt"),
                ("Paneer with sautéed vegetables", "Cauliflower bowl with tofu", "Mushroom and spinach skillet", "Seeds and unsweetened yogurt"),
                ("Tofu scramble", "Paneer lettuce wraps", "Vegetable and tofu stir-fry", "Olives or nuts"),
                ("Yogurt with chia seeds", "Paneer/tofu salad", "Cauliflower rice with greens", "Cucumber dip"),
            ],
            "Diabetic-Friendly": [
                ("Vegetable oats with unsweetened curd", "Dal, measured roti and vegetables", "Tofu/paneer with non-starchy vegetables", "Small fruit with nuts"),
                ("Besan chilla and curd", "Rajma with a measured rice portion and salad", "Vegetable dalia with tofu/paneer", "Plain yogurt"),
                ("Idli with extra sambar and vegetables", "Chickpea salad with measured roti", "Dal and mixed vegetables", "Roasted chana"),
                ("Unsweetened oats with seeds", "Paneer/tofu salad and roti", "Moong dal khichdi with vegetables", "Cucumber and curd"),
                ("Vegetable poha with sprouts", "Chole salad and measured roti", "Tofu/paneer with greens", "Small fruit and seeds"),
                ("Besan chilla", "Dal, vegetables and measured rice", "Vegetable soup with protein side", "Plain yogurt"),
                ("Dosa with sambar and vegetables", "Rajma salad bowl", "Dal, roti and vegetables", "Roasted chana"),
            ]
        }
        chosen_menu = menu[diet]
        meal_rows = []
        grocery_set = set()
        for day_idx, day_menu in enumerate(chosen_menu):
            breakfast, lunch, dinner, snack = day_menu
            meal_rows.append({"Day": f"Day {day_idx + 1}", "Breakfast": breakfast, "Lunch": lunch, "Dinner": dinner, "Snack": snack})
        st.session_state.meal_plan = {
            "profile": {"age": age, "gender_equation": gender, "height_cm": height_cm, "weight_kg": weight_kg, "activity": activity, "goal": goal, "diet": diet, "allergies": allergies_list, "cuisine": cuisine},
            "bmr": int(round(bmr)), "tdee": int(round(tdee)), "target": target,
            "schedule": meal_rows,
            "grocery_list": sorted(set([
                "oats", "banana", "seasonal fruit", "milk or fortified plant milk", "dal/lentils", "roti flour", "rice", "mixed vegetables", "tomatoes", "onions", "curd or suitable alternative", "beans/chickpeas/rajma", "paneer or tofu", "leafy greens", "seeds", "roasted chana", "spices", "lemon"
            ])),
            "notice": "Menu is a template, not a gram-by-gram calorie prescription. Portion sizes must be adjusted to the target and actual nutrition labels."
        }
    plan = st.session_state.get("meal_plan")
    if plan:
        st.markdown("### Your daily energy estimate")
        m1, m2, m3 = st.columns(3)
        m1.metric("BMR", f"{plan['bmr']:,} kcal/day")
        m2.metric("TDEE", f"{plan['tdee']:,} kcal/day")
        m3.metric("Daily target", f"{plan['target']:,} kcal/day")
        st.caption("BMR uses the Mifflin–St Jeor equation; TDEE = BMR × activity factor. Goal adjustment is a broad estimate.")
        st.markdown("### 7-day meal schedule")
        df = pd.DataFrame(plan["schedule"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown("### Automated grocery list")
        st.caption("Consolidated starter list for the template; verify it against your final portions, pantry stock, allergies, and chosen recipes.")
        grocery_cols = st.columns(3)
        grocery = plan["grocery_list"]
        for i, item in enumerate(grocery):
            with grocery_cols[i % 3]:
                st.checkbox(item, key=f"grocery_{i}_{item}")
        if plan["profile"]["allergies"]:
            st.warning("Allergy/exclusion reminders: " + ", ".join(plan["profile"]["allergies"]) + ". Check every label and avoid cross-contact. The generic template may need manual substitutions.")
        st.info(plan["notice"])
        st.download_button("Download meal plan (CSV)", df.to_csv(index=False).encode("utf-8"), file_name="food_verse_7_day_meal_plan.csv", mime="text/csv")
        st.download_button("Download grocery list (TXT)", "\n".join(grocery), file_name="food_verse_grocery_list.txt", mime="text/plain")
