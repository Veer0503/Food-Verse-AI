# Food Verse AI — Multimodal Food Intelligence Agent
**Skill Wallet Capstone Project 3**

A Streamlit application for food-photo nutrition estimates, visible freshness review, kitchen hygiene checklists, pantry recipe ideas, and a seven-day meal planner. It has no database dependency and supports a no-key demonstration mode.

## Features
- **Food Recognition & Nutrition Audit:** multimodal dish identification, ingredient and portion estimates, calories/macros/fiber, possible allergen alerts, heuristic health score, interactive chart, JSON export.
- **Freshness & Spoilage Diagnostic:** visible texture/discoloration/mold cues, one of three freshness labels, conservative action guidance, storage tips.
- **Kitchen Hygiene:** image and short-video upload, sampled-frame visual checklist, hair covering, gloves, surface/sanitization evidence, cross-contamination observations, itemized violations and score.
- **Recipe Assistant:** pantry inventory, diet preference, time and servings, three recipe ideas with instructions, substitutions and approximate nutrition.
- **Meal Planner:** adult profile, Mifflin–St Jeor BMR, activity-adjusted TDEE, goal adjustment, seven-day schedule and grocery list.

## Run locally
```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Configure AI
Choose **Gemini** or **Groq** in the sidebar, enter your API key in the password field, and set a supported model ID. The key is kept in Streamlit session state and is not written to a file by this app. For deployment, use Streamlit Community Cloud secrets or the provider's recommended secret-management approach if you adapt the code. Never commit API keys.

Suggested model IDs (provider availability can change):
- Gemini: `gemini-2.5-flash`
- Groq vision: `meta-llama/llama-4-scout-17b-16e-instruct`

If a model is unavailable to your account, choose another vision-capable model supported by that provider. Demo mode does not call an API and its outputs are clearly labelled illustrative.

## Deploy to Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository and set the main file to `app.py`.
3. Deploy. Start in Demo mode to verify the interface.
4. For live inference, enter the key at runtime. For a team deployment, prefer Streamlit secrets and restrict repository/app access appropriately.

No external database is required. The app uses in-memory session state and downloads reports directly to the user's device.

## Safety, reliability, and limitations
- Nutrition is an estimate from a photo, not a laboratory analysis. Hidden ingredients, oil, sauce, and serving mass can change results materially.
- Allergens cannot be reliably confirmed visually. Always check labels, recipes, and cross-contact controls.
- **A photo cannot prove that food is microbiologically safe.** Pathogens and unsafe temperatures may not be visible. Never taste food to test spoilage. Follow local food-safety requirements and discard food when safety is uncertain.
- Hygiene score is a visual heuristic, not a regulatory inspection or certification. A photo cannot verify handwashing, sanitizer concentration/contact time, temperatures, or activity outside the frame.
- The meal planner is a general adult template, not medical nutrition therapy. Diabetic-friendly means a general meal pattern only and does not replace clinician/dietitian guidance. Keto and other restrictive diets may not be suitable for everyone.
- Review generated recipes for allergies and safe cooking temperatures before use.

## Project structure
```text
food_verse_ai/
├── app.py
├── requirements.txt
└── README.md
```

## Troubleshooting
- **401/403:** check key permissions and model access.
- **429:** quota/rate limit; retry later or switch to Demo mode.
- **Model not found:** use a currently supported model ID for your provider.
- **Video cannot be decoded:** use a short MP4 or upload a still image; browser/server codecs can vary.
