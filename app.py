import os
import asyncio
from datetime import datetime, timezone
import streamlit as st
from google import genai
from google.genai import errors as genai_errors
from openai import AsyncOpenAI

# ---------------------------------------------------------
# 1. API KEYS (read from environment variables, never hardcoded)
# ---------------------------------------------------------
# Before running (macOS terminal):
#   export GEMINI_API_KEY="your-key"
#   export GROQ_API_KEY="your-key"
#   streamlit run chatFCB_v1_2.py
#
# Or use a .env file with `pip install python-dotenv`, then uncomment:
# from dotenv import load_dotenv
# load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="Multi-Model AI Assistant", layout="wide")

if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error(
        "Missing API key. Please set the GEMINI_API_KEY and GROQ_API_KEY "
        "environment variables before running `streamlit run chatFCB_v1_2.py`."
    )
    st.stop()

# Gemini model candidates, tried in order. Google has been retiring/renaming
# Flash models quickly lately, so a 404 (model no longer exists / not
# available to this project) automatically falls through to the next one.
# Check the live list + quota for your project at:
#   https://ai.google.dev/gemini-api/docs/models
#   https://aistudio.google.com/rate-limit
GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",     # latest stable (GA) Flash model, released 2026-07-21
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",  # alias that always points at Google's current default
]

# Display name for each ensemble member (used in the UI and in the leader prompt)
MODEL_GEMINI_LABEL = "Gemini"
MODEL_QWEN_LABEL = "Qwen 3.6 27B"
MODEL_GPTOSS_LABEL = "GPT-OSS 120B"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ---------------------------------------------------------
# 2. ASYNC API CALLS — retry + model fallback + run in a worker thread
# ---------------------------------------------------------
async def fetch_gemini(prompt: str, models: list = None, retries_per_model: int = 1) -> str:
    """Call Gemini, trying each model in `models` in order.
    - 404 (model no longer exists / not available) -> move to the next model immediately.
    - 429 (rate limit) -> wait and retry the current model a couple of times first.
    Runs in a worker thread (asyncio.to_thread) since the SDK call is sync,
    so it doesn't block the event loop while the Groq calls run in parallel."""
    models = models or GEMINI_MODEL_CANDIDATES
    last_err = None
    for model in models:
        for attempt in range(retries_per_model + 1):
            try:
                response = await asyncio.to_thread(
                    gemini_client.models.generate_content,
                    model=model,
                    contents=prompt,
                )
                return response.text
            except genai_errors.ClientError as e:
                last_err = e
                code = getattr(e, "code", None)
                if code == 429 and attempt < retries_per_model:
                    await asyncio.sleep(3 * (attempt + 1))
                    continue
                break  # 404 or out of retries -> try next model
            except Exception as e:
                last_err = e
                break
    return f"[Error from Gemini (tried: {', '.join(models)})]: {last_err}"


async def fetch_groq(prompt: str, model_name: str) -> str:
    try:
        response = await groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from {model_name}]: {e}"


def is_error_text(text: str) -> bool:
    return isinstance(text, str) and text.startswith("[Error")


async def run_ensemble(full_query: str):
    return await asyncio.gather(
        fetch_gemini(full_query),
        fetch_groq(full_query, "qwen/qwen3.6-27b"),      # replaces llama-3.3-70b-versatile (Groq retired it ~Aug 2026)
        fetch_groq(full_query, "openai/gpt-oss-120b"),   # replaces deepseek-r1-distill-llama-70b (decommissioned by Groq)
    )


def build_context_line() -> str:
    # None of these models know the real current date/time on their own — without
    # this, they'll guess based on training data (often wrong/"hallucinated").
    # Note: this does NOT solve queries that need live data (e.g. "weather today") —
    # that would require a real search tool (e.g. Gemini's Google Search grounding).
    now_utc = datetime.now(timezone.utc)
    return (
        f"[SYSTEM CONTEXT — for reference only, not part of the user's question]: "
        f"The current date/time is {now_utc.strftime('%A, %Y-%m-%d %H:%M')} UTC. "
        f"If the question involves a date/time in a specific location, convert it "
        f"yourself using that location's UTC offset (e.g. Tokyo/JST is UTC+9).\n\n"
    )


def read_attachments(files) -> tuple[str, list[str]]:
    """Read uploaded text-like files, truncated per file to stay within free-tier
    token budgets. Returns (block_to_append_to_prompt, list_of_filenames)."""
    block = ""
    names = []
    for f in files:
        try:
            content = f.read().decode("utf-8", errors="ignore")[:4000]
        except Exception:
            content = "(could not read this file as text)"
        names.append(f.name)
        block += f"\n\n[ATTACHED FILE: {f.name}]\n{content}"
    return block, names


def render_model_comparison(per_model: list[tuple[str, str]]):
    with st.expander(f"Compare {len(per_model)} individual model responses"):
        cols = st.columns(len(per_model))
        for col, (name, text) in zip(cols, per_model):
            with col:
                if is_error_text(text):
                    st.error(f"**{name}**")
                else:
                    st.info(f"**{name}**")
                st.write(text)


def build_leader_prompt(context_line: str, user_text: str, per_model: list[tuple[str, str]]) -> str:
    answers_block = "\n\n".join(
        f"[{name.upper()} ANSWER]:\n{text}" for name, text in per_model
    )
    return f"""
You are an AI lead analyst who synthesizes information from multiple sources.
{context_line}
Below is the user's question and answers from {len(per_model)} different AI models.
If an answer starts with "[Error ...]", that model failed to respond — ignore it, don't invent content in its place.
Use the [SYSTEM CONTEXT] above as ground truth — if an answer's date/time matches it, treat that answer as CORRECT; don't call it a "hallucination" just because it gave a specific figure.

[QUESTION]: {user_text}

{answers_block}

Your task:
1. Compare the valid answers: note what they agree on and any key disagreements/differences.
2. Assess the accuracy/reliability of any differences.
3. Synthesize one FINAL answer that is complete, accurate, and easy to understand for the user.
"""


async def get_final_answer(leader_prompt: str) -> str:
    final_summary = await fetch_gemini(leader_prompt)
    if is_error_text(final_summary):
        # Fallback leader if Gemini itself is down, so the user isn't left empty-handed.
        final_summary = await fetch_groq(leader_prompt, "openai/gpt-oss-120b")
    return final_summary


# ---------------------------------------------------------
# 3. STREAMLIT UI
# ---------------------------------------------------------
st.title("Multi-Model AI Assistant")
st.caption("Combines answers from several AI models into one cross-checked response.")

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"user": str, "per_model": [...], "synthesis": str}

# Render past turns
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        render_model_comparison(turn["per_model"])
        st.markdown(turn["synthesis"])

# Chat input: Enter (or the built-in send arrow) submits; the "+" icon (via
# accept_file) lets the user attach files, matching modern chat-app UIs.
prompt = st.chat_input(
    "Ask fcb anything...",
    accept_file="multiple",
    file_type=["txt", "md", "csv", "json", "py", "log"],
)

if prompt:
    user_text = prompt if isinstance(prompt, str) else prompt.text
    files = [] if isinstance(prompt, str) else prompt.files

    if not user_text and files:
        user_text = "Please analyze the attached file(s)."

    if user_text or files:
        attachment_block, attachment_names = read_attachments(files)
        display_user_text = user_text + (
            f"\n\n\U0001F4CE {', '.join(attachment_names)}" if attachment_names else ""
        )

        context_line = build_context_line()
        full_query = context_line + user_text + attachment_block

        with st.chat_message("user"):
            st.write(display_user_text)

        with st.chat_message("assistant"):
            with st.spinner("Querying 3 AI models in parallel..."):
                res_gemini, res_qwen, res_gptoss = asyncio.run(run_ensemble(full_query))

            per_model = [
                (MODEL_GEMINI_LABEL, res_gemini),
                (MODEL_QWEN_LABEL, res_qwen),
                (MODEL_GPTOSS_LABEL, res_gptoss),
            ]
            render_model_comparison(per_model)

            leader_prompt = build_leader_prompt(context_line, user_text, per_model)
            with st.spinner("Synthesizing final answer..."):
                final_summary = asyncio.run(get_final_answer(leader_prompt))

            st.markdown(final_summary)

        st.session_state.history.append({
            "user": display_user_text,
            "per_model": per_model,
            "synthesis": final_summary,
        })
