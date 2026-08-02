import os
import uuid
import asyncio
import threading
import urllib.parse
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

# st.set_page_config() PHẢI là lệnh Streamlit đầu tiên trong script (trước
# mọi st.markdown/st.write/...), nếu không Streamlit sẽ báo lỗi
# StreamlitAPIException ngay khi chạy.
st.set_page_config(
    page_title="Multi-Model AI Assistant",
    layout="wide",
    # NOTE: intentionally NOT setting initial_sidebar_state="collapsed".
    # Its reopen affordance depends on Streamlit version / how the app is
    # embedded (e.g. Streamlit Cloud iframes), and was unreliable in
    # testing. Default "auto" always gives a visible, working sidebar
    # toggle, at the cost of the sidebar being open on first load.
)

# CSS to hide Streamlit's default hamburger menu and "Deploy" button /
# footer watermark. IMPORTANT: target these specific elements only —
# do NOT hide the whole `header` or `[data-testid="stToolbar"]` container,
# because the sidebar's re-open arrow (data-testid="stExpandSidebarButton")
# lives inside that same toolbar. Hiding the whole toolbar hides that arrow
# too, making a collapsed sidebar impossible to reopen.
hide_streamlit_chrome = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppViewerFooter {display: none;}

    /* Toolbar bên trong header — chứa menu 3 chấm, nút Deploy, icon
    GitHub (Octocat), status widget "Running"... Ẩn đúng cụm này thay vì
    ẩn cả <header> để KHÔNG đụng tới nút mở/đóng sidebar (nút đó là một
    phần tử khác nằm cạnh stToolbar, không nằm bên trong nó), nên New
    Chat / History vẫn đóng-mở lại bình thường sau khi bấm "<<". */
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    /* "Manage app" / "Hosted with Streamlit" badge (Community Cloud) */
    [class*="viewerBadge"] {display: none !important;}

    /* Thu gọn khoảng trắng thừa phía trên */
    .block-container { padding-top: 2rem; }
    </style>
"""
st.markdown(hide_streamlit_chrome, unsafe_allow_html=True)

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


# ---------------------------------------------------------
# 2b. IMAGE GENERATION (Pollinations.ai — free, no API key needed)
# ---------------------------------------------------------
# Keywords (Vietnamese + English) used to detect "the user wants an image,
# not a text answer". Keep this simple/explicit rather than asking a model
# to classify every message — cheaper and predictable.
IMAGE_KEYWORDS = [
    "vẽ ảnh", "vẽ hình", "vẽ giúp", "vẽ cho", "vẽ một",
    "tạo ảnh", "tạo hình ảnh", "tạo hình", "hình ảnh của", "ảnh của",
    "generate image", "generate a picture", "generate picture",
    "draw a picture", "draw an image", "draw me",
    "create an image", "create a picture", "make an image", "make a picture",
    "picture of", "image of", "photo of",
    "/image", "/img",
]

# If the message *starts* with one of these verbs, treat it as an image
# request even without a following keyword phrase — this is what catches
# short commands like "Draw Leo Messi in Tokyo" or "Vẽ Messi ở Tokyo".
# Trade-off: a message like "Draw conclusions from this data" would also be
# (mis)classified as an image request. Let us know if that becomes an issue
# and we can switch to an explicit toggle instead of keyword-guessing.
IMAGE_LEADING_VERBS = ["draw", "paint", "sketch", "illustrate", "vẽ"]


def is_image_request(text: str) -> bool:
    """True if the message looks like an image-generation request."""
    if not text:
        return False
    lowered = text.lower().strip()
    first_word = lowered.split(None, 1)[0].strip(",.:!?") if lowered else ""
    if first_word in IMAGE_LEADING_VERBS:
        return True
    return any(kw in lowered for kw in IMAGE_KEYWORDS)


async def translate_prompt_to_english(raw_text: str) -> str:
    """Turn the user's (possibly Vietnamese) request into a short English
    image-generation prompt, using Gemini. Falls back to the raw text if
    Gemini is unavailable (Pollinations still works with non-English text,
    just less reliably)."""
    instruction = (
        "You are a prompt engineer for a text-to-image model. Read the "
        "following user request (it may be in Vietnamese or English) and "
        "write ONE short, vivid English image-generation prompt describing "
        "the scene it asks for. Reply with ONLY the English prompt itself — "
        "no quotes, no explanation, no leading phrase like 'Sure' or 'Here'.\n\n"
        f"User request: {raw_text}"
    )
    result = await fetch_gemini(instruction)
    if is_error_text(result):
        return raw_text.strip()
    return result.strip().strip('"').strip("'")


def build_pollinations_url(prompt_en: str, width: int = 1024, height: int = 1024) -> str:
    """Build a direct image URL from Pollinations.ai's image endpoint.
    NOTE 1: the working endpoint is https://image.pollinations.ai/prompt/... —
    the bare 'pollinations.ai/prompt/...' form does not serve the image
    directly, so we use the correct 'image.' subdomain here.
    NOTE 2: model=flux is set explicitly (Pollinations' free, unrestricted
    model) and enhance=false disables Pollinations' own prompt-rewriting
    step. Without enhance=false, Pollinations can silently rewrite the
    prompt server-side (e.g. when it flags a real person's name), which is
    why the returned image can end up completely unrelated to what was
    asked for. Even with this fix, free open models like Flux were not
    fine-tuned on specific real people, so the likeness of named public
    figures (e.g. a football player) may still be inaccurate — that's a
    model-capability limit, not a bug in this code."""
    encoded = urllib.parse.quote(prompt_en)
    # Random seed so re-running the same prompt doesn't just return a cached
    # identical image.
    seed = int(datetime.now().timestamp())
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width={width}&height={height}&nologo=true&seed={seed}"
        f"&model=flux&enhance=false"
    )


def render_model_comparison(per_model: list[tuple[str, str]]):
    if not per_model:
        return
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

# ---------------------------------------------------------
# 3a. MULTI-CHAT SESSIONS (New Chat + Chat History, sidebar)
# ---------------------------------------------------------
# Kept in st.session_state, so it lives for the browser tab/session only
# (resets on server restart) — same trade-off as the original single
# `history` list, just split into multiple named conversations.

def _new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "New chat", "history": []}
    st.session_state.current_id = conv_id
    return conv_id


def _maybe_set_title(conv: dict, user_text: str):
    """First message in a conversation becomes its title in the sidebar."""
    if conv["title"] == "New chat" and user_text:
        conv["title"] = (user_text[:40] + "…") if len(user_text) > 40 else user_text


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_id" not in st.session_state:
    st.session_state.current_id = None
if not st.session_state.conversations:
    _new_conversation()
if st.session_state.current_id not in st.session_state.conversations:
    st.session_state.current_id = next(iter(st.session_state.conversations))

with st.sidebar:
    st.header("💬 Chats")
    if st.button("➕ New Chat", use_container_width=True):
        _new_conversation()
        st.rerun()
    st.divider()
    st.caption("History")
    # Newest conversation on top
    for conv_id, conv in reversed(list(st.session_state.conversations.items())):
        is_active = conv_id == st.session_state.current_id
        label = ("🟢 " if is_active else "⚪ ") + conv["title"]
        if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
            st.session_state.current_id = conv_id
            st.rerun()
    st.divider()
    st.caption("🎨 Type \"draw ...\" / \"generate image ...\" to create an image (Pollinations.ai).")

current_conv = st.session_state.conversations[st.session_state.current_id]

# ---------------------------------------------------------
# 3b. BACKGROUND JOBS (so a "Stop" button can actually cancel a call)
# ---------------------------------------------------------
# Streamlit normally blocks the whole UI thread while `asyncio.run(...)` is
# in flight, so a Stop button rendered next to it can never be clicked in
# time. To make Stop actually work, the AI calls run on a background thread
# (with its own event loop) while the main thread stays free to render a
# live "Stop" button inside an auto-refreshing fragment. Clicking Stop
# cancels the underlying asyncio task, which aborts the in-flight request.

def _start_job(coro, **meta) -> dict:
    """Launch `coro` on a background thread. Returns a job dict that the
    polling fragment below reads from and that the Stop button cancels."""
    job = {
        "stop_event": threading.Event(),
        "loop": None,
        "task": None,
        "done": False,
        "cancelled": False,
        "error": None,
        "result": None,
        **meta,
    }

    def _runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        job["loop"] = loop
        job["task"] = loop.create_task(coro)
        try:
            job["result"] = loop.run_until_complete(job["task"])
        except asyncio.CancelledError:
            job["cancelled"] = True
        except Exception as e:  # surfaced to the user instead of crashing the app
            job["error"] = e
        finally:
            job["done"] = True
            loop.close()

    threading.Thread(target=_runner, daemon=True).start()
    return job


def _cancel_job(job: dict):
    job["stop_event"].set()
    if job.get("loop") is not None and job.get("task") is not None:
        job["loop"].call_soon_threadsafe(job["task"].cancel)


_STAGE_LABELS = {
    "ensemble": "Querying 3 AI models in parallel...",
    "synthesis": "Synthesizing final answer...",
    "image": "Translating prompt to English...",
}


@st.fragment(run_every=0.3)
def _job_progress_fragment():
    """Polls the active job every 0.3s and shows a live Stop button.
    Runs as its own fragment so clicking Stop doesn't need to wait for a
    full-page rerun."""
    job = st.session_state.job
    if job is None:
        return
    if job["done"]:
        st.rerun()  # hand off to the main script to finalize / chain the job
        return
    st.info(f"⏳ {_STAGE_LABELS.get(job['stage'], 'Working...')}")
    if st.button("⏹ Stop", key="stop_button"):
        _cancel_job(job)


def _finalize_job(job: dict):
    """Called once a job's `done` flag is set. Either appends the finished
    turn to history, chains to the next stage (ensemble -> synthesis), or
    records that the user stopped it."""
    if job["cancelled"]:
        st.session_state.job = None
        current_conv["history"].append({
            "type": "text",
            "user": job.get("display_user_text", job.get("user_text", "")),
            "per_model": job.get("per_model", []),
            "synthesis": "⏹ *Stopped by user.*",
        })
        return

    if job["error"] is not None:
        st.session_state.job = None
        current_conv["history"].append({
            "type": "text",
            "user": job.get("display_user_text", job.get("user_text", "")),
            "per_model": job.get("per_model", []),
            "synthesis": f"⚠️ Error: {job['error']}",
        })
        return

    if job["stage"] == "image":
        prompt_en = job["result"] if not is_error_text(job["result"]) else job["user_text"].strip()
        img_url = build_pollinations_url(prompt_en)
        _maybe_set_title(current_conv, job["user_text"])
        current_conv["history"].append({
            "type": "image",
            "user": job["user_text"],
            "image_prompt_en": prompt_en,
            "image_url": img_url,
        })
        st.session_state.job = None
        return

    if job["stage"] == "ensemble":
        res_gemini, res_qwen, res_gptoss = job["result"]
        per_model = [
            (MODEL_GEMINI_LABEL, res_gemini),
            (MODEL_QWEN_LABEL, res_qwen),
            (MODEL_GPTOSS_LABEL, res_gptoss),
        ]
        leader_prompt = build_leader_prompt(job["context_line"], job["user_text"], per_model)
        # Chain straight into stage 2 (still cancelable via a fresh Stop button)
        st.session_state.job = _start_job(
            get_final_answer(leader_prompt),
            stage="synthesis",
            user_text=job["user_text"],
            display_user_text=job["display_user_text"],
            per_model=per_model,
        )
        return

    if job["stage"] == "synthesis":
        st.session_state.job = None
        _maybe_set_title(current_conv, job["user_text"])
        current_conv["history"].append({
            "type": "text",
            "user": job["display_user_text"],
            "per_model": job["per_model"],
            "synthesis": job["result"],
        })
        return


if "job" not in st.session_state:
    st.session_state.job = None

# A job that finished since the last poll gets resolved before we render
# history, so it shows up as a normal completed turn below.
if st.session_state.job is not None and st.session_state.job["done"]:
    _finalize_job(st.session_state.job)

# Render past turns of the active conversation
for turn in current_conv["history"]:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("type") == "image":
            st.image(turn["image_url"], caption=f"Prompt: {turn['image_prompt_en']}")
        else:
            render_model_comparison(turn["per_model"])
            st.markdown(turn["synthesis"])

# Render the in-flight turn (if any) with its live Stop button
if st.session_state.job is not None:
    job = st.session_state.job
    with st.chat_message("user"):
        st.write(job.get("display_user_text", job.get("user_text", "")))
    with st.chat_message("assistant"):
        _job_progress_fragment()

# Chat input: Enter (or the built-in send arrow) submits; the "+" icon (via
# accept_file) lets the user attach files, matching modern chat-app UIs.
prompt = st.chat_input(
    "Ask fcb anything... (or type 'draw ...' to generate an image)",
    accept_file="multiple",
    file_type=["txt", "md", "csv", "json", "py", "log"],
    disabled=st.session_state.job is not None,
)

if prompt and st.session_state.job is None:
    user_text = prompt if isinstance(prompt, str) else prompt.text
    files = [] if isinstance(prompt, str) else prompt.files

    if not user_text and files:
        user_text = "Please analyze the attached file(s)."

    if user_text or files:
        # ---------------- IMAGE GENERATION BRANCH ----------------
        if is_image_request(user_text):
            st.session_state.job = _start_job(
                translate_prompt_to_english(user_text),
                stage="image",
                user_text=user_text,
            )
            st.rerun()

        # ---------------- NORMAL TEXT / ENSEMBLE BRANCH ----------------
        else:
            attachment_block, attachment_names = read_attachments(files)
            display_user_text = user_text + (
                f"\n\n\U0001F4CE {', '.join(attachment_names)}" if attachment_names else ""
            )
            context_line = build_context_line()
            full_query = context_line + user_text + attachment_block

            st.session_state.job = _start_job(
                run_ensemble(full_query),
                stage="ensemble",
                user_text=user_text,
                display_user_text=display_user_text,
                context_line=context_line,
            )
            st.rerun()
