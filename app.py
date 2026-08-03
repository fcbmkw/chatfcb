import os
import uuid
import asyncio
import threading
import urllib.parse
from datetime import datetime, timezone
import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import errors as genai_errors
from openai import AsyncOpenAI

# ---------------------------------------------------------
# 1. API KEYS & PAGE CONFIG
# ---------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(
    page_title="Multi-Model AI Assistant",
    layout="wide",
)

# Inject viewport fix
components.html(
    """
    <script>
        var head = window.parent.document.getElementsByTagName('head')[0];
        if (!window.parent.document.querySelector('meta[name="viewport"]')) {
            var meta = document.createElement('meta');
            meta.name = "viewport";
            meta.content = "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no";
            head.appendChild(meta);
        }
    </script>
    """,
    height=0,
)

# Custom CSS for clean UI
hide_streamlit_chrome = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppViewerFooter {display: none;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }

    [class*="viewerBadge"] {display: none !important;}
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    [data-testid="stSidebar"] button {
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(120, 120, 120, 0.12) !important;
    }

    .block-container { padding-top: 2rem; }

    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    </style>
"""
st.markdown(hide_streamlit_chrome, unsafe_allow_html=True)

if not GEMINI_API_KEY or not GROQ_API_KEY:
    st.error("Missing API key. Please set GEMINI_API_KEY and GROQ_API_KEY.")
    st.stop()

GEMINI_MODEL_CANDIDATES = [
    "gemini-2.5-flash", 
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]

MODEL_GEMINI_LABEL = "Gemini"
MODEL_QWEN_LABEL = "Qwen 3.6 27B"
MODEL_GPTOSS_LABEL = "GPT-OSS 120B"

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY
)

# ---------------------------------------------------------
# 2. ASYNC API CALLS WITH PIPELINED SPEEDUP
# ---------------------------------------------------------
async def fetch_gemini(prompt: str, models: list = None, retries_per_model: int = 1) -> str:
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
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                break
            except Exception as e:
                last_err = e
                break
    return f"[Error from Gemini]: {last_err}"


async def fetch_groq(prompt: str, model_name: str) -> str:
    try:
        response = await groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Error from {model_name}]: {e}"


def is_error_text(text: str) -> bool:
    return isinstance(text, str) and text.startswith("[Error")


def build_leader_prompt(context_line: str, user_text: str, per_model: list[tuple[str, str]]) -> str:
    answers_block = "\n\n".join(
        f"--- [{name.upper()} RESPONSE] ---\n{text}" for name, text in per_model
    )
    return f"""
You are the Executive Lead AI Judge. Your task is to analyze answers from 3 sub-models and synthesize ONE definitive, highly accurate, and superior answer.

{context_line}

[USER QUESTION]:
{user_text}

[SUB-MODELS RESPONSES]:
{answers_block}

CRITICAL EVALUATION RULES:
1. Ignore any sub-model response starting with "[Error...]".
2. **Fact-Checking & Consensus**: Find common ground. If 2 models agree and 1 model hallucinates or disagrees without proof, ignore the outlier.
3. **Domain Expertise Priority**:
   - For Coding, Math, STEM, or Vietnamese fluency -> Value {MODEL_QWEN_LABEL}'s input highly.
   - For Complex Logic & Rule Following -> Value {MODEL_GPTOSS_LABEL}'s input highly.
   - For Overall Context & Structure -> Value {MODEL_GEMINI_LABEL}'s input highly.
4. **Final Output Format**:
   - Do NOT mention "Model A said...", "According to Qwen...", or "I synthesized this from...".
   - Answer directly to the user in a clean, clear, well-formatted Markdown response.
   - Make it complete, accurate, and better than any single model's individual answer.
"""


async def run_full_pipeline(full_query: str, user_text: str, context_line: str):
    """
    ⚡ SPEED OPTIMIZATION:
    Runs all 3 models in parallel, then IMMEDIATELY passes results to Gemini Leader.
    """
    # Step 1: Parallel Fetch
    results = await asyncio.gather(
        fetch_gemini(full_query),
        fetch_groq(full_query, "qwen/qwen3.6-27b"),
        fetch_groq(full_query, "openai/gpt-oss-120b"),
    )
    
    res_gemini, res_qwen, res_gptoss = results
    per_model = [
        (MODEL_GEMINI_LABEL, res_gemini),
        (MODEL_QWEN_LABEL, res_qwen),
        (MODEL_GPTOSS_LABEL, res_gptoss),
    ]

    # Step 2: Immediate Leader Synthesis
    leader_prompt = build_leader_prompt(context_line, user_text, per_model)
    final_synthesis = await fetch_gemini(leader_prompt)
    
    # Fallback to GPT-OSS if Gemini fails as leader
    if is_error_text(final_synthesis):
        final_synthesis = await fetch_groq(leader_prompt, "openai/gpt-oss-120b")
        
    return per_model, final_synthesis


def build_history_block(history: list, max_turns: int = 5, max_chars: int = 4000) -> str:
    if not history:
        return ""
    lines = []
    for turn in history[-max_turns:]:
        if turn.get("type") == "image":
            continue
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn.get('synthesis', '')}")
    if not lines:
        return ""
    recap = "\n".join(lines)
    if len(recap) > max_chars:
        recap = recap[-max_chars:]
    return (
        "[CONVERSATION HISTORY - FOR CONTEXT ONLY]:\n"
        f"{recap}\n\n"
        "[NEW USER QUESTION]:\n"
    )


def build_context_line() -> str:
    now_utc = datetime.now(timezone.utc)
    return f"[SYSTEM CONTEXT]: Current UTC Date/Time is {now_utc.strftime('%A, %Y-%m-%d %H:%M')}.\n"


def read_attachments(files) -> tuple[str, list[str]]:
    block = ""
    names = []
    for f in files:
        try:
            content = f.read().decode("utf-8", errors="ignore")[:4000]
        except Exception:
            content = "(could not read file)"
        names.append(f.name)
        block += f"\n\n[ATTACHED FILE: {f.name}]\n{content}"
    return block, names

# ---------------------------------------------------------
# 3. IMAGE GENERATION (Pollinations.ai)
# ---------------------------------------------------------
IMAGE_KEYWORDS = [
    "vẽ ảnh", "vẽ hình", "vẽ giúp", "vẽ cho", "vẽ một",
    "tạo ảnh", "tạo hình ảnh", "tạo hình", "hình ảnh của", "ảnh của",
    "generate image", "draw a picture", "create an image", "picture of"
]
IMAGE_LEADING_VERBS = ["draw", "paint", "sketch", "illustrate", "vẽ"]

def is_image_request(text: str) -> bool:
    if not text:
        return False
    lowered = text.lower().strip()
    first_word = lowered.split(None, 1)[0].strip(",.:!?") if lowered else ""
    if first_word in IMAGE_LEADING_VERBS:
        return True
    return any(kw in lowered for kw in IMAGE_KEYWORDS)


async def translate_prompt_to_english(raw_text: str) -> str:
    instruction = (
        "Translate/adapt this image request into ONE short English prompt for an AI image generator. "
        "Output ONLY the raw English prompt string, no quotes, no explanations.\n"
        f"User: {raw_text}"
    )
    result = await fetch_gemini(instruction)
    return raw_text.strip() if is_error_text(result) else result.strip().strip('"')


def build_pollinations_url(prompt_en: str, width: int = 1024, height: int = 1024) -> str:
    encoded = urllib.parse.quote(prompt_en)
    seed = int(datetime.now().timestamp())
    return f"https://image.pollinations.ai/prompt/{encoded}?width={width}&height={height}&nologo=true&seed={seed}&model=flux&enhance=false"


def render_turn_response(turn: dict, key: str):
    if turn.get("type") == "image":
        st.image(turn["image_url"], caption=f"Prompt: {turn['image_prompt_en']}")
    else:
        # High quality UX: Show final synthesized response first, collapsible sub-models below
        st.markdown(turn["synthesis"])
        
        per_model = turn.get("per_model", [])
        if per_model:
            with st.expander("🔍 Xem phản hồi gốc từ 3 mô hình AI (Click để mở)"):
                cols = st.columns(len(per_model))
                for col, (name, text) in zip(cols, per_model):
                    with col:
                        st.caption(f"**{name}**")
                        if is_error_text(text):
                            st.error(text)
                        else:
                            st.info(text[:300] + "..." if len(text) > 300 else text)

# ---------------------------------------------------------
# 4. BACKGROUND WORKER & STREAMLIT UI
# ---------------------------------------------------------
def _start_job(coro, **meta) -> dict:
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
        except Exception as e:
            job["error"] = e
        finally:
            job["done"] = True
            loop.close()

    threading.Thread(target=_runner, daemon=True).start()
    return job


def _cancel_job(job: dict):
    job["stop_event"].set()
    if job.get("loop") and job.get("task"):
        job["loop"].call_soon_threadsafe(job["task"].cancel)


def _new_conversation() -> str:
    conv_id = str(uuid.uuid4())
    st.session_state.conversations[conv_id] = {"title": "New chat", "history": []}
    st.session_state.current_id = conv_id
    return conv_id


def _maybe_set_title(conv: dict, user_text: str):
    if conv["title"] == "New chat" and user_text:
        conv["title"] = (user_text[:35] + "…") if len(user_text) > 35 else user_text


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "current_id" not in st.session_state:
    st.session_state.current_id = None
if not st.session_state.conversations:
    _new_conversation()
if st.session_state.current_id not in st.session_state.conversations:
    st.session_state.current_id = next(iter(st.session_state.conversations))

if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = True

# Top Header Bar
st.markdown(
    """
    <style>
    .st-key-hamburger_btn {
        position: fixed !important;
        top: 0.85rem !important;
        right: 0.9rem !important;
        z-index: 999999 !important;
        width: 40px !important;
    }
    .st-key-hamburger_btn button {
        width: 40px !important;
        height: 40px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(key="hamburger_btn"):
    if st.button("", key="toggle_sidebar_main", icon=":material/menu:", help="Menu", use_container_width=True):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
        st.rerun()

st.markdown(
    """
    <div style="line-height:1.2; padding-right:52px;">
        <div style="font-size:clamp(1.05rem, 4vw, 1.8rem); font-weight:700;">
            Multi-Model AI Assistant
        </div>
        <div style="font-size:0.85rem; opacity:0.65;">
            Sức mạnh tổng hợp từ Gemini + Qwen 3.6 27B + GPT-OSS 120B
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Logic
_desktop_width = "68px" if st.session_state.sidebar_collapsed else "230px"
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        min-width: {_desktop_width} !important;
        max-width: {_desktop_width} !important;
        width: {_desktop_width} !important;
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.session_state.sidebar_collapsed:
        if st.button("", key="expand_sidebar", icon=":material/dock_to_right:", use_container_width=True):
            st.session_state.sidebar_collapsed = False
            st.rerun()
        if st.button("", key="collapsed_new_chat", icon=":material/edit_square:", use_container_width=True):
            _new_conversation()
            st.rerun()
    else:
        head_col, collapse_col = st.columns([5, 1])
        with head_col:
            st.header("Chats")
        with collapse_col:
            if st.button("", key="collapse_sidebar", icon=":material/dock_to_left:"):
                st.session_state.sidebar_collapsed = True
                st.rerun()
        if st.button("New chat", key="new_chat_full", icon=":material/edit_square:", use_container_width=True):
            _new_conversation()
            st.rerun()
        st.divider()
        for conv_id, conv in reversed(list(st.session_state.conversations.items())):
            is_active = conv_id == st.session_state.current_id
            icon = ":material/forum:" if is_active else ":material/chat_bubble:"
            if st.button(conv["title"], key=f"conv_{conv_id}", icon=icon, use_container_width=True):
                st.session_state.current_id = conv_id
                st.rerun()

current_conv = st.session_state.conversations[st.session_state.current_id]

@st.fragment(run_every=0.3)
def _job_progress_fragment():
    job = st.session_state.job
    if job is None:
        return
    if job["done"]:
        st.rerun()
        return
    st.info("⚡ Đang truy vấn song song 3 AI & Gemini Leader đang tổng hợp...")
    if st.button("⏹ Stop", key="stop_button"):
        _cancel_job(job)


def _finalize_job(job: dict):
    if job["cancelled"]:
        st.session_state.job = None
        current_conv["history"].append({
            "type": "text",
            "user": job.get("display_user_text", job.get("user_text", "")),
            "per_model": [],
            "synthesis": "⏹ *Đã dừng bởi người dùng.*",
        })
        return

    if job["error"] is not None:
        st.session_state.job = None
        current_conv["history"].append({
            "type": "text",
            "user": job.get("display_user_text", job.get("user_text", "")),
            "per_model": [],
            "synthesis": f"⚠️ Lỗi: {job['error']}",
        })
        return

    if job["stage"] == "image":
        prompt_en = job["result"]
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

    if job["stage"] == "full_pipeline":
        per_model, final_synthesis = job["result"]
        st.session_state.job = None
        _maybe_set_title(current_conv, job["user_text"])
        current_conv["history"].append({
            "type": "text",
            "user": job["display_user_text"],
            "per_model": per_model,
            "synthesis": final_synthesis,
        })
        return


if "job" not in st.session_state:
    st.session_state.job = None

if st.session_state.job is not None and st.session_state.job["done"]:
    _finalize_job(st.session_state.job)

# Render Chat History
for _turn_idx, turn in enumerate(current_conv["history"]):
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        render_turn_response(turn, key=f"{st.session_state.current_id}_{_turn_idx}")

# In-flight Job Progress
if st.session_state.job is not None:
    job = st.session_state.job
    with st.chat_message("user"):
        st.write(job.get("display_user_text", job.get("user_text", "")))
    with st.chat_message("assistant"):
        _job_progress_fragment()

# Chat Input
prompt = st.chat_input(
    "Hỏi bất kỳ điều gì...",
    accept_file="multiple",
    file_type=["txt", "md", "csv", "json", "py", "log"],
    disabled=st.session_state.job is not None,
)

if prompt and st.session_state.job is None:
    user_text = prompt if isinstance(prompt, str) else prompt.text
    files = [] if isinstance(prompt, str) else prompt.files

    if not user_text and files:
        user_text = "Hãy phân tích (các) file đính kèm này."

    if user_text or files:
        if is_image_request(user_text):
            st.session_state.job = _start_job(
                translate_prompt_to_english(user_text),
                stage="image",
                user_text=user_text,
            )
            st.rerun()
        else:
            attachment_block, attachment_names = read_attachments(files)
            display_user_text = user_text + (
                f"\n\n📎 {', '.join(attachment_names)}" if attachment_names else ""
            )
            context_line = build_context_line()
            history_block = build_history_block(current_conv["history"])
            full_query = context_line + history_block + user_text + attachment_block

            st.session_state.job = _start_job(
                run_full_pipeline(full_query, user_text, context_line),
                stage="full_pipeline",
                user_text=user_text,
                display_user_text=display_user_text,
            )
            st.rerun()
