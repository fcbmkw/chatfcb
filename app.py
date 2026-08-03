import os
import uuid
import asyncio
import threading
import urllib.parse
import html
from datetime import datetime, timezone
import streamlit as st
import streamlit.components.v1 as components
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

# --- FIX #1: viewport meta tag -----------------------------------------
# st.markdown(..., unsafe_allow_html=True) chỉ inject HTML vào giữa <body>,
# KHÔNG BAO GIỜ chạm được vào <head> — vì vậy nếu chỉ viết
# st.markdown('<meta name="viewport" ...>') thì trình duyệt không đọc thẻ
# đó (nó không nằm trong <head>). Safari trên iPhone, thiếu đúng
# <meta name="viewport"> trong <head>, sẽ tự coi trang là rộng ~980px
# (layout "desktop giả lập") rồi thu nhỏ lại vừa màn hình — nên
# `window.innerWidth`/`@media (max-width: 640px)` không bao giờ true dù
# màn hình vật lý chỉ ~390px. Đây là lý do sidebar đổi size trên máy tính
# (đã có viewport mặc định đúng của trình duyệt desktop) nhưng không đổi
# trên iPhone.
# Cách duy nhất để chèn được vào đúng <head> là qua components.html: nó
# render trong 1 iframe cùng-origin (srcdoc), và script bên trong dùng
# `window.parent.document` để thao tác lên <head> của trang cha (trang
# Streamlit thật), rồi tự huỷ iframe (height=0) sau khi chạy xong.
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

# CSS ẩn menu 3 chấm / nút Deploy / icon GitHub / badge "Manage app"...
# Nút mở/đóng sidebar gốc của Streamlit nằm trong stToolbar nên cũng bị ẩn
# theo — không sao, vì New Chat/History giờ dùng nút "«"/"»" tự viết
# riêng (xem phần MULTI-CHAT SESSIONS bên dưới), không phụ thuộc nút gốc
# của Streamlit nữa.
hide_streamlit_chrome = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppViewerFooter {display: none;}

    /* Toolbar bên trong header — chứa menu 3 chấm, nút Deploy, icon
    GitHub (Octocat), status widget "Running"... */
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stStatusWidget"] {visibility: hidden;}
    [data-testid="stDecoration"] {visibility: hidden;}

    /* FIX: thanh <header> gốc của Streamlit (chứa các mục ở trên) trước
    giờ chỉ bị ẩn TỪNG PHẦN bên trong (visibility:hidden cho từng icon),
    nhưng bản thân cái <header> bao ngoài vẫn còn nguyên đó — trong suốt,
    nhưng vẫn chiếm đúng dải trên cùng màn hình với z-index rất cao. Vô
    hại khi nút hamburger còn nằm trong luồng bố cục bình thường (bên
    dưới dải này), nhưng từ khi đổi nút hamburger sang position:fixed
    (đặt ở top: 0.85rem — nằm ngay trong dải header gốc), header trong
    suốt đó đè lên trên và chặn mất click/hiển thị của nút. Ẩn hẳn luôn
    cả khối header gốc (không chỉ từng phần bên trong) để giải phóng
    hoàn toàn dải trên cùng cho nút hamburger tự vẽ của mình.
    height: 0 thay vì display:none để tránh Streamlit tính lại layout
    bị giật/nhảy khi header biến mất đột ngột. */
    header[data-testid="stHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }

    /* "Manage app" / "Hosted with Streamlit" badge (Community Cloud) */
    [class*="viewerBadge"] {display: none !important;}

    /* Nút "<<" gốc của Streamlit để đóng/mở sidebar — nằm bên TRONG
    sidebar, không nằm trong stToolbar nên chưa bị ẩn ở trên. Giờ dùng
    nút "«"/"»" tự viết rồi nên ẩn hẳn cái gốc để khỏi có 2 nút chồng
    nhau (cái gốc bấm không còn tác dụng gì vì width đã bị CSS ép). */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    /* Bo góc + hiệu ứng hover nhẹ cho các nút trong sidebar, cho gọn
    gàng/hiện đại hơn kiểu mặc định của Streamlit. */
    [data-testid="stSidebar"] button {
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: rgba(120, 120, 120, 0.12) !important;
    }

    /* Thu gọn khoảng trắng thừa phía trên */
    .block-container { padding-top: 2rem; }

    /* FIX (bảo hiểm chung): khoá cứng chiều ngang = đúng viewport trên
    mọi kích thước màn hình. Nếu có phần tử nào khác (không phải sidebar,
    không phải header) lỡ tràn ra ngoài do padding/margin cộng dồn, dòng
    này chặn nó sinh ra thanh cuộn ngang / đẩy nội dung dịch sang phải,
    thay vì phải sửa từng chỗ một. Không ảnh hưởng cuộn dọc. */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }

    /* FIX: làm nổi bật expander "Compare N individual model responses" —
    trước đó nó chỉ là 1 dòng chữ thường, rất dễ bị lướt qua/không nhận ra
    là có thể bấm vào xem 3 câu trả lời gốc. Thêm viền màu + nền nhạt để
    mắt người dùng dừng lại ở đây.
    LƯU Ý: container key giờ là động (compare_wrap_<turn_key>, để tránh
    trùng key giữa các turn) nên class thật là "st-key-compare_wrap_xxx" —
    selector class CHÍNH XÁC ".st-key-compare_wrap" không còn khớp nữa
    (đó là lý do highlight tự nhiên "biến mất" dù code CSS vẫn còn).
    Đổi sang attribute-selector kiểu "chứa chuỗi" để khớp mọi suffix. */
    [class*="st-key-compare_wrap"] [data-testid="stExpander"] {
        border: 1px solid rgba(255, 149, 0, 0.55) !important;
        background: rgba(255, 149, 0, 0.06) !important;
        border-radius: 10px !important;
    }
    [class*="st-key-compare_wrap"] [data-testid="stExpander"] summary {
        font-weight: 600 !important;
    }

    /* Khung hiển thị 3 câu trả lời gốc bên trong expander "Compare..." đã
    thu gọn (KHÔNG áp dụng lúc đang stream — lúc đó hiện full độ dài).
    Yêu cầu: chiều cao 3 khung bằng nhau, khớp với khung của kết quả NGẮN
    NHẤT, 2 khung còn lại cuộn lên xuống để xem hết. CSS thuần không thể
    "đo" chiều cao thật đã render (Streamlit không trả pixel thật của DOM
    về Python, cần JS đo runtime mới chính xác tuyệt đối) — nên chiều cao
    ở đây được ƯỚC LƯỢNG từ số ký tự của câu trả lời ngắn nhất (xem hàm
    `_estimate_box_height_px`) rồi gán trực tiếp qua `style="height:...px"`
    ngay tại nơi gọi (không cố định ở đây nữa, để không bị "ngắn quá,
    không khớp kết quả nào" như trước). */
    .stream-box {
        overflow-y: auto;
        border: 1px solid rgba(150, 150, 150, 0.35);
        border-radius: 8px;
        padding: 8px 10px;
        font-size: 0.82rem;
        line-height: 1.35;
        white-space: pre-wrap;
        word-break: break-word;
    }
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
#
# STREAMING: mỗi fetch_* nhận thêm `on_chunk` (callback, optional) — mỗi
# khi có thêm 1 đoạn chữ mới từ model, gọi on_chunk(toàn_bộ_chữ_đã_có_đến_
# giờ). Bên gọi (run_ensemble / get_final_answer) truyền vào 1 lambda ghi
# thẳng vào 1 dict dùng chung với `_job_progress_fragment` — nhờ vậy UI có
# thể vẽ lại chữ đang "gõ dần" mỗi 0.3s thay vì màn hình trống đợi đủ câu
# trả lời mới hiện. Không thay đổi tổng thời gian model trả lời xong,
# nhưng người dùng thấy có tiến triển ngay từ giây đầu, đỡ cảm giác "đơ".
# ---------------------------------------------------------
async def fetch_gemini(prompt: str, models: list = None, retries_per_model: int = 1, on_chunk=None) -> str:
    """Call Gemini, trying each model in `models` in order.
    - 404 (model no longer exists / not available) -> move to the next model immediately.
    - 429 (rate limit) -> wait and retry the current model a couple of times first.
    Uses the SDK's native async streaming client (`client.aio`) so chunks can be
    reported via `on_chunk` as they arrive, instead of blocking until the full
    response is done."""
    models = models or GEMINI_MODEL_CANDIDATES
    last_err = None
    for model in models:
        for attempt in range(retries_per_model + 1):
            try:
                text = ""
                stream = await gemini_client.aio.models.generate_content_stream(
                    model=model, contents=prompt
                )
                async for chunk in stream:
                    piece = getattr(chunk, "text", None) or ""
                    if piece:
                        text += piece
                        if on_chunk:
                            on_chunk(text)
                return text
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


async def fetch_groq(prompt: str, model_name: str, on_chunk=None) -> str:
    try:
        text = ""
        stream = await groq_client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            piece = chunk.choices[0].delta.content or ""
            if piece:
                text += piece
                if on_chunk:
                    on_chunk(text)
        return text
    except Exception as e:
        return f"[Error from {model_name}]: {e}"


def is_error_text(text: str) -> bool:
    return isinstance(text, str) and text.startswith("[Error")


async def run_ensemble(full_query: str, model_partials: dict = None):
    mp = model_partials if model_partials is not None else {}
    return await asyncio.gather(
        fetch_gemini(full_query, on_chunk=lambda t: mp.__setitem__("gemini", t)),
        fetch_groq(full_query, "qwen/qwen3.6-27b", on_chunk=lambda t: mp.__setitem__("qwen", t)),      # replaces llama-3.3-70b-versatile (Groq retired it ~Aug 2026)
        fetch_groq(full_query, "openai/gpt-oss-120b", on_chunk=lambda t: mp.__setitem__("gptoss", t)),  # replaces deepseek-r1-distill-llama-70b (decommissioned by Groq)
    )



def build_history_block(history: list, max_turns: int = 6, max_chars: int = 6000) -> str:
    """Format the last `max_turns` completed turns of the CURRENT conversation
    into a plain-text recap prepended to the next prompt.
    Without this, fetch_gemini/fetch_groq are called with only `user_text` —
    a single isolated sentence — every single time, so a follow-up like "Co
    link youtube nao khong?" arrives with zero memory of the Milky Way
    question just asked. This is not a bug in the API calls themselves, it's
    that nothing in `full_query` ever carried prior turns.
    Only the Leader's synthesized answer is included per turn (not all 3
    individual model answers) to keep the recap compact and cheap. Image
    turns are skipped (no useful text answer to recap). If the recap still
    exceeds `max_chars`, it's trimmed from the start so the MOST RECENT turns
    (usually most relevant to a follow-up) are kept, not the oldest."""
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
        "[CONVERSATION SO FAR — for context only; the user cannot see this "
        "block again, so don't quote it back verbatim]:\n"
        f"{recap}\n\n"
        "[NEW MESSAGE FROM THE USER — if it's a follow-up (e.g. \"any link "
        "for that?\", \"explain more\"), resolve it using the conversation "
        "above; otherwise treat it as a fresh question]:\n"
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


def _estimate_box_height_px(texts: list[str]) -> int:
    """Ước lượng chiều cao (px) vừa khít với câu trả lời NGẮN NHẤT trong
    `texts`, để 3 khung cao bằng nhau và khớp với kết quả ngắn nhất (câu
    dài hơn sẽ tự cuộn — xử lý bằng CSS overflow-y:auto ở nơi gọi).
    Đây là ước lượng dựa trên SỐ KÝ TỰ (~44 ký tự/dòng, ~19px/dòng khớp
    với font-size 0.82rem + line-height 1.35 đang dùng cho .stream-box),
    KHÔNG phải đo pixel thật đã render — Streamlit không trả kích thước
    DOM thật về phía Python để đo chính xác tuyệt đối (cần thêm 1 custom
    JS component mới làm được), nên đây là cách gần đúng, đủ dùng cho
    mục đích "3 khung cao đều & khớp cỡ kết quả ngắn nhất" mà không cần
    thêm component riêng. Có chặn min/max để không bao giờ quá thấp
    (không đọc được) hay quá cao (mất tác dụng thu gọn)."""
    if not texts:
        return 160
    shortest = min(texts, key=len)
    chars_per_line = 44
    lines_by_length = -(-len(shortest) // chars_per_line)  # ceil
    lines_by_breaks = shortest.count("\n") + 1
    lines = max(1, lines_by_length, lines_by_breaks)
    height = lines * 19 + 24  # 19px/dòng + đệm trên dưới
    return max(90, min(height, 420))


def render_model_comparison(per_model: list[tuple[str, str]], key: str):
    if not per_model:
        return
    box_h = _estimate_box_height_px([text for _, text in per_model])
    with st.container(key=f"compare_wrap_{key}"):
        with st.expander(f"🔍 **Compare {len(per_model)} individual model responses** — click to see what each model said"):
            cols = st.columns(len(per_model))
            for col, (name, text) in zip(cols, per_model):
                with col:
                    if is_error_text(text):
                        st.error(f"**{name}**")
                    else:
                        st.info(f"**{name}**")
                    # Khung cuộn chỉ áp dụng ở ĐÂY (khi người dùng bấm mở
                    # expander xem lại 3 câu trả lời đã xong) — lúc đang
                    # stream thì hiện full độ dài, không giới hạn chiều
                    # cao. Chiều cao 3 khung khớp với câu trả lời NGẮN
                    # NHẤT (box_h, ước lượng ở trên); câu dài hơn sẽ tự
                    # cuộn được nhờ overflow-y:auto trong CSS .stream-box.
                    st.markdown(
                        f'<div class="stream-box" style="height:{box_h}px;">{html.escape(text)}</div>',
                        unsafe_allow_html=True,
                    )


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

Silently reconcile the valid answers — resolve any disagreements in favor of whichever is more accurate/reliable — and reply with ONLY the final answer: complete, accurate, and easy to understand for the user. Do NOT include a comparison/analysis section, do NOT mention "Model A said X, Model B said Y", do NOT explain your reconciliation process. Go straight to the answer itself, as if you were answering the question yourself.
"""


async def get_final_answer(leader_prompt: str, on_chunk=None) -> str:
    final_summary = await fetch_groq(leader_prompt, "openai/gpt-oss-120b", on_chunk=on_chunk)
    if is_error_text(final_summary):
        # Fallback leader if Groq itself is down, so the user isn't left empty-handed.
        # Reset the streamed-so-far text first — otherwise the UI would show Groq's
        # half-written (broken) draft glued in front of Gemini's fresh restart.
        if on_chunk:
            on_chunk("")
        final_summary = await fetch_gemini(leader_prompt, on_chunk=on_chunk)
    return final_summary


# ---------------------------------------------------------
# 3. STREAMLIT UI
# ---------------------------------------------------------
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

# Chế độ hiển thị của sidebar: tự quản lý bằng session_state, KHÔNG dùng
# cơ chế collapse/expand có sẵn của Streamlit nữa (tên nút đó đổi khác
# nhau tuỳ phiên bản, dễ bị kẹt-đóng như đã gặp). Bề rộng sidebar được
# set động qua CSS theo giá trị này, nên toggle luôn hoạt động chắc chắn.
# Mặc định GỌN (chỉ icon), theo yêu cầu.
if "sidebar_collapsed" not in st.session_state:
    st.session_state.sidebar_collapsed = True

# ---------------------------------------------------------
# 3. STREAMLIT UI — tiêu đề (compact, tự co chữ trên màn hình hẹp) +
# nút hamburger để mở/đóng sidebar.
#
# LƯU Ý: bản trước dùng st.columns([10, 1]) để đặt tiêu đề + nút hamburger
# trên cùng 1 hàng, rồi cố ép chiều rộng bằng CSS (flex-basis/padding/margin
# của [data-testid="column"]/[data-testid="stHorizontalBlock"]). Cách đó
# vẫn bị tràn/cắt nút trên màn hình hẹp (báo lại vẫn thấy nút bị che hơn
# 1 nửa) vì độ rộng thật của hàng phụ thuộc vào nhiều lớp CSS mặc định
# khác của Streamlit (min-width của block-container, gap giữa cột, v.v.)
# mà mỗi bản Streamlit có thể tính khác nhau — rất dễ vỡ lại bất cứ lúc
# nào Streamlit đổi cấu trúc DOM.
#
# Đổi sang cách chắc chắn hơn: nút hamburger KHÔNG còn nằm trong cột nào
# cả, mà dùng `position: fixed` để tự neo cứng vào góc trên-phải màn
# hình, hoàn toàn tách khỏi mọi phép tính chiều rộng cột/hàng ở trên. Vì
# vậy dù hàng chứa tiêu đề có tràn/co giãn thế nào, nút vẫn luôn nằm
# đúng 1 vị trí cố định so với viewport, không bao giờ bị đẩy ra ngoài
# mép màn hình nữa. Tiêu đề chỉ cần chừa khoảng trống bên phải (padding-
# right) để chữ không bị nút đè lên.
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .st-key-hamburger_btn {
        position: fixed !important;
        top: 0.85rem !important;
        right: 0.9rem !important;
        z-index: 999999 !important;   /* cực cao để không lớp nào (kể cả header gốc) đè lên được nữa */
        width: 40px !important;
    }
    .st-key-hamburger_btn button {
        width: 40px !important;
        height: 40px !important;
    }
    /* Nút "New chat" ngay bên trái nút hamburger — cùng kỹ thuật
    position:fixed để không bao giờ bị lệch/che dù màn hình rộng hẹp
    thế nào, giống hệt lý do đã đổi nút hamburger ở trên. */
    .st-key-newchat_btn_main {
        position: fixed !important;
        top: 0.85rem !important;
        right: 3.05rem !important;   /* 0.9rem (lề nút hamburger) + 40px (rộng nút) + 0.35rem (khoảng cách) */
        z-index: 999999 !important;
        width: 40px !important;
    }
    .st-key-newchat_btn_main button {
        width: 40px !important;
        height: 40px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.container(key="newchat_btn_main"):
    if st.button("", key="new_chat_main", icon=":material/edit_square:", help="New chat", use_container_width=True):
        _new_conversation()
        st.rerun()
with st.container(key="hamburger_btn"):
    if st.button("", key="toggle_sidebar_main", icon=":material/menu:", help="Menu", use_container_width=True):
        st.session_state.sidebar_collapsed = not st.session_state.sidebar_collapsed
        st.rerun()

st.markdown(
    """
    <div style="line-height:1.2; padding-right:98px;">
        <div style="font-size:clamp(1.05rem, 4vw, 1.8rem); font-weight:700;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            Multi-Model AI Assistant
        </div>
        <div style="font-size:0.85rem; opacity:0.65;
                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
            Combines answers from several AI models.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Bề rộng sidebar khi mở rộng — thu nhỏ lại theo yêu cầu.
_desktop_width = "68px" if st.session_state.sidebar_collapsed else "230px"
# Trên mobile: nếu gọn -> ẩn hẳn (0px, không còn dải icon nào che chatbox
# nữa); nếu mở -> hiện như một lớp overlay (position:fixed) đè lên trên
# nội dung chính thay vì đẩy nội dung sang một bên, giống ngăn kéo
# (drawer) thường thấy trên app di động.
if st.session_state.sidebar_collapsed:
    _mobile_sidebar_css = """
        [data-testid="stSidebar"] {
            min-width: 0px !important;
            max-width: 0px !important;
            width: 0px !important;
            overflow: hidden !important;
        }
    """
else:
    _mobile_sidebar_css = """
        [data-testid="stSidebar"] {
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            height: 100vh !important;
            min-width: 68vw !important;
            max-width: 68vw !important;
            width: 68vw !important;
            z-index: 999 !important;
            box-shadow: 2px 0 16px rgba(0, 0, 0, 0.3) !important;
        }
    """
st.markdown(f"""
    <style>
    [data-testid="stSidebar"] {{
        min-width: {_desktop_width} !important;
        max-width: {_desktop_width} !important;
        width: {_desktop_width} !important;
        transform: none !important;
        visibility: visible !important;
    }}
    @media (max-width: 640px) {{
        {_mobile_sidebar_css}
    }}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    if st.session_state.sidebar_collapsed:
        # ---------------- CHẾ ĐỘ GỌN: chỉ icon (desktop) ----------------
        # Trên mobile sidebar đang bị ẩn 0px nên các nút này không hiện
        # ra được — người dùng mobile mở sidebar qua nút hamburger phía
        # trên thay vì nút "»" này.
        if st.button("", key="expand_sidebar", help="Expand", icon=":material/dock_to_right:", use_container_width=True):
            st.session_state.sidebar_collapsed = False
            st.rerun()
        if st.button("", key="collapsed_new_chat", help="New chat", icon=":material/edit_square:", use_container_width=True):
            _new_conversation()
            st.rerun()
        if st.button("", key="collapsed_history", help="Chat history", icon=":material/forum:", use_container_width=True):
            # Icon "History" ở chế độ gọn không đủ chỗ liệt kê tên chat,
            # nên bấm vào sẽ mở rộng sidebar ra để xem danh sách đầy đủ.
            st.session_state.sidebar_collapsed = False
            st.rerun()
    else:
        # ---------------- CHẾ ĐỘ ĐẦY ĐỦ ----------------
        head_col, collapse_col = st.columns([5, 1])
        with head_col:
            st.header("Chats")
        with collapse_col:
            if st.button("", key="collapse_sidebar", help="Collapse", icon=":material/dock_to_left:"):
                st.session_state.sidebar_collapsed = True
                st.rerun()
        if st.button("New chat", key="new_chat_full", icon=":material/edit_square:", use_container_width=True):
            _new_conversation()
            st.rerun()
        st.divider()
        st.caption("Recent")
        # Newest conversation on top
        for conv_id, conv in reversed(list(st.session_state.conversations.items())):
            is_active = conv_id == st.session_state.current_id
            icon = ":material/forum:" if is_active else ":material/chat_bubble:"
            if st.button(conv["title"], key=f"conv_{conv_id}", icon=icon, use_container_width=True):
                st.session_state.current_id = conv_id
                st.rerun()

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


@st.fragment(run_every=0.12)
def _job_progress_fragment():
    """Polls the active job and shows a live Stop button, plus whatever
    text has streamed in so far for the current stage (typing effect), so
    the user watches progress instead of a blank "⏳ ..." message.

    Special case for "synthesis" (the Leader step): Groq can finish
    generating the ENTIRE answer in well under a second — faster than a
    human can perceive as "streaming" even at a fast poll rate, so it
    just looked like "thinking... -> full answer" with nothing in
    between. Instead of handing off to finalize the instant the API call
    itself returns, we keep "revealing" the already-known text a bit at a
    time for a few more ticks — a steady top-to-bottom typing effect that
    doesn't depend on how fast the backend actually finished. This adds
    at most ~0.5-0.7s of visual delay, never blocks on real network
    calls (the text is already fully in memory by then)."""
    job = st.session_state.job
    if job is None:
        return

    if job["stage"] == "synthesis":
        render_model_comparison(job.get("per_model") or [], key="streaming_live")
        st.caption("⏳ Synthesizing final answer...")
        full_text = (job.get("answer_partial") or {}).get("text", "")
        revealed = job.get("reveal_len", 0)
        remaining = len(full_text) - revealed
        if remaining > 0:
            step = max(10, remaining // 5)  # ~5 ticks (~0.6s) to catch up to whatever's known right now
            revealed = min(len(full_text), revealed + step)
            job["reveal_len"] = revealed
        st.markdown((full_text[:revealed] or "_...writing_") + (" ▌" if revealed else ""))

        finished_typing = revealed >= len(full_text)
        if job["done"] and (job["error"] is not None or job["cancelled"] or finished_typing):
            st.rerun()  # API done AND (reveal caught up, or error/cancel) -> hand off to finalize
            return
        if st.button("⏹ Stop", key="stop_button"):
            _cancel_job(job)
        return

    if job["done"]:
        st.rerun()  # hand off to the main script to finalize / chain the job
        return

    if job["stage"] == "ensemble":
        st.caption("⏳ Querying 3 AI models in parallel...")
        mp = job.get("model_partials") or {}
        labels_keys = [
            (MODEL_GEMINI_LABEL, "gemini"),
            (MODEL_QWEN_LABEL, "qwen"),
            (MODEL_GPTOSS_LABEL, "gptoss"),
        ]
        cols = st.columns(3)
        for col, (label, k) in zip(cols, labels_keys):
            with col:
                st.markdown(f"**{label}**")
                text = mp.get(k, "")
                # Hiện FULL độ dài trong lúc đang stream (không giới hạn
                # chiều cao/không cuộn) — khung cuộn cố định chỉ áp dụng
                # sau, bên trong expander "Compare..." lúc đã thu gọn.
                st.write((text + " ▌") if text else "_...thinking_")
    else:
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
        answer_partial = {"text": ""}
        # Chain straight into stage 2 (still cancelable via a fresh Stop button)
        st.session_state.job = _start_job(
            get_final_answer(leader_prompt, on_chunk=lambda t: answer_partial.__setitem__("text", t)),
            stage="synthesis",
            user_text=job["user_text"],
            display_user_text=job["display_user_text"],
            per_model=per_model,
            answer_partial=answer_partial,
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
for _turn_idx, turn in enumerate(current_conv["history"]):
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        if turn.get("type") == "image":
            st.image(turn["image_url"], caption=f"Prompt: {turn['image_prompt_en']}")
        else:
            render_model_comparison(turn["per_model"], key=f"{st.session_state.current_id}_{_turn_idx}")
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
    "Ask fcb everything...",
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
            history_block = build_history_block(current_conv["history"])
            full_query = context_line + history_block + user_text + attachment_block

            model_partials = {"gemini": "", "qwen": "", "gptoss": ""}
            st.session_state.job = _start_job(
                run_ensemble(full_query, model_partials),
                stage="ensemble",
                user_text=user_text,
                display_user_text=display_user_text,
                context_line=context_line,
                model_partials=model_partials,
            )
            st.rerun()
