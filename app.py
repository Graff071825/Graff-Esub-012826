import os
import time
import random
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import streamlit as st
import yaml

# --- PDF/Text processing ---
from PyPDF2 import PdfReader

# --- AI SDKs ---
import google.generativeai as genai
from openai import OpenAI
import anthropic
from xai_sdk import Client as XAIClient
from xai_sdk.chat import user as xai_user, system as xai_system

# Optional: if you use the separate prompts module
try:
    from prompts import BASE_SYSTEM_PROMPT
except ImportError:
    BASE_SYSTEM_PROMPT = ""


# =========================
# 1. CONFIG / CONSTANTS
# =========================

AI_MODELS = {
    "gemini": [
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
        "gemini-2.5-flash-lite",
        "gemini-3-pro-preview",
    ],
    "openai": [
        "gpt-4o-mini",
        "gpt-4.1-mini",
    ],
    "anthropic": [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "xai": [
        "grok-4-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
    ],
}

LANG_LABELS = {
    "en": "English",
    "zh": "繁體中文",
}

THEME_MODE_LABELS = {
    "light": "Light",
    "dark": "Dark",
}

# 20 painter-based “WOW” themes
PAINTER_THEMES = [
    {
        "id": "vangogh_starry",
        "name_en": "Van Gogh · Starry Night",
        "name_zh": "梵谷·星夜",
        "primary": "#1B3B6F",
        "secondary": "#F4D03F",
        "accent": "#F39C12",
        "bg": "#0B1020",
    },
    {
        "id": "monet_waterlilies",
        "name_en": "Monet · Water Lilies",
        "name_zh": "莫內·睡蓮",
        "primary": "#5DADE2",
        "secondary": "#ABEBC6",
        "accent": "#F9E79F",
        "bg": "#F4F9FB",
    },
    {
        "id": "klimt_gold",
        "name_en": "Klimt · Golden Leaf",
        "name_zh": "克林姆·金色",
        "primary": "#D4AC0D",
        "secondary": "#FAD7A0",
        "accent": "#CD6155",
        "bg": "#FEF9E7",
    },
    {
        "id": "picasso_blue",
        "name_en": "Picasso · Blue Period",
        "name_zh": "畢卡索·藍色時期",
        "primary": "#154360",
        "secondary": "#5DADE2",
        "accent": "#A569BD",
        "bg": "#EBF5FB",
    },
    {
        "id": "hokusai_wave",
        "name_en": "Hokusai · Great Wave",
        "name_zh": "北齋·神奈川沖浪裏",
        "primary": "#154360",
        "secondary": "#1ABC9C",
        "accent": "#E67E22",
        "bg": "#ECF0F1",
    },
    {
        "id": "matisse_cutouts",
        "name_en": "Matisse · Cut-Outs",
        "name_zh": "馬諦斯·剪紙",
        "primary": "#E74C3C",
        "secondary": "#3498DB",
        "accent": "#F1C40F",
        "bg": "#FDFEFE",
    },
    {
        "id": "rothko_sunset",
        "name_en": "Rothko · Sunset Fields",
        "name_zh": "羅斯科·暮色",
        "primary": "#922B21",
        "secondary": "#E59866",
        "accent": "#AF601A",
        "bg": "#FDF2E9",
    },
    {
        "id": "dali_surreal",
        "name_en": "Dali · Surreal Sands",
        "name_zh": "達利·超現實",
        "primary": "#7D6608",
        "secondary": "#F8C471",
        "accent": "#1F618D",
        "bg": "#FEF5E7",
    },
    {
        "id": "turner_mist",
        "name_en": "Turner · Misty Harbor",
        "name_zh": "透納·迷霧港灣",
        "primary": "#566573",
        "secondary": "#AEB6BF",
        "accent": "#F7DC6F",
        "bg": "#F8F9F9",
    },
    {
        "id": "pollock_drip",
        "name_en": "Pollock · Drip Energy",
        "name_zh": "波洛克·飛濺",
        "primary": "#2E86C1",
        "secondary": "#F4D03F",
        "accent": "#C0392B",
        "bg": "#FBFCFC",
    },
    {
        "id": "seurat_pointillism",
        "name_en": "Seurat · Pointillism",
        "name_zh": "修拉·點彩",
        "primary": "#1F618D",
        "secondary": "#F7DC6F",
        "accent": "#52BE80",
        "bg": "#FDFEFE",
    },
    {
        "id": "frida_viva",
        "name_en": "Frida · Viva Color",
        "name_zh": "芙烈達·生命色塊",
        "primary": "#196F3D",
        "secondary": "#EC7063",
        "accent": "#F4D03F",
        "bg": "#FEF9E7",
    },
    {
        "id": "cezanne_mountains",
        "name_en": "Cézanne · Mountains",
        "name_zh": "塞尚·群山",
        "primary": "#2E86C1",
        "secondary": "#A9CCE3",
        "accent": "#58D68D",
        "bg": "#EBF5FB",
    },
    {
        "id": "bosch_fantasy",
        "name_en": "Bosch · Dreamscape",
        "name_zh": "波希·夢境",
        "primary": "#512E5F",
        "secondary": "#A569BD",
        "accent": "#F5B041",
        "bg": "#FDF2E9",
    },
    {
        "id": "rembrandt_chiaroscuro",
        "name_en": "Rembrandt · Chiaroscuro",
        "name_zh": "林布蘭·明暗",
        "primary": "#6E2C00",
        "secondary": "#CA6F1E",
        "accent": "#F7DC6F",
        "bg": "#FBEEE6",
    },
    {
        "id": "vermeer_light",
        "name_en": "Vermeer · Soft Light",
        "name_zh": "維梅爾·靜光",
        "primary": "#1F618D",
        "secondary": "#F7DC6F",
        "accent": "#A2D9CE",
        "bg": "#FEF9E7",
    },
    {
        "id": "kandinsky_abstract",
        "name_en": "Kandinsky · Abstract Rhythm",
        "name_zh": "康丁斯基·抽象律動",
        "primary": "#2E86C1",
        "secondary": "#F4D03F",
        "accent": "#E74C3C",
        "bg": "#FBFCFC",
    },
    {
        "id": "chagall_dream",
        "name_en": "Chagall · Midnight Dream",
        "name_zh": "夏卡爾·午夜夢",
        "primary": "#1A237E",
        "secondary": "#5C6BC0",
        "accent": "#FFCA28",
        "bg": "#E8EAF6",
    },
    {
        "id": "yayoi_polka",
        "name_en": "Yayoi Kusama · Polka Infinity",
        "name_zh": "草間彌生·圓點無限",
        "primary": "#C0392B",
        "secondary": "#F7DC6F",
        "accent": "#27AE60",
        "bg": "#FDEDEC",
    },
    {
        "id": "haring_pop",
        "name_en": "Keith Haring · Pop Lines",
        "name_zh": "哈林·塗鴉線條",
        "primary": "#212F3C",
        "secondary": "#E74C3C",
        "accent": "#F1C40F",
        "bg": "#FDFEFE",
    },
]

CORAL_COLOR = "#FF7F50"


@dataclass
class AgentConfig:
    id: str
    name: str
    description: str
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str
    provider: str


@dataclass
class AppState:
    language: str = "en"
    theme_mode: str = "light"
    current_theme_id: str = PAINTER_THEMES[0]["id"]
    health: int = 100
    mana: int = 100
    experience: int = 0
    level: int = 1
    api_keys: Dict[str, Optional[str]] = None


# =========================
# 2. SESSION INIT
# =========================

def load_agents_yaml(path: str) -> List[AgentConfig]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    agents = []
    for a in data.get("agents", []):
        agents.append(
            AgentConfig(
                id=a["id"],
                name=a["name"],
                description=a.get("description", ""),
                model=a["model"],
                max_tokens=int(a.get("max_tokens", 12000)),
                temperature=float(a.get("temperature", 0.2)),
                system_prompt=a.get("system_prompt", ""),
                provider=a["provider"],
            )
        )
    return agents


def init_session_state():
    if "app_state" not in st.session_state:
        st.session_state.app_state = AppState(
            api_keys={
                "gemini": os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY"),
                "openai": os.getenv("OPENAI_API_KEY"),
                "anthropic": os.getenv("ANTHROPIC_API_KEY"),
                "xai": os.getenv("XAI_API_KEY"),
            }
        )
    if "agents" not in st.session_state:
        st.session_state.agents = load_agents_yaml("agents.yaml")
    if "pipeline_results" not in st.session_state:
        st.session_state.pipeline_results = {}
    if "pipeline_inputs" not in st.session_state:
        st.session_state.pipeline_inputs = {}
    if "pipeline_view_modes" not in st.session_state:
        st.session_state.pipeline_view_modes = {}
    if "execution_log" not in st.session_state:
        st.session_state.execution_log = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "total_runs": 0,
            "provider_calls": {"gemini": 0, "openai": 0, "anthropic": 0, "xai": 0},
            "tokens_used": 0,
            "last_run_duration": 0.0,
        }
    if "agent_status" not in st.session_state:
        st.session_state.agent_status = {a.id: "idle" for a in st.session_state.agents}
    if "ocr_text" not in st.session_state:
        st.session_state.ocr_text = ""
    if "note_content" not in st.session_state:
        st.session_state.note_content = ""


# =========================
# 3. THEME & STYLING
# =========================

def get_current_theme() -> Dict[str, str]:
    theme_id = st.session_state.app_state.current_theme_id
    for theme in PAINTER_THEMES:
        if theme["id"] == theme_id:
            return theme
    return PAINTER_THEMES[0]


def inject_global_css():
    theme = get_current_theme()
    mode = st.session_state.app_state.theme_mode

    bg_color = theme["bg"] if mode == "light" else "#05080F"
    text_color = "#0B1725" if mode == "light" else "#ECF0F1"

    css = f"""
    <style>
    :root {{
        --primary: {theme["primary"]};
        --secondary: {theme["secondary"]};
        --accent: {theme["accent"]};
        --bg-color: {bg_color};
        --text-color: {text_color};
        --coral: {CORAL_COLOR};
    }}
    body {{
        background: radial-gradient(circle at top, var(--bg-color) 0%, #02040f 100%);
        color: var(--text-color);
    }}
    .nordic-card {{
        background: rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.0rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(18px);
    }}
    .nordic-badge {{
        border-radius: 999px;
        padding: 0.1rem 0.8rem;
        font-size: 0.7rem;
        border: 1px solid rgba(255,255,255,0.3);
    }}
    .mana-orb {{
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #ffffff, var(--accent));
        box-shadow: 0 0 25px rgba(130, 224, 170, 0.8);
        position: relative;
    }}
    .mana-orb-inner {{
        position: absolute;
        inset: 10px;
        border-radius: 50%;
        background: radial-gradient(circle at 20% 20%, rgba(255,255,255,0.8), transparent);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0% {{ box-shadow: 0 0 0 0 rgba(130,224,170,0.6); }}
        70% {{ box-shadow: 0 0 0 18px rgba(130,224,170,0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(130,224,170,0); }}
    }}
    .status-dot {{
        height: 10px;
        width: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }}
    .status-dot-idle {{ background: #95A5A6; }}
    .status-dot-running {{ background: #F4D03F; animation: blink 1s infinite; }}
    .status-dot-success {{ background: #2ECC71; }}
    .status-dot-error {{ background: #E74C3C; }}
    @keyframes blink {{
        0% {{ opacity: 0.2; }}
        50% {{ opacity: 1; }}
        100% {{ opacity: 0.2; }}
    }}
    .coral-text {{
        color: {CORAL_COLOR};
        font-weight: 600;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =========================
# 4. API KEY HANDLING
# =========================

def get_api_key(provider: str) -> Optional[str]:
    # Prefer environment (do not show), fallback to session/UI field
    key_env = None
    if provider == "gemini":
        key_env = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    elif provider == "openai":
        key_env = os.getenv("OPENAI_API_KEY")
    elif provider == "anthropic":
        key_env = os.getenv("ANTHROPIC_API_KEY")
    elif provider == "xai":
        key_env = os.getenv("XAI_API_KEY")

    if key_env:
        st.session_state.app_state.api_keys[provider] = key_env
        return key_env

    return st.session_state.app_state.api_keys.get(provider)


def api_key_input_ui():
    st.subheader("🔐 API Keys (Client-Side Only)")
    st.caption(
        "Keys stay in memory only. In production on Hugging Face Spaces, prefer environment variables."
    )

    cols = st.columns(4)
    providers = ["gemini", "openai", "anthropic", "xai"]
    labels = ["Google Gemini", "OpenAI", "Anthropic", "Grok (xAI)"]
    for col, provider, label in zip(cols, providers, labels):
        with col:
            env_present = get_api_key(provider) is not None
            if env_present:
                st.success(f"{label}: using env var")
            else:
                val = st.text_input(
                    f"{label} API Key",
                    type="password",
                    key=f"{provider}_manual_api_key",
                )
                if val:
                    st.session_state.app_state.api_keys[provider] = val


# =========================
# 5. PROVIDER CALLS
# =========================

def call_gemini(model: str, system_prompt: str, user_input: str,
                max_tokens: int, temperature: float, api_key: str) -> str:
    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(
        model_name=model,
        system_instruction=(system_prompt or BASE_SYSTEM_PROMPT or None),
    )

    try:
        response = model_client.generate_content(
            [user_input],
            generation_config={
                "temperature": float(temperature),
                "max_output_tokens": int(max_tokens),
            },
        )
    except Exception as e:
        msg = str(e)
        upper_msg = msg.upper()
        if "SAFETY" in upper_msg or "HARM_" in upper_msg or "HARM CATEGORY" in upper_msg:
            return (
                "⚠️ Gemini 已封鎖此輸入，原因與其內建安全機制相關。\n"
                "建議：\n"
                "- 嘗試稍微調整描述方式，避免過於敏感或模糊的語句；或\n"
                "- 在此情境下可改用 OpenAI / Anthropic / Grok 等其他模型執行相同步驟。"
            )
        return f"⚠️ Gemini 呼叫失敗：{msg}"

    try:
        return response.text
    except Exception:
        if hasattr(response, "candidates") and response.candidates:
            parts = response.candidates[0].content.parts
            txt = "".join(
                getattr(p, "text", "") for p in parts
                if hasattr(p, "text")
            )
            if txt.strip():
                return txt
        return "⚠️ 無法從 Gemini 回應中解析文字內容。"


def call_openai(model: str, system_prompt: str, user_input: str,
                max_tokens: int, temperature: float, api_key: str) -> str:
    client = OpenAI(api_key=api_key)
    messages = []
    if BASE_SYSTEM_PROMPT or system_prompt:
        messages.append({"role": "system", "content": (BASE_SYSTEM_PROMPT + "\n\n" + system_prompt)})
    else:
        messages.append({"role": "system", "content": "You are a helpful regulatory assistant."})
    messages.append({"role": "user", "content": user_input})

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def call_anthropic(model: str, system_prompt: str, user_input: str,
                   max_tokens: int, temperature: float, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    m = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=(BASE_SYSTEM_PROMPT + "\n\n" + system_prompt) if system_prompt else BASE_SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": user_input},
        ],
    )
    return "".join(block.text for block in m.content if hasattr(block, "text"))


def call_xai(model: str, system_prompt: str, user_input: str,
             max_tokens: int, temperature: float, api_key: str) -> str:
    client = XAIClient(api_key=api_key, timeout=3600)
    chat = client.chat.create(model=model)
    sys_text = (BASE_SYSTEM_PROMPT + "\n\n" + system_prompt) if system_prompt else BASE_SYSTEM_PROMPT
    chat.append(xai_system(sys_text or "You are Grok, a highly intelligent, helpful AI assistant."))
    chat.append(xai_user(user_input))
    response = chat.sample(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    return response.content


def run_agent(agent: AgentConfig, input_text: str) -> str:
    provider = agent.provider
    api_key = get_api_key(provider)
    if not api_key:
        raise RuntimeError(f"No API key configured for provider '{provider}'")

    system_prompt = agent.system_prompt or ""
    model = agent.model
    t0 = time.time()

    st.session_state.agent_status[agent.id] = "running"

    if provider == "gemini":
        out = call_gemini(model, system_prompt, input_text, agent.max_tokens, agent.temperature, api_key)
    elif provider == "openai":
        out = call_openai(model, system_prompt, input_text, agent.max_tokens, agent.temperature, api_key)
    elif provider == "anthropic":
        out = call_anthropic(model, system_prompt, input_text, agent.max_tokens, agent.temperature, api_key)
    elif provider == "xai":
        out = call_xai(model, system_prompt, input_text, agent.max_tokens, agent.temperature, api_key)
    else:
        st.session_state.agent_status[agent.id] = "error"
        raise ValueError(f"Unsupported provider: {provider}")

    duration = time.time() - t0
    st.session_state.metrics["provider_calls"][provider] += 1
    st.session_state.metrics["total_runs"] += 1
    st.session_state.metrics["last_run_duration"] = duration

    st.session_state.app_state.mana = max(0, st.session_state.app_state.mana - 20)
    st.session_state.app_state.experience += 10
    st.session_state.app_state.level = 1 + st.session_state.app_state.experience // 100

    st.session_state.agent_status[agent.id] = "success"
    return out


def run_ad_hoc_llm(provider: str, model: str, system_prompt: str, user_prompt: str,
                   max_tokens: int, temperature: float) -> str:
    dummy = AgentConfig(
        id="adhoc",
        name="AdHoc",
        description="",
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
        provider=provider,
    )
    # For ad-hoc, don’t use agent_status
    return run_agent(dummy, user_prompt)


# =========================
# 6. GAMIFIED STATUS / WOW FEATURES
# =========================

def wow_header():
    theme = get_current_theme()
    lang = st.session_state.app_state.language
    name = theme["name_en"] if lang == "en" else theme["name_zh"]
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
          <div>
            <h1 style="margin-bottom:0.2rem;">FDA 510(k) Review Studio · Painter Edition</h1>
            <div style="font-size:0.85rem;opacity:0.85;">
              Agentic Regulatory Workspace · {name}
            </div>
          </div>
          <div style="text-align:right;font-size:0.8rem;opacity:0.85;">
            <span class="nordic-badge">510(k) · Agentic AI</span>
            <span class="nordic-badge">WOW UI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def wow_status_bar():
    app = st.session_state.app_state
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        st.markdown("**Health**")
        st.progress(app.health / 100)
    with col2:
        st.markdown("**Mana**")
        st.progress(app.mana / 100)
    with col3:
        st.metric("Level", app.level, help="Level based on cumulative XP")
        st.caption(f"XP: {app.experience}")
    with col4:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:1rem;">
              <div class="mana-orb">
                <div class="mana-orb-inner"></div>
              </div>
              <div style="flex:1;">
                <div style="font-size:0.8rem;opacity:0.9;">Regulatory Stress Meter</div>
            """,
            unsafe_allow_html=True,
        )
        stress = max(0, 100 - app.health)
        st.progress(stress / 100, text=f"Stress: {stress}%")
        st.markdown("</div></div>", unsafe_allow_html=True)

    unlocked = []
    if app.experience >= 50:
        unlocked.append("🌸 First Bloom Reviewer (50+ XP)")
    if app.experience >= 200:
        unlocked.append("🌺 Seasoned 510(k) Reviewer (200+ XP)")
    if st.session_state.metrics["total_runs"] >= 10:
        unlocked.append("🌷 Ten Runs of Tranquility (10+ AI calls)")

    if unlocked:
        st.markdown(
            "<div class='nordic-card'><strong>Achievement Blossoms</strong><br>" +
            "<br>".join(unlocked) +
            "</div>",
            unsafe_allow_html=True,
        )


def style_jackpot():
    if st.button("🎰 Style Jackpot"):
        theme = random.choice(PAINTER_THEMES)
        st.session_state.app_state.current_theme_id = theme["id"]
        st.toast(f"Theme changed to {theme['name_en']} / {theme['name_zh']}")


def render_agent_status_bar():
    if not st.session_state.agents:
        return
    st.markdown("##### 🔄 Agent Status")
    cols = st.columns(len(st.session_state.agents))
    for col, agent in zip(cols, st.session_state.agents):
        status = st.session_state.agent_status.get(agent.id, "idle")
        with col:
            st.markdown(
                f"""
                <div style="font-size:0.8rem;">
                  <span class="status-dot status-dot-{status}"></span>
                  <strong>{agent.name}</strong><br>
                  <span style="opacity:0.7;font-size:0.7rem;">{status}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================
# 7. PDF UTILITIES
# =========================

def render_pdf_viewer(uploaded_file):
    if not uploaded_file:
        return
    b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def extract_pdf_pages_text(uploaded_file, selected_pages: List[int]) -> str:
    if not uploaded_file:
        return ""
    reader = PdfReader(uploaded_file)
    texts = []
    for p in selected_pages:
        if 0 <= p < len(reader.pages):
            try:
                page_text = reader.pages[p].extract_text() or ""
            except Exception:
                page_text = ""
            texts.append(f"\n\n--- Page {p+1} ---\n{page_text}")
    return "\n".join(texts)


# =========================
# 8. PIPELINE UI (ORIGINAL FEATURE, IMPROVED)
# =========================

def pipeline_tab():
    st.subheader("🔗 Multi-Agent 510(k) Review Pipeline")

    agents = st.session_state.agents
    if not agents:
        st.error("No agents loaded from agents.yaml")
        return

    render_agent_status_bar()

    global_input = st.text_area(
        "Global Case Input (e.g. device description, indications, test summaries)",
        height=180,
        key="pipeline_global_input",
    )

    st.caption("You can run agents one-by-one, or chain them using **Run Full Pipeline**.")
    run_all = st.button("🚀 Run Full Pipeline (sequential chaining)", type="primary")

    if run_all:
        if st.session_state.app_state.mana < 20:
            st.error("Not enough Mana to start the pipeline (need at least 20).")
            return

        st.session_state.execution_log.append(
            {
                "time": time.strftime("%H:%M:%S"),
                "type": "info",
                "msg": "Full pipeline execution started.",
            }
        )

        for idx, a in enumerate(agents):
            provider_key = f"provider_{a.id}"
            model_key = f"model_{a.id}"
            temp_key = f"temp_{a.id}"
            max_tokens_key = f"max_tokens_{a.id}"
            sys_key = f"system_prompt_{a.id}"

            if provider_key in st.session_state:
                a.provider = st.session_state[provider_key]
            if model_key in st.session_state:
                a.model = st.session_state[model_key]
            if temp_key in st.session_state:
                a.temperature = float(st.session_state[temp_key])
            if max_tokens_key in st.session_state:
                a.max_tokens = int(st.session_state[max_tokens_key])
            if sys_key in st.session_state:
                a.system_prompt = st.session_state[sys_key]

            if idx == 0:
                step_input = global_input or ""
            else:
                prev_agent = agents[idx - 1]
                prev_id = prev_agent.id
                prev_output_key = f"output_{prev_id}"
                if prev_output_key in st.session_state and str(st.session_state[prev_output_key]).strip():
                    step_input = st.session_state[prev_output_key]
                else:
                    step_input = st.session_state.pipeline_results.get(prev_id, "")

            st.session_state.agent_status[a.id] = "running"
            with st.spinner(f"Running Agent {idx+1}: {a.name}…"):
                try:
                    result = run_agent(a, step_input or "")
                    st.session_state.pipeline_results[a.id] = result
                    st.session_state[f"output_{a.id}"] = result
                    st.session_state.execution_log.append(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "type": "success",
                            "msg": f"Agent {idx+1} ({a.name}) completed (full pipeline).",
                        }
                    )
                except Exception as e:
                    st.session_state.agent_status[a.id] = "error"
                    st.session_state.execution_log.append(
                        {
                            "time": time.strftime("%H:%M:%S"),
                            "type": "error",
                            "msg": f"Agent {idx+1} ({a.name}) failed during full pipeline: {e}",
                        }
                    )
                    st.error(f"Agent {a.name} failed: {e}")
                    break

    st.markdown("### 📄 Per-Agent Configuration & Editable Chain")

    prev_agent_id = None
    for idx, a in enumerate(agents):
        st.markdown(f"#### Step {idx+1}: {a.name}")
        st.caption(a.description)

        with st.container():
            cfg_cols = st.columns([1, 1, 1, 1])
            with cfg_cols[0]:
                provider = st.selectbox(
                    "Provider",
                    options=list(AI_MODELS.keys()),
                    index=list(AI_MODELS.keys()).index(a.provider) if a.provider in AI_MODELS else 0,
                    key=f"provider_{a.id}",
                )
            with cfg_cols[1]:
                model = st.selectbox(
                    "Model",
                    options=AI_MODELS[provider],
                    index=AI_MODELS[provider].index(a.model) if a.model in AI_MODELS[provider] else 0,
                    key=f"model_{a.id}",
                )
            with cfg_cols[2]:
                temp = st.slider(
                    "Temperature",
                    0.0,
                    1.0,
                    value=float(a.temperature),
                    key=f"temp_{a.id}",
                )
            with cfg_cols[3]:
                max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=256,
                    max_value=12000,
                    value=int(a.max_tokens),
                    step=256,
                    key=f"max_tokens_{a.id}",
                )

            a.provider = provider
            a.model = model
            a.temperature = temp
            a.max_tokens = max_tokens

            a.system_prompt = st.text_area(
                "System Prompt (可編輯，繁體中文/English 混用皆可)",
                value=a.system_prompt,
                key=f"system_prompt_{a.id}",
                height=160,
            )

            st.markdown("---")

            input_key = f"input_{a.id}"
            if input_key in st.session_state:
                default_input = st.session_state[input_key]
            else:
                if idx == 0:
                    default_input = global_input
                else:
                    prev_id = prev_agent_id
                    prev_output_key = f"output_{prev_id}"
                    if prev_output_key in st.session_state and str(st.session_state[prev_output_key]).strip():
                        default_input = st.session_state[prev_output_key]
                    else:
                        default_input = st.session_state.pipeline_results.get(prev_id, "")

            input_text = st.text_area(
                "Input to this agent (editable; can be used for the next agent)",
                value=default_input,
                height=180,
                key=input_key,
            )

            run_step = st.button(f"▶️ Run only this step: {a.name}", key=f"run_step_{a.id}")

            if run_step:
                if st.session_state.app_state.mana < 20:
                    st.error("Not enough Mana to run this agent (need at least 20).")
                else:
                    st.session_state.agent_status[a.id] = "running"
                    with st.spinner(f"Running {a.name}…"):
                        try:
                            result = run_agent(a, input_text or "")
                            st.session_state.pipeline_results[a.id] = result
                            st.session_state[f"output_{a.id}"] = result
                            st.session_state.execution_log.append(
                                {
                                    "time": time.strftime("%H:%M:%S"),
                                    "type": "success",
                                    "msg": f"Agent {idx+1} ({a.name}) completed (single step).",
                                }
                            )
                        except Exception as e:
                            st.session_state.agent_status[a.id] = "error"
                            st.session_state.execution_log.append(
                                {
                                    "time": time.strftime("%H:%M:%S"),
                                    "type": "error",
                                    "msg": f"Agent {idx+1} ({a.name}) failed (single step): {e}",
                                }
                            )
                            st.error(f"Agent {a.name} failed: {e}")

            if a.id in st.session_state.pipeline_results:
                st.markdown("**Output of this agent**")

                view_mode = st.session_state.pipeline_view_modes.get(a.id, "Edit (Text)")
                view_mode = st.radio(
                    "View mode",
                    options=["Edit (Text)", "Preview (Markdown)"],
                    index=0 if view_mode == "Edit (Text)" else 1,
                    horizontal=True,
                    key=f"view_{a.id}",
                )
                st.session_state.pipeline_view_modes[a.id] = view_mode

                output_key = f"output_{a.id}"
                if output_key not in st.session_state:
                    st.session_state[output_key] = st.session_state.pipeline_results[a.id]

                if view_mode == "Edit (Text)":
                    edited_output = st.text_area(
                        "Editable Output (used as potential input for next step)",
                        value=st.session_state[output_key],
                        height=220,
                        key=output_key,
                    )
                    st.session_state.pipeline_results[a.id] = edited_output
                else:
                    st.markdown(
                        st.session_state.pipeline_results[a.id],
                        help="Markdown preview of the agent output.",
                    )

                st.info(
                    "Next agent’s default input comes from the latest version of this output, "
                    "unless you manually override it."
                )

        prev_agent_id = a.id


# =========================
# 9. 510(k) SUMMARY ANALYSIS
# =========================

def summary_tab():
    st.subheader("📑 510(k) Summary Analysis")

    col_in, col_cfg = st.columns([2, 1])

    with col_in:
        st.markdown("**1. Input 510(k) Summary (text/markdown 或 PDF)**")
        uploaded = st.file_uploader("Upload 510(k) summary (PDF)", type=["pdf"], key="summary_pdf")
        summary_text = st.text_area(
            "Or paste 510(k) summary (text / markdown)",
            height=260,
            key="summary_text",
        )

        if uploaded and not summary_text.strip():
            reader = PdfReader(uploaded)
            text_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                text_pages.append(f"\n\n--- Page {i+1} ---\n{t}")
            summary_text = "\n".join(text_pages)

        st.markdown("**2. Preview (optional)**")
        if uploaded:
            with st.expander("Preview PDF"):
                render_pdf_viewer(uploaded)

    with col_cfg:
        st.markdown("**Model & Prompt Settings**")
        provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="summary_provider")
        model = st.selectbox("Model", AI_MODELS[provider], key="summary_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, key="summary_temp")
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=12000,
            value=4000,
            step=256,
            key="summary_max_tokens",
        )
        default_prompt = (
            "You are an FDA 510(k) reviewer.\n"
            "Create a comprehensive, well-structured Markdown summary of the provided 510(k) summary.\n"
            "- Length: approximately 2000–3000 words.\n"
            "- Organize into clear sections: Device Overview, Intended Use, Indications for Use, "
            "Technological Characteristics, Performance Testing, Biocompatibility, Sterilization, "
            "Shelf Life, Substantial Equivalence, and Regulatory Notes.\n"
            f"- Highlight 20–40 critical keywords by wrapping them in "
            f"<span style='color:{CORAL_COLOR};font-weight:600;'>keyword</span>.\n"
            "- Use tables where helpful. Do not invent data; only reorganize and clarify.\n"
        )
        user_prompt = st.text_area(
            "Analysis Prompt (customizable)",
            value=default_prompt,
            height=220,
            key="summary_prompt",
        )

        run_btn = st.button("🧠 Analyze 510(k) Summary", type="primary")

    st.markdown("---")
    st.markdown("### 📘 AI-Generated Summary")

    if run_btn:
        if not summary_text.strip():
            st.error("Please provide summary text or upload a PDF.")
            return
        with st.spinner("Analyzing 510(k) summary…"):
            try:
                out = run_ad_hoc_llm(
                    provider=provider,
                    model=model,
                    system_prompt=user_prompt,
                    user_prompt=summary_text,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                st.markdown(out, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Summary analysis failed: {e}")


# =========================
# 10. REVIEW GUIDANCE BUILDER
# =========================

def guidance_tab():
    st.subheader("📘 Review Guidance Builder")

    col_in, col_cfg = st.columns([2, 1])

    with col_in:
        st.markdown("**1. Input Review Guidance (PDF / text / markdown)**")
        g_pdf = st.file_uploader("Upload review guidance (PDF)", type=["pdf"], key="guidance_pdf")
        g_text = st.text_area(
            "Or paste review guidance text / markdown",
            height=260,
            key="guidance_text",
        )

        if g_pdf and not g_text.strip():
            reader = PdfReader(g_pdf)
            text_pages = []
            for i, page in enumerate(reader.pages):
                try:
                    t = page.extract_text() or ""
                except Exception:
                    t = ""
                text_pages.append(f"\n\n--- Page {i+1} ---\n{t}")
            g_text = "\n".join(text_pages)

        with st.expander("Preview PDF (optional)"):
            if g_pdf:
                render_pdf_viewer(g_pdf)
            else:
                st.info("Upload a PDF to preview.")

        st.markdown("**2. Reviewer Instructions (optional)**")
        instructions = st.text_area(
            "Special reviewer instructions (e.g., focus on software validation, cybersecurity, pediatric use)",
            height=160,
            key="guidance_instructions",
        )

    with col_cfg:
        st.markdown("**Model & Prompt Settings**")
        provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="guidance_provider")
        model = st.selectbox("Model", AI_MODELS[provider], key="guidance_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.25, key="guidance_temp")
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=12000,
            value=5000,
            step=256,
            key="guidance_max_tokens",
        )

        default_prompt = (
            "You are creating an FDA 510(k) review guideline for reviewers.\n"
            "Based on the supplied guidance text and optional reviewer instructions, produce:\n"
            "1) A concise narrative guideline explaining how to review this type of device, and\n"
            "2) A detailed checklist (Markdown) that a reviewer can use during 510(k) review.\n"
            "Checklist must include sections such as: Administrative, Indications/IFU, Device Description, "
            "Risk Management, Bench Testing, Biocompatibility, Sterilization, EMC/Electrical Safety, "
            "Software/Cybersecurity (when applicable), Labeling, and Summary of SE.\n"
            "For each checklist item, include columns: Item, Question, Evidence to Confirm, Pass/Fail/NA.\n"
        )
        sys_prompt = st.text_area(
            "Guideline Prompt (customizable)",
            value=default_prompt,
            height=220,
            key="guidance_prompt",
        )

        run_btn = st.button("📋 Generate Review Guideline + Checklist", type="primary")

    st.markdown("---")
    st.markdown("### 📑 Generated Guideline & Checklist")

    if run_btn:
        if not g_text.strip():
            st.error("Please provide review guidance text or upload a PDF.")
            return
        combined = f"=== REVIEW GUIDANCE ===\n{g_text}\n\n=== REVIEWER INSTRUCTIONS ===\n{instructions}"
        with st.spinner("Building guideline and checklist…"):
            try:
                out = run_ad_hoc_llm(
                    provider=provider,
                    model=model,
                    system_prompt=sys_prompt,
                    user_prompt=combined,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                st.markdown(out)
            except Exception as e:
                st.error(f"Guideline generation failed: {e}")


# =========================
# 11. SUBMISSION OCR & AGENTS
# =========================

def ocr_agents_tab():
    st.subheader("📂 Submission Material · PDF Viewer, OCR/Text & Agents")

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("**1. Upload Submission PDF & Select Pages**")
        sub_pdf = st.file_uploader("Upload 510(k) submission (PDF)", type=["pdf"], key="sub_pdf")
        if sub_pdf:
            reader = PdfReader(sub_pdf)
            num_pages = len(reader.pages)
            page_idx = list(range(num_pages))
            selected = st.multiselect(
                "Select pages for OCR/text extraction",
                options=page_idx,
                default=page_idx[: min(5, num_pages)],
                format_func=lambda i: f"Page {i+1}",
                key="sub_pages",
            )
            st.markdown("**PDF Preview**")
            render_pdf_viewer(sub_pdf)
        else:
            selected = []

        st.markdown("**2. OCR/Text Extraction Settings**")
        ocr_lang = st.selectbox(
            "Language for OCR/Text focus",
            options=["English", "繁體中文 / Traditional Chinese"],
            key="ocr_lang",
        )
        extract_btn = st.button("🔍 Extract Text from Selected Pages")

        if extract_btn:
            if not sub_pdf or not selected:
                st.error("Upload a PDF and select at least one page.")
            else:
                text = extract_pdf_pages_text(sub_pdf, selected)
                if not text.strip():
                    st.warning("No text extracted. (If this is image-only PDF, you may need true OCR.)")
                st.session_state.ocr_text = text

        st.markdown("**3. OCR/Text Document Editor**")
        view_mode = st.radio(
            "View mode",
            options=["Edit (Markdown/Text)", "Preview (Markdown)"],
            horizontal=True,
            key="ocr_view_mode",
        )
        if view_mode == "Edit (Markdown/Text)":
            txt = st.text_area(
                "OCR / extracted document (you can edit freely)",
                value=st.session_state.ocr_text,
                height=320,
                key="ocr_text_area",
            )
            st.session_state.ocr_text = txt
        else:
            st.markdown(st.session_state.ocr_text or "_No OCR text yet_", unsafe_allow_html=True)

    with col_right:
        st.markdown("**4. Auto Summary of OCR Document (2000–3000 words)**")
        provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="ocr_summary_provider")
        model = st.selectbox("Model", AI_MODELS[provider], key="ocr_summary_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, key="ocr_summary_temp")
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=1024,
            max_value=12000,
            value=6000,
            step=256,
            key="ocr_summary_max_tokens",
        )
        default_prompt = (
            "You are an FDA 510(k) reviewer.\n"
            "Create a comprehensive Markdown summary of the provided OCR/extracted submission content.\n"
            "Length: 2000–3000 words. Use headings, bullet points, and tables.\n"
            f"Highlight important regulatory keywords by wrapping them in "
            f"<span style='color:{CORAL_COLOR};font-weight:600;'>keyword</span>.\n"
            "Focus on: device description, intended use, indications, technological characteristics, "
            "bench/animal/clinical testing, risk management, comparison to predicate, labeling, and any gaps.\n"
        )
        summary_prompt = st.text_area(
            "Summary Prompt (customizable)",
            value=default_prompt,
            height=250,
            key="ocr_summary_prompt",
        )
        run_summary = st.button("🧾 Generate 2000–3000 word summary")

    st.markdown("---")
    st.markdown("### 📄 OCR-Based Summary")

    if run_summary:
        if not st.session_state.ocr_text.strip():
            st.error("No OCR/extracted text available. Please upload PDF and run extraction first.")
        else:
            with st.spinner("Summarizing OCR document…"):
                try:
                    out = run_ad_hoc_llm(
                        provider=provider,
                        model=model,
                        system_prompt=summary_prompt,
                        user_prompt=st.session_state.ocr_text,
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                    )
                    st.markdown(out, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"OCR summary failed: {e}")

    st.markdown("---")
    st.markdown("### 🤖 Keep Prompting on OCR Document / Run Agents")

    col_chat, col_agents = st.columns(2)

    with col_chat:
        st.markdown("**Ad-hoc Q&A on OCR Document**")
        chat_provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="ocr_chat_provider")
        chat_model = st.selectbox("Model", AI_MODELS[chat_provider], key="ocr_chat_model")
        chat_temp = st.slider("Temperature", 0.0, 1.0, 0.3, key="ocr_chat_temp")
        chat_max_tokens = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=12000,
            value=4000,
            step=256,
            key="ocr_chat_max",
        )
        chat_prompt = st.text_area(
            "Enter your question/instructions about the OCR document",
            height=160,
            key="ocr_chat_prompt",
        )
        run_chat = st.button("💬 Ask AI about OCR document")

        if run_chat:
            if not st.session_state.ocr_text.strip():
                st.error("No OCR text available.")
            else:
                combined = f"=== OCR DOCUMENT ===\n{st.session_state.ocr_text}\n\n=== REVIEWER QUESTION ===\n{chat_prompt}"
                with st.spinner("Answering based on OCR document…"):
                    try:
                        out = run_ad_hoc_llm(
                            provider=chat_provider,
                            model=chat_model,
                            system_prompt="You are an FDA 510(k) review assistant. Answer only from the OCR document.",
                            user_prompt=combined,
                            max_tokens=int(chat_max_tokens),
                            temperature=float(chat_temp),
                        )
                        st.markdown(out)
                    except Exception as e:
                        st.error(f"OCR chat failed: {e}")

    with col_agents:
        st.markdown("**Run Agent from agents.yaml on OCR Document**")
        agents = st.session_state.agents
        if not agents:
            st.info("No agents defined in agents.yaml.")
        else:
            selected_agent_name = st.selectbox(
                "Select agent",
                options=[a.name for a in agents],
                key="ocr_agent_select",
            )
            agent = next(a for a in agents if a.name == selected_agent_name)

            provider_override = st.selectbox(
                "Override provider (optional)",
                options=["(use agent config)"] + list(AI_MODELS.keys()),
                key="ocr_agent_provider_override",
            )
            model_override = None
            if provider_override != "(use agent config)":
                model_override = st.selectbox(
                    "Override model",
                    options=AI_MODELS[provider_override],
                    key="ocr_agent_model_override",
                )

            max_tokens_override = st.number_input(
                "Max Tokens (override)",
                min_value=512,
                max_value=12000,
                value=int(agent.max_tokens),
                step=256,
                key="ocr_agent_max_tokens_override",
            )
            temp_override = st.slider(
                "Temperature (override)",
                0.0,
                1.0,
                value=float(agent.temperature),
                key="ocr_agent_temp_override",
            )
            sys_prompt_override = st.text_area(
                "System Prompt (override/append)",
                value=agent.system_prompt,
                height=160,
                key="ocr_agent_sys_override",
            )
            run_agent_btn = st.button("🤖 Run selected agent on OCR document")

            if run_agent_btn:
                if not st.session_state.ocr_text.strip():
                    st.error("No OCR text available.")
                else:
                    # Clone agent with overrides
                    custom_agent = AgentConfig(
                        id=agent.id + "_ocr",
                        name=agent.name + " (OCR Run)",
                        description=agent.description,
                        model=model_override or agent.model,
                        max_tokens=int(max_tokens_override),
                        temperature=float(temp_override),
                        system_prompt=sys_prompt_override,
                        provider=(agent.provider if provider_override == "(use agent config)" else provider_override),
                    )
                    with st.spinner(f"Running agent '{agent.name}' on OCR document…"):
                        try:
                            out = run_agent(custom_agent, st.session_state.ocr_text)
                            st.markdown(out)
                        except Exception as e:
                            st.error(f"OCR agent run failed: {e}")


# =========================
# 12. 510(k) REVIEW REPORT GENERATOR
# =========================

def review_report_tab():
    st.subheader("📝 510(k) Review Report Composer")

    col_in, col_cfg = st.columns([2, 1])

    with col_in:
        st.markdown("**1. Review Report Instructions**")
        report_instr = st.text_area(
            "Describe how you want the review report structured "
            "(e.g., include benefit-risk, gap analysis vs guidance, decision rationale, etc.)",
            height=200,
            key="report_instr",
        )

        st.markdown("**2. Raw Review Materials**")
        raw_file = st.file_uploader(
            "Upload raw review materials (PDF / TXT / MD, optional)",
            type=["pdf", "txt", "md", "markdown"],
            key="report_raw_file",
        )
        raw_text = st.text_area(
            "Or paste raw review materials (e.g. review notes, AI outputs, test summaries)",
            height=260,
            key="report_raw_text",
        )

        if raw_file and not raw_text.strip():
            if raw_file.type == "application/pdf" or raw_file.name.lower().endswith(".pdf"):
                reader = PdfReader(raw_file)
                text_pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    text_pages.append(f"\n\n--- Page {i+1} ---\n{t}")
                raw_text = "\n".join(text_pages)
            else:
                raw_text = raw_file.getvalue().decode("utf-8", errors="ignore")

    with col_cfg:
        st.markdown("**Model & Prompt Settings**")
        provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="report_provider")
        model = st.selectbox("Model", AI_MODELS[provider], key="report_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, key="report_temp")
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=2000,
            max_value=12000,
            value=8000,
            step=256,
            key="report_max_tokens",
        )
        default_prompt = (
            "You are writing a formal FDA 510(k) review report based on:\n"
            "1) Reviewer instructions, and\n"
            "2) Raw review materials.\n"
            "Produce a comprehensive Markdown report of about 2000–3000 words with:\n"
            "- Executive Summary\n"
            "- Device Description & Indications for Use\n"
            "- Regulatory Background (predicate, product code, classification)\n"
            "- Summary of Submitted Testing (bench, biocompatibility, software, EMC/electrical, etc.)\n"
            "- Risk/Benefit and SE Discussion\n"
            "- Issues/Deficiencies (tables) and recommendations\n"
            "- Conclusion and Suggested Decision.\n"
            "Use tables for: testing overview, comparison to predicate, and deficiency tracking.\n"
            f"Highlight key regulatory keywords in <span style='color:{CORAL_COLOR};font-weight:600;'>coral</span>.\n"
        )
        sys_prompt = st.text_area(
            "Report Prompt (customizable)",
            value=default_prompt,
            height=260,
            key="report_prompt",
        )
        run_btn = st.button("📄 Generate 510(k) Review Report", type="primary")

    st.markdown("---")
    st.markdown("### 📜 Generated Review Report")

    if run_btn:
        if not raw_text.strip():
            st.error("Provide raw review materials (upload or paste).")
            return
        combined = (
            f"=== REVIEW REPORT INSTRUCTIONS ===\n{report_instr}\n\n"
            f"=== RAW REVIEW MATERIALS ===\n{raw_text}"
        )
        with st.spinner("Generating 510(k) review report…"):
            try:
                out = run_ad_hoc_llm(
                    provider=provider,
                    model=model,
                    system_prompt=sys_prompt,
                    user_prompt=combined,
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                )
                st.markdown(out, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Review report generation failed: {e}")


# =========================
# 13. NOTE KEEPER + AI MAGICS
# =========================

def parse_keyword_color_pairs(keyword_spec: str):
    """
    Format: 'keyword1#FF0000, keyword2#00FF00, keyword3'
    If color omitted, coral is used.
    """
    pairs = []
    for part in keyword_spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "#" in part:
            kw, color = part.split("#", 1)
            kw = kw.strip()
            color = "#" + color.strip().lstrip("#")
        else:
            kw = part
            color = CORAL_COLOR
        pairs.append((kw, color))
    return pairs


def highlight_keywords_colored(text: str, keyword_spec: str) -> str:
    if not keyword_spec.strip():
        return text
    pairs = parse_keyword_color_pairs(keyword_spec)
    highlighted = text
    for kw, color in pairs:
        if not kw:
            continue
        highlighted = highlighted.replace(
            kw,
            f"<span style='color:{color};font-weight:600;'>{kw}</span>",
        )
    return highlighted


def build_magic_prompt(magic: str) -> str:
    if magic == "Structured Markdown (510(k)-aware)":
        return (
            "You are an expert regulatory scribe.\n"
            "Convert the user's raw note into clean, well-structured Markdown suitable for a 510(k) file.\n"
            "Use clear headings, bullets, and tables. Do not invent new information.\n"
            f"Highlight 10–30 important keywords by wrapping them in "
            f"<span style='color:{CORAL_COLOR};font-weight:600;'>keyword</span>.\n"
        )
    if magic == "Regulatory Entity Table (20 entities)":
        return (
            "Extract exactly 20 key regulatory entities from the note, focusing on:\n"
            "- Device identity, indications, patient population\n"
            "- Predicate devices\n"
            "- Risk controls / mitigations\n"
            "- Types of testing and main outcomes\n"
            "- Key standards/guidances cited\n"
            "Return a Markdown table:\n"
            "| # | Entity | Category | Value | Notes |\n"
            "with 20 rows."
        )
    if magic == "Risk & Mitigation Matrix":
        return (
            "From the note, extract risks and mitigations and build a Markdown table:\n"
            "| Hazard | Failure Mode / Scenario | Harm | Mitigation / Control | Residual Risk |\n"
            "Add brief text before and after the table summarizing the overall risk picture."
        )
    if magic == "Testing Coverage Map":
        return (
            "From the note, infer what testing has been done (bench, biocomp, EMC/electrical, "
            "software, shelf life, packaging, clinical, etc.) and build:\n"
            "1) A summary narrative, and\n"
            "2) A Markdown table:\n"
            "| Test Category | Standard / Method | Status | Key Results | Gaps / Comments |"
        )
    if magic == "Action Items / Deficiency List":
        return (
            "Extract all action items, open questions, and potential deficiencies from the note.\n"
            "Return:\n"
            "- A bullet list of issues, and\n"
            "- A Markdown table:\n"
            "| ID | Issue / Deficiency | Impact | Suggested Action | Owner | Due Date |"
        )
    if magic == "AI Keywords (color-customizable)":
        return (
            "You will receive user-specified keywords and colors separately. "
            "Do NOT add color markup yourself. Just leave the text as-is; "
            "color will be applied by the client."
        )
    return "You are a helpful AI Note Keeper for 510(k) reviewers."


def note_keeper_tab():
    st.subheader("🧾 AI Note Keeper · With AI Magics")

    col_in, col_out = st.columns([2, 2])

    with col_in:
        st.markdown("**1. Input Note (text / markdown / file)**")
        note_file = st.file_uploader(
            "Upload note (PDF / TXT / MD)",
            type=["pdf", "txt", "md", "markdown"],
            key="note_file",
        )
        raw_text = st.text_area(
            "Or paste note here",
            height=220,
            key="note_raw_text",
        )

        if note_file and not raw_text.strip():
            if note_file.type == "application/pdf" or note_file.name.lower().endswith(".pdf"):
                reader = PdfReader(note_file)
                text_pages = []
                for i, page in enumerate(reader.pages):
                    try:
                        t = page.extract_text() or ""
                    except Exception:
                        t = ""
                    text_pages.append(f"\n\n--- Page {i+1} ---\n{t}")
                raw_text = "\n".join(text_pages)
            else:
                raw_text = note_file.getvalue().decode("utf-8", errors="ignore")

        if st.button("🔁 Load into editor"):
            st.session_state.note_content = raw_text

        st.markdown("**2. AI Magics**")
        magic = st.selectbox(
            "Choose an AI Magic",
            options=[
                "Structured Markdown (510(k)-aware)",
                "Regulatory Entity Table (20 entities)",
                "Risk & Mitigation Matrix",
                "Testing Coverage Map",
                "Action Items / Deficiency List",
                "AI Keywords (color-customizable)",
            ],
            key="note_magic",
        )

        provider = st.selectbox("Provider", list(AI_MODELS.keys()), key="note_provider")
        model = st.selectbox("Model", AI_MODELS[provider], key="note_model")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.3, key="note_temp")
        max_tokens = st.number_input(
            "Max Tokens",
            min_value=512,
            max_value=12000,
            value=4000,
            step=256,
            key="note_max_tokens",
        )

        if magic == "AI Keywords (color-customizable)":
            st.markdown(
                "Enter keywords and optional colors: `keyword#FF0000, another keyword#00FF00, third keyword`.\n"
                "Keywords without color use coral by default."
            )
            keyword_spec = st.text_input(
                "Keywords + colors",
                key="note_keyword_spec",
            )
            run_magic = st.button("🎨 Apply AI Keywords (on current note)")
        else:
            run_magic = st.button("✨ Run Selected AI Magic")

    with col_out:
        st.markdown("**3. Note Editor & Preview**")
        view_mode = st.radio(
            "View mode",
            options=["Edit (Markdown/Text)", "Preview (Markdown)"],
            horizontal=True,
            key="note_view_mode",
        )

        if view_mode == "Edit (Markdown/Text)":
            edited = st.text_area(
                "Working note (editable)",
                value=st.session_state.note_content,
                height=380,
                key="note_editor",
            )
            st.session_state.note_content = edited
        else:
            st.markdown(st.session_state.note_content, unsafe_allow_html=True)

        if run_magic:
            if magic == "AI Keywords (color-customizable)":
                # Client-side coloring
                base_text = st.session_state.note_content
                spec = st.session_state.get("note_keyword_spec", "")
                colored = highlight_keywords_colored(base_text, spec)
                st.session_state.note_content = colored
                st.success("Keywords highlighted with chosen colors.")
            else:
                if not st.session_state.note_content.strip():
                    st.error("Note is empty. Load or paste some content first.")
                else:
                    sys_prompt = build_magic_prompt(magic)
                    with st.spinner(f"Running Magic: {magic}…"):
                        try:
                            out = run_ad_hoc_llm(
                                provider=provider,
                                model=model,
                                system_prompt=sys_prompt,
                                user_prompt=st.session_state.note_content,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )
                            st.session_state.note_content = out
                            st.success("Magic applied. Check the editor/preview.")
                        except Exception as e:
                            st.error(f"Note Magic error: {e}")


# =========================
# 14. DASHBOARD TAB
# =========================

def dashboard_tab():
    st.subheader("📊 Interactive Analytics Dashboard")

    m = st.session_state.metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total AI Calls", m["total_runs"])
    c2.metric("Gemini Calls", m["provider_calls"]["gemini"])
    c3.metric("OpenAI Calls", m["provider_calls"]["openai"])
    c4.metric("Last Run Duration (s)", round(m["last_run_duration"], 2))

    st.markdown("#### Provider Usage")
    providers = list(m["provider_calls"].keys())
    values = list(m["provider_calls"].values())
    st.bar_chart({"providers": providers, "calls": values}, x="providers", y="calls")

    st.markdown("#### Execution Log Timeline")
    for log in reversed(st.session_state.execution_log[-50:]):
        style = {
            "info": "color:#5DADE2",
            "success": "color:#2ECC71",
            "error": "color:#E74C3C",
        }.get(log["type"], "")
        st.markdown(
            f"<span style='font-size:0.75rem;opacity:0.8;'>{log['time']}</span> "
            f"<span style='{style}'>{log['msg']}</span>",
            unsafe_allow_html=True,
        )


# =========================
# 15. AGENTS & SKILLS (EDIT / UPLOAD / DOWNLOAD)
# =========================

def agents_skills_tab():
    st.subheader("🤖 Agents & Skills Configuration")

    col_agents, col_skills = st.columns(2)

    with col_agents:
        st.markdown("### agents.yaml")
        path = "agents.yaml"
        current_yaml = ""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                current_yaml = f.read()
        yaml_text = st.text_area(
            "Edit agents.yaml",
            value=current_yaml,
            height=320,
            key="agents_yaml_editor",
        )
        col_a1, col_a2, col_a3 = st.columns(3)
        with col_a1:
            if st.button("💾 Save agents.yaml"):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(yaml_text)
                st.session_state.agents = load_agents_yaml(path)
                st.success("agents.yaml saved and reloaded.")
        with col_a2:
            if st.button("🔄 Reload from disk"):
                st.session_state.agents = load_agents_yaml(path)
                st.experimental_rerun()
        with col_a3:
            st.download_button(
                "⬇️ Download agents.yaml",
                data=yaml_text.encode("utf-8"),
                file_name="agents.yaml",
                mime="text/x-yaml",
            )
        uploaded_yaml = st.file_uploader("Upload new agents.yaml", type=["yaml", "yml"], key="upload_agents_yaml")
        if uploaded_yaml:
            txt = uploaded_yaml.getvalue().decode("utf-8", errors="ignore")
            with open(path, "w", encoding="utf-8") as f:
                f.write(txt)
            st.session_state.agents = load_agents_yaml(path)
            st.success("Uploaded agents.yaml and reloaded agents.")

    with col_skills:
        st.markdown("### SKILL.md")
        skill_path = "SKILL.md"
        skill_text = ""
        if os.path.exists(skill_path):
            with open(skill_path, "r", encoding="utf-8") as f:
                skill_text = f.read()
        skill_editor = st.text_area(
            "Edit SKILL.md",
            value=skill_text,
            height=320,
            key="skill_md_editor",
        )
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            if st.button("💾 Save SKILL.md"):
                with open(skill_path, "w", encoding="utf-8") as f:
                    f.write(skill_editor)
                st.success("SKILL.md saved.")
        with col_s2:
            st.download_button(
                "⬇️ Download SKILL.md",
                data=skill_editor.encode("utf-8"),
                file_name="SKILL.md",
                mime="text/markdown",
            )
        with col_s3:
            uploaded_skill = st.file_uploader("Upload SKILL.md", type=["md", "markdown"], key="upload_skill_md")
            if uploaded_skill:
                txt = uploaded_skill.getvalue().decode("utf-8", errors="ignore")
                with open(skill_path, "w", encoding="utf-8") as f:
                    f.write(txt)
                st.success("Uploaded and saved SKILL.md.")


# =========================
# 16. SETTINGS / LANGUAGE / THEME
# =========================

def settings_sidebar():
    app = st.session_state.app_state
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        lang = st.selectbox(
            "Language 語言",
            options=list(LANG_LABELS.keys()),
            format_func=lambda k: LANG_LABELS[k],
            index=list(LANG_LABELS.keys()).index(app.language),
        )
        mode = st.selectbox(
            "Theme Mode",
            options=list(THEME_MODE_LABELS.keys()),
            format_func=lambda k: THEME_MODE_LABELS[k],
            index=list(THEME_MODE_LABELS.keys()).index(app.theme_mode),
        )
        app.language = lang
        app.theme_mode = mode

        st.markdown("### 🎨 Painter Style")
        theme_ids = [t["id"] for t in PAINTER_THEMES]
        idx = theme_ids.index(app.current_theme_id) if app.current_theme_id in theme_ids else 0

        def label_func(i):
            t = PAINTER_THEMES[i]
            if lang == "en":
                return t["name_en"]
            return t["name_zh"]

        selected = st.selectbox(
            "Theme",
            options=list(range(len(PAINTER_THEMES))),
            index=idx,
            format_func=label_func,
        )
        app.current_theme_id = PAINTER_THEMES[selected]["id"]

        style_jackpot()

        st.markdown("---")
        api_key_input_ui()


# =========================
# 17. MAIN APP
# =========================

def main():
    st.set_page_config(
        page_title="FDA 510(k) Review Studio · Painter Edition",
        page_icon="🌸",
        layout="wide",
    )
    init_session_state()
    inject_global_css()
    settings_sidebar()
    wow_header()
    wow_status_bar()

    tabs = st.tabs(
        [
            "📑 510(k) Summary",
            "📘 Review Guidance",
            "📂 Submission OCR & Agents",
            "📝 Review Report",
            "🔗 Agent Pipeline",
            "🧾 AI Note Keeper",
            "🤖 Agents & Skills",
            "📊 Dashboard",
        ]
    )
    with tabs[0]:
        summary_tab()
    with tabs[1]:
        guidance_tab()
    with tabs[2]:
        ocr_agents_tab()
    with tabs[3]:
        review_report_tab()
    with tabs[4]:
        pipeline_tab()
    with tabs[5]:
        note_keeper_tab()
    with tabs[6]:
        agents_skills_tab()
    with tabs[7]:
        dashboard_tab()


if __name__ == "__main__":
    main()
