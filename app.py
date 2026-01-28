from __future__ import annotations

import base64
import concurrent.futures as cf
import contextlib
import dataclasses
import datetime as dt
import difflib
import io
import json
import math
import os
import random
import re
import shutil
import string
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import streamlit as st
import yaml

# Optional deps (handle gracefully)
with contextlib.suppress(Exception):
    from pypdf import PdfReader, PdfWriter

with contextlib.suppress(Exception):
    from pdf2image import convert_from_bytes

with contextlib.suppress(Exception):
    import pytesseract

# Providers (handle gracefully if missing)
with contextlib.suppress(Exception):
    from openai import OpenAI

with contextlib.suppress(Exception):
    import google.generativeai as genai


# =========================
# WOW UI: i18n + Style packs
# =========================

I18N = {
    "en": {
        "app_title": "OPAL — Agentic AI Document Intelligence",
        "sidebar_global": "Global Settings",
        "theme": "Theme",
        "light": "Light",
        "dark": "Dark",
        "language": "Language",
        "style_pack": "Painter Style Pack",
        "jackpot": "Jackpot (Random Style)",
        "provider": "Provider",
        "provider_help": "Select which LLM provider to use for text/vision calls.",
        "local_only": "Local-only mode (no cloud calls)",
        "local_only_help": "Disables Vision OCR and cloud summarization/agents. Useful for sensitive docs.",
        "api_keys": "API Keys",
        "openai_key": "OpenAI API Key",
        "google_key": "Google (Gemini) API Key",
        "anthropic_key": "Anthropic API Key",
        "xai_key": "XAI (Grok) API Key",
        "using_env": "System Default (env secret)",
        "using_user": "User Override (session only)",
        "missing": "Missing",
        "model": "Model",
        "temperature": "Temperature",
        "max_tokens": "Max tokens",
        "max_tokens_help": "Default 12000. Adjust down if you hit context limits.",
        "keywords": "Keyword Highlighting",
        "default_keywords": "Default regulatory keywords",
        "custom_keywords": "Custom keywords (one per line)",
        "advanced": "Advanced",
        "ocr_dpi": "OCR DPI",
        "concurrency": "Concurrency",
        "backoff_max": "Max backoff (seconds)",
        "tabs_dashboard": "Dashboard",
        "tabs_ingest": "Ingest & Preview",
        "tabs_trim_ocr": "Trim & OCR",
        "tabs_workspace": "Markdown Workspace",
        "tabs_agents": "Agents",
        "tabs_toc": "ToC Pipeline",
        "tabs_agent_mgmt": "Agent Management",
        "tabs_notes": "AI Note Keeper",
        "active_doc": "Active Document",
        "set_active": "Set as active document",
        "upload_files": "Upload files (.pdf, .txt, .md)",
        "paste_text": "Or paste text/markdown",
        "pdf_preview": "PDF Preview",
        "page_range": "Page range (e.g., 1-5, 10, 15-20)",
        "compute_pages": "Computed pages",
        "ocr_method": "OCR method",
        "ocr_local": "Local OCR (Tesseract)",
        "ocr_vision": "Vision LLM OCR → Markdown",
        "ocr_hybrid": "Hybrid (Local OCR + LLM cleanup)",
        "run_trim_ocr": "Run Trim & OCR",
        "raw_text": "Raw extracted text",
        "md_transform": "Transformed Markdown",
        "md_editor": "Markdown editor (source)",
        "md_preview": "Rendered preview",
        "toggle_highlight": "Highlight keywords in preview",
        "download_md": "Download .md",
        "download_txt": "Download .txt",
        "agents_select": "Select agent",
        "agent_details": "Agent details",
        "system_prompt": "System prompt",
        "skills": "Skills",
        "run_agent": "Run Agent",
        "user_instruction": "Optional instruction (what do you want?)",
        "output": "Output",
        "history": "Run history",
        "restore": "Restore",
        "chain_runner": "Chain Runner",
        "chain_mode": "Chaining mode",
        "chain_append": "Append previous output as extra context",
        "chain_replace": "Replace input with previous output",
        "run_chain": "Run chain",
        "toc_base_dir": "Base directory (relative path allowed)",
        "run_toc": "Run ToC pipeline",
        "toc_preview": "Generated ToC.md",
        "download_toc": "Download ToC.md",
        "upload_agents": "Upload agents.yaml",
        "upload_skill": "Upload SKILL.md",
        "align_schema": "Align schema",
        "validation": "Validation",
        "diff": "Diff",
        "notes_input": "Paste note text/markdown, or upload pdf/txt/md",
        "notes_organize": "Organize note → Markdown",
        "notes_editor": "Note editor (Markdown/Text)",
        "ai_magics": "AI Magics",
        "magic_keywords": "AI Keywords (choose color per keyword)",
        "run_magic": "Run Magic",
        "run_log": "Run Log",
        "cleanup_tmp": "Cleanup temporary artifacts",
        "self_checks": "Self-checks",
    },
    "zh-TW": {
        "app_title": "OPAL — 代理式 AI 文件智慧系統",
        "sidebar_global": "全域設定",
        "theme": "主題",
        "light": "淺色",
        "dark": "深色",
        "language": "語言",
        "style_pack": "名畫家風格包",
        "jackpot": "Jackpot（隨機風格）",
        "provider": "供應商",
        "provider_help": "選擇用於文字/視覺呼叫的 LLM 供應商。",
        "local_only": "僅本機模式（不進行雲端呼叫）",
        "local_only_help": "停用 Vision OCR 與雲端摘要/代理。適合敏感文件。",
        "api_keys": "API 金鑰",
        "openai_key": "OpenAI API Key",
        "google_key": "Google（Gemini）API Key",
        "anthropic_key": "Anthropic API Key",
        "xai_key": "XAI（Grok）API Key",
        "using_env": "系統預設（環境變數秘密）",
        "using_user": "使用者覆寫（僅本次 Session）",
        "missing": "缺少",
        "model": "模型",
        "temperature": "溫度",
        "max_tokens": "最大 tokens",
        "max_tokens_help": "預設 12000。若遇到上下文限制可調低。",
        "keywords": "關鍵字高亮",
        "default_keywords": "預設法規/臨床關鍵字",
        "custom_keywords": "自訂關鍵字（每行一個）",
        "advanced": "進階",
        "ocr_dpi": "OCR DPI",
        "concurrency": "併發數",
        "backoff_max": "最大退避（秒）",
        "tabs_dashboard": "儀表板",
        "tabs_ingest": "匯入與預覽",
        "tabs_trim_ocr": "裁切與 OCR",
        "tabs_workspace": "Markdown 工作區",
        "tabs_agents": "代理（Agents）",
        "tabs_toc": "目錄管線（ToC）",
        "tabs_agent_mgmt": "代理管理",
        "tabs_notes": "AI 筆記管家",
        "active_doc": "目前文件",
        "set_active": "設為目前文件",
        "upload_files": "上傳檔案（.pdf, .txt, .md）",
        "paste_text": "或貼上文字/Markdown",
        "pdf_preview": "PDF 預覽",
        "page_range": "頁碼範圍（例如 1-5, 10, 15-20）",
        "compute_pages": "計算後頁碼",
        "ocr_method": "OCR 方法",
        "ocr_local": "本機 OCR（Tesseract）",
        "ocr_vision": "Vision LLM OCR → Markdown",
        "ocr_hybrid": "混合（本機 OCR + LLM 清理）",
        "run_trim_ocr": "執行裁切與 OCR",
        "raw_text": "原始擷取文字",
        "md_transform": "轉換後 Markdown",
        "md_editor": "Markdown 編輯器（原文）",
        "md_preview": "渲染預覽",
        "toggle_highlight": "在預覽中高亮關鍵字",
        "download_md": "下載 .md",
        "download_txt": "下載 .txt",
        "agents_select": "選擇代理",
        "agent_details": "代理資訊",
        "system_prompt": "System prompt",
        "skills": "Skills",
        "run_agent": "執行代理",
        "user_instruction": "選填指令（你想要什麼？）",
        "output": "輸出",
        "history": "執行歷史",
        "restore": "還原",
        "chain_runner": "鏈式執行器",
        "chain_mode": "鏈式模式",
        "chain_append": "把前一個輸出附加為額外上下文",
        "chain_replace": "用前一個輸出取代輸入",
        "run_chain": "執行鏈",
        "toc_base_dir": "基底資料夾（可用相對路徑）",
        "run_toc": "執行 ToC 管線",
        "toc_preview": "產生的 ToC.md",
        "download_toc": "下載 ToC.md",
        "upload_agents": "上傳 agents.yaml",
        "upload_skill": "上傳 SKILL.md",
        "align_schema": "對齊 schema",
        "validation": "驗證",
        "diff": "差異（diff）",
        "notes_input": "貼上筆記文字/Markdown，或上傳 pdf/txt/md",
        "notes_organize": "整理筆記 → Markdown",
        "notes_editor": "筆記編輯器（Markdown/文字）",
        "ai_magics": "AI 魔法",
        "magic_keywords": "AI Keywords（可為每個關鍵字選顏色）",
        "run_magic": "施放魔法",
        "run_log": "執行紀錄",
        "cleanup_tmp": "清除暫存檔",
        "self_checks": "自我檢查",
    },
}

PAINTER_STYLES = [
    ("Monet", "Impressionist softness, airy gradients, gentle watercolor-like UI"),
    ("Van Gogh", "Bold strokes, vibrant contrast, energetic accents"),
    ("Picasso", "Geometric blocks, sharp edges, modernist minimalism"),
    ("Da Vinci", "Classical balance, parchment tones, subtle ornament"),
    ("Rembrandt", "Chiaroscuro, deep shadows, warm highlights"),
    ("Hokusai", "Clean waves, indigo accents, crisp whitespace"),
    ("Klimt", "Gold accents, ornamental patterns, luxurious panels"),
    ("Matisse", "Cutout colors, playful composition, bright sections"),
    ("Pollock", "Dynamic splatter accents, lively separators"),
    ("Rothko", "Large calm color fields, meditative spacing"),
    ("Vermeer", "Soft light, quiet elegance, pearl-toned surfaces"),
    ("Magritte", "Surreal clarity, witty micro-interactions (visual cues)"),
    ("Cézanne", "Structured forms, earthy palette, balanced grids"),
    ("Dali", "Dreamlike gradients, fluid shapes, subtle distortions"),
    ("O’Keeffe", "Organic curves, desert palette, clean typography"),
    ("Turner", "Misty luminance, sea-sky gradients, delicate separators"),
    ("Kandinsky", "Abstract geometry, colored nodes, dashboard-like cues"),
    ("Basquiat", "Raw annotations, bold typographic highlights"),
    ("Seurat", "Pointillist dots, subtle texture, precise alignment"),
    ("Ukiyo-e", "Flat color planes, elegant linework, minimalist nav"),
]


# =========================
# Defaults: agents.yaml + SKILL.md (used if files missing)
# =========================

DEFAULT_AGENTS_YAML = """\
agents:
  - name: "Regulatory Analyst"
    id: "reg_analyst"
    role: "Regulatory logic and structured reasoning"
    default_model: "gemini-2.5-flash"
    temperature: 0.2
    max_tokens: 12000
    skills: ["citation_style", "executive_brief", "data_quality"]
    system_prompt: |
      You are an expert regulatory affairs specialist.
      Parse complex documents and extract actionable, auditable insights.
      Do not invent facts. If uncertain, say so.

  - name: "Safety Signal Detector"
    id: "safety_signal_detector"
    role: "Pharmacovigilance & post-market safety signal detection"
    default_model: "gemini-2.5-flash"
    temperature: 0.2
    max_tokens: 12000
    skills: ["citation_style", "executive_brief", "data_quality"]
    system_prompt: |
      You detect potential safety signals and rank risks.
      Always cite the source (filename + page) when possible.
      Do not guess. If evidence is weak, say so and ask for missing pages/sections.

  - name: "ToC Summarizer"
    id: "toc_summarizer"
    role: "Fast document triage summarizer for directory ToC pipeline"
    default_model: "gemini-2.5-flash"
    temperature: 0.2
    max_tokens: 1200
    skills: ["data_quality"]
    system_prompt: |
      Summarize the provided first-page text into ~100 words.
      Focus on what the document is, who/what it concerns, dates, and any safety/regulatory signals.
      Do not invent details not present.

  - name: "Schema Alignment Agent"
    id: "schema_alignment_agent"
    role: "Deterministic schema aligner for agents.yaml"
    default_model: "gpt-4o-mini"
    temperature: 0.0
    max_tokens: 2000
    skills: ["json_mode", "data_quality"]
    system_prompt: |
      You align a user-provided agents YAML into the standard schema.
      Be deterministic and conservative. Never create new agent intents, only map/normalize fields.

  - name: "Markdown Rewriter"
    id: "markdown_rewriter"
    role: "Non-hallucinatory Markdown cleanup and reconstruction"
    default_model: "gemini-2.5-flash"
    temperature: 0.1
    max_tokens: 12000
    skills: ["data_quality"]
    system_prompt: |
      Convert the given raw/OCR text into clean Markdown.
      Do not change meaning. Do not add facts. Mark uncertain text as [illegible] if needed.
      Prefer plain text over inventing tables if structure is ambiguous.
"""

DEFAULT_SKILL_MD = """\
## skill: citation_style
When referencing extracted information, cite as: (Source: <filename>, p.<page>).
If page is not known, cite as: (Source: <filename>, page unknown).
Never fabricate page numbers.

## skill: executive_brief
Provide:
1) Executive summary (3–6 bullets)
2) Key findings (grouped)
3) Recommended actions
4) Open questions / missing evidence

## skill: data_quality
Flag OCR uncertainty, missing fields, suspicious numbers, and potential extraction errors.
If tables look malformed, mention it and propose what pages/sections to re-OCR.

## skill: json_mode
If asked to output JSON, output:
- One fenced code block with json
- No trailing commentary outside the JSON block
"""

DEFAULT_REGULATORY_KEYWORDS = [
    "Class I", "Class II", "Class III", "510(k)", "PMA",
    "Contraindication", "Warning", "Caution",
    "Adverse Event", "Serious Injury", "Death", "Malfunction",
    "Recall", "Field Safety Notice", "Post-market", "MDR", "Vigilance",
]


# =========================
# Utilities
# =========================

def t(lang: str, key: str) -> str:
    return I18N.get(lang, I18N["en"]).get(key, key)

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "agent"

def now_ts() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars/token in English; degrade gracefully for mixed text.
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))

def mask_key(key: Optional[str], show_last4: bool = True) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "●" * len(key)
    if show_last4:
        return "●" * (len(key) - 4) + key[-4:]
    return "●" * len(key)

def scrub_secrets(msg: str, secrets: List[str]) -> str:
    out = msg
    for s in secrets:
        if s:
            out = out.replace(s, "●●●●●")
    return out

def ensure_tmp_dir() -> Path:
    sid = st.session_state.get("session_id")
    if not sid:
        sid = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(10))
        st.session_state["session_id"] = sid
    p = Path("/tmp") / f"opal_{sid}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def cleanup_tmp_dir() -> None:
    p = Path("/tmp") / f"opal_{st.session_state.get('session_id', '')}"
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)

def add_log(message: str, level: str = "INFO") -> None:
    st.session_state.setdefault("run_log", [])
    st.session_state["run_log"].append(f"[{now_ts()}] {level}: {message}")

def show_run_log(lang: str) -> None:
    with st.expander(t(lang, "run_log"), expanded=False):
        logs = st.session_state.get("run_log", [])
        if not logs:
            st.caption("—")
        else:
            st.code("\n".join(logs), language="text")


# =========================
# WOW Theme CSS injection
# =========================

def inject_wow_css(theme: str, painter_style: str) -> None:
    # Streamlit doesn't fully support runtime theme switching; we overlay a “WOW skin”.
    base_bg = "#0b1220" if theme == "dark" else "#f7f8fb"
    panel_bg = "#101a33" if theme == "dark" else "#ffffff"
    text = "#e7eefc" if theme == "dark" else "#111827"
    muted = "#9bb0d0" if theme == "dark" else "#6b7280"
    coral = "#ff7f50"

    # Slight style variations by painter (subtle, stable)
    seed = sum(ord(c) for c in painter_style)
    random.seed(seed)
    accent = random.choice(["#7c3aed", "#2563eb", "#db2777", "#0891b2", "#16a34a", "#f59e0b", coral])
    accent2 = random.choice(["#22c55e", "#38bdf8", "#fb7185", "#a78bfa", coral])

    texture = {
        "Monet": "radial-gradient(1200px 600px at 10% 0%, rgba(37,99,235,0.18), transparent 55%), radial-gradient(900px 500px at 90% 20%, rgba(255,127,80,0.14), transparent 50%)",
        "Van Gogh": "radial-gradient(1000px 600px at 0% 30%, rgba(245,158,11,0.20), transparent 55%), radial-gradient(900px 500px at 85% 10%, rgba(124,58,237,0.18), transparent 50%)",
        "Klimt": "radial-gradient(1000px 600px at 50% 0%, rgba(245,158,11,0.22), transparent 55%), radial-gradient(800px 500px at 90% 40%, rgba(255,127,80,0.14), transparent 50%)",
        "Hokusai": "radial-gradient(900px 600px at 15% 0%, rgba(37,99,235,0.22), transparent 55%), radial-gradient(800px 500px at 85% 10%, rgba(56,189,248,0.14), transparent 50%)",
        "Rothko": "linear-gradient(180deg, rgba(124,58,237,0.18), transparent 40%), linear-gradient(0deg, rgba(255,127,80,0.10), transparent 45%)",
    }.get(painter_style, "radial-gradient(1000px 600px at 12% 0%, rgba(124,58,237,0.16), transparent 55%), radial-gradient(900px 500px at 90% 20%, rgba(255,127,80,0.12), transparent 50%)")

    st.markdown(
        f"""
<style>
:root {{
  --opal-bg: {base_bg};
  --opal-panel: {panel_bg};
  --opal-text: {text};
  --opal-muted: {muted};
  --opal-accent: {accent};
  --opal-accent2: {accent2};
  --opal-coral: {coral};
}}

.stApp {{
  background: var(--opal-bg);
  color: var(--opal-text);
}}

.stApp::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background: {texture};
  opacity: 1;
  z-index: 0;
}}

[data-testid="stHeader"], [data-testid="stToolbar"] {{
  background: transparent !important;
}}

[data-testid="stSidebar"] > div:first-child {{
  background: color-mix(in srgb, var(--opal-panel) 92%, transparent) !important;
  border-right: 1px solid color-mix(in srgb, var(--opal-accent) 18%, transparent);
}}

.block-container {{
  position: relative;
  z-index: 1;
}}

.opal-card {{
  background: color-mix(in srgb, var(--opal-panel) 96%, transparent);
  border: 1px solid color-mix(in srgb, var(--opal-accent) 20%, transparent);
  border-radius: 14px;
  padding: 14px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.10);
}}

.opal-badge {{
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--opal-accent) 35%, transparent);
  background: color-mix(in srgb, var(--opal-accent) 14%, transparent);
  color: var(--opal-text);
  font-size: 12px;
  line-height: 20px;
}}

.opal-status-ok {{
  border-color: color-mix(in srgb, #22c55e 45%, transparent) !important;
  background: color-mix(in srgb, #22c55e 12%, transparent) !important;
}}

.opal-status-warn {{
  border-color: color-mix(in srgb, #f59e0b 55%, transparent) !important;
  background: color-mix(in srgb, #f59e0b 12%, transparent) !important;
}}

.opal-status-bad {{
  border-color: color-mix(in srgb, #ef4444 55%, transparent) !important;
  background: color-mix(in srgb, #ef4444 12%, transparent) !important;
}}

a {{
  color: var(--opal-accent2) !important;
}}

hr {{
  border-color: color-mix(in srgb, var(--opal-accent) 18%, transparent) !important;
}}

.opal-small {{
  color: var(--opal-muted);
  font-size: 12px;
}}

</style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Providers: unified client (Gemini + OpenAI + stubs)
# =========================

@dataclass
class LLMUsage:
    approx_input_tokens: int = 0
    approx_output_tokens: int = 0
    elapsed_s: float = 0.0
    provider: str = ""
    model: str = ""
    error: Optional[str] = None

class LLMClient:
    def __init__(self, provider: str, api_key: str, backoff_max_s: int = 40):
        self.provider = provider
        self.api_key = api_key
        self.backoff_max_s = backoff_max_s

        if provider == "openai":
            if "OpenAI" not in globals():
                raise RuntimeError("OpenAI SDK not installed.")
            self._client = OpenAI(api_key=api_key)
        elif provider == "gemini":
            if "genai" not in globals():
                raise RuntimeError("google-generativeai not installed.")
            genai.configure(api_key=api_key)
            self._client = genai
        elif provider in ("anthropic", "xai"):
            # Stubs; you can add real SDK wiring later.
            raise RuntimeError(f"Provider '{provider}' not wired in this single-file build.")
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _retry(self, fn, *, secrets_to_scrub: List[str], action_name: str) -> Tuple[Any, LLMUsage]:
        start = time.time()
        attempt = 0
        last_err = None
        while True:
            attempt += 1
            try:
                result = fn()
                usage = LLMUsage(elapsed_s=time.time() - start)
                return result, usage
            except Exception as e:
                msg = scrub_secrets(str(e), secrets_to_scrub)
                last_err = msg
                # naive detection
                is_rate = ("429" in msg) or ("rate" in msg.lower()) or ("quota" in msg.lower())
                is_transient = is_rate or ("timeout" in msg.lower()) or ("temporarily" in msg.lower()) or ("503" in msg)
                if attempt >= 6 or not is_transient:
                    usage = LLMUsage(elapsed_s=time.time() - start, error=last_err)
                    return None, usage
                sleep_s = min(self.backoff_max_s, (2 ** (attempt - 1)) + random.random())
                add_log(f"{action_name} transient error (attempt {attempt}): {msg} — sleeping {sleep_s:.1f}s", "WARN")
                time.sleep(sleep_s)

    def generate_text(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        secrets_to_scrub: List[str],
    ) -> Tuple[str, LLMUsage]:
        approx_in = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)

        if self.provider == "openai":
            def _call():
                # Uses Responses API (OpenAI python >= 1.0)
                resp = self._client.responses.create(
                    model=model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                # Consolidate text output
                out = ""
                for item in resp.output:
                    if getattr(item, "type", None) == "message":
                        for c in item.content:
                            if getattr(c, "type", None) == "output_text":
                                out += c.text
                return out.strip()

            out, usage = self._retry(_call, secrets_to_scrub=secrets_to_scrub, action_name="OpenAI text")
            if usage:
                usage.provider, usage.model = self.provider, model
                usage.approx_input_tokens = approx_in
                usage.approx_output_tokens = estimate_tokens(out or "")
            if out is None:
                raise RuntimeError(usage.error or "OpenAI call failed.")
            return out, usage

        if self.provider == "gemini":
            def _call():
                m = self._client.GenerativeModel(model)
                resp = m.generate_content(
                    [
                        {"role": "user", "parts": [
                            {"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}
                        ]}
                    ],
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return (resp.text or "").strip()

            out, usage = self._retry(_call, secrets_to_scrub=secrets_to_scrub, action_name="Gemini text")
            if usage:
                usage.provider, usage.model = self.provider, model
                usage.approx_input_tokens = approx_in
                usage.approx_output_tokens = estimate_tokens(out or "")
            if out is None:
                raise RuntimeError(usage.error or "Gemini call failed.")
            return out, usage

        raise RuntimeError("Unsupported provider in generate_text")

    def generate_vision_markdown(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        images: List[bytes],
        temperature: float,
        max_tokens: int,
        secrets_to_scrub: List[str],
    ) -> Tuple[str, LLMUsage]:
        approx_in = estimate_tokens(system_prompt) + estimate_tokens(user_prompt) + sum(max(1, len(b) // 4000) for b in images)

        if self.provider == "openai":
            def _call():
                content: List[Dict[str, Any]] = [{"type": "input_text", "text": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
                for b in images:
                    b64 = base64.b64encode(b).decode("utf-8")
                    content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64}"})
                resp = self._client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": content}],
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                out = ""
                for item in resp.output:
                    if getattr(item, "type", None) == "message":
                        for c in item.content:
                            if getattr(c, "type", None) == "output_text":
                                out += c.text
                return out.strip()

            out, usage = self._retry(_call, secrets_to_scrub=secrets_to_scrub, action_name="OpenAI vision")
            if usage:
                usage.provider, usage.model = self.provider, model
                usage.approx_input_tokens = approx_in
                usage.approx_output_tokens = estimate_tokens(out or "")
            if out is None:
                raise RuntimeError(usage.error or "OpenAI vision call failed.")
            return out, usage

        if self.provider == "gemini":
            def _call():
                m = self._client.GenerativeModel(model)
                parts: List[Dict[str, Any]] = [{"text": f"{system_prompt}\n\n---\n\n{user_prompt}"}]
                for b in images:
                    parts.append({"inline_data": {"mime_type": "image/png", "data": b}})
                resp = m.generate_content(
                    [{"role": "user", "parts": parts}],
                    generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
                )
                return (resp.text or "").strip()

            out, usage = self._retry(_call, secrets_to_scrub=secrets_to_scrub, action_name="Gemini vision")
            if usage:
                usage.provider, usage.model = self.provider, model
                usage.approx_input_tokens = approx_in
                usage.approx_output_tokens = estimate_tokens(out or "")
            if out is None:
                raise RuntimeError(usage.error or "Gemini vision call failed.")
            return out, usage

        raise RuntimeError("Unsupported provider in generate_vision_markdown")


# =========================
# Agents + Skills
# =========================

REQUIRED_AGENT_FIELDS = ["id", "name", "role", "default_model", "temperature", "max_tokens", "skills", "system_prompt"]

def load_text_file_or_default(path: Path, default_text: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    return default_text

def parse_skills(skill_md: str) -> Dict[str, str]:
    # Delimiter format: ## skill: name
    skills: Dict[str, str] = {}
    pattern = r"^##\s*skill:\s*([a-zA-Z0-9_\-]+)\s*$"
    lines = skill_md.splitlines()
    current = None
    buf: List[str] = []
    for line in lines:
        m = re.match(pattern, line.strip())
        if m:
            if current:
                skills[current] = "\n".join(buf).strip()
            current = m.group(1).strip()
            buf = []
        else:
            if current is not None:
                buf.append(line)
    if current:
        skills[current] = "\n".join(buf).strip()
    return skills

def align_agents_schema(raw: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Deterministic alignment: map near-equivalents, fill defaults, generate stable IDs.
    Returns: aligned_yaml_dict, warnings
    """
    warnings: List[str] = []
    agents = raw.get("agents")
    if agents is None and isinstance(raw, dict):
        # sometimes user provides list at root
        if isinstance(raw, list):
            agents = raw
        else:
            # try common alternative keys
            for k in ("agent", "items", "definitions"):
                if k in raw:
                    agents = raw[k]
                    warnings.append(f"Mapped root key '{k}' to 'agents'.")
                    break

    if not isinstance(agents, list):
        return {"agents": []}, ["No agents list found; created empty schema."]

    aligned: List[Dict[str, Any]] = []
    for a in agents:
        if not isinstance(a, dict):
            warnings.append("Skipped non-dict agent entry.")
            continue

        # map equivalents
        name = a.get("name") or a.get("title") or a.get("agent_name") or "Untitled Agent"
        aid = a.get("id") or slugify(name)
        role = a.get("role") or a.get("persona") or a.get("description") or "General agent"
        system_prompt = a.get("system_prompt") or a.get("prompt") or a.get("system") or a.get("instructions") or ""
        default_model = a.get("default_model") or a.get("model") or "gemini-2.5-flash"
        temperature = a.get("temperature")
        if temperature is None:
            temperature = 0.2
        max_tokens = a.get("max_tokens") or a.get("max_output_tokens") or 12000
        skills = a.get("skills") or []
        if isinstance(skills, str):
            # comma or newline separated
            skills = [s.strip() for s in re.split(r"[,\n]+", skills) if s.strip()]

        aligned.append({
            "name": str(name),
            "id": str(aid),
            "role": str(role),
            "default_model": str(default_model),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "skills": list(skills),
            "system_prompt": str(system_prompt),
        })

    return {"agents": aligned}, warnings

def validate_agents_schema(doc: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    agents = doc.get("agents")
    if not isinstance(agents, list) or not agents:
        errs.append("agents must be a non-empty list.")
        return errs
    ids = set()
    for i, a in enumerate(agents):
        if not isinstance(a, dict):
            errs.append(f"agent[{i}] must be a dict.")
            continue
        for f in REQUIRED_AGENT_FIELDS:
            if f not in a:
                errs.append(f"agent[{i}] missing field: {f}")
        aid = a.get("id")
        if aid in ids:
            errs.append(f"duplicate agent id: {aid}")
        ids.add(aid)
    return errs

def get_agent_by_id(agents_doc: Dict[str, Any], agent_id: str) -> Optional[Dict[str, Any]]:
    for a in agents_doc.get("agents", []):
        if a.get("id") == agent_id:
            return a
    return None

def assemble_system_prompt(
    agent: Dict[str, Any],
    skill_library: Dict[str, str],
    global_safety: str,
) -> str:
    chunks = []
    chunks.append(agent.get("system_prompt", "").strip())
    for sk in agent.get("skills", []) or []:
        snippet = skill_library.get(sk)
        if snippet:
            chunks.append(f"\n\n[SKILL:{sk}]\n{snippet}".strip())
    chunks.append(global_safety.strip())
    return "\n\n---\n\n".join([c for c in chunks if c])

GLOBAL_SAFETY_RULES = """\
Global safety rules:
- Do NOT fabricate facts. If the document does not contain the information, say so.
- Prefer quoting/pointing to exact excerpts.
- If citations (page numbers) are unknown, explicitly say "page unknown".
- If OCR seems uncertain, flag it clearly.
- Output in Markdown unless the user explicitly requests JSON (then include one fenced JSON code block).
"""


# =========================
# Keyword highlighting (Markdown-safe-ish)
# =========================

FENCED_CODE_RE = re.compile(r"(^```.*?$.*?^```$)", re.MULTILINE | re.DOTALL)

def highlight_keywords_markdown_safe(md: str, keywords: List[str], color: str = "coral") -> str:
    """
    Best-effort highlighting that avoids fenced code blocks and inline code.
    Also tries to avoid mangling markdown links by not touching link URLs.
    """
    if not md or not keywords:
        return md

    # Sort by length desc to avoid partial overlaps
    kws = sorted({k for k in keywords if k.strip()}, key=len, reverse=True)
    if not kws:
        return md

    def highlight_segment(seg: str) -> str:
        # Skip inline code portions split by backticks
        parts = seg.split("`")
        for i in range(0, len(parts), 2):  # only non-code
            text = parts[i]

            # Avoid highlighting inside link targets: (... ) right after ](
            # We'll temporarily mask link targets.
            link_targets: List[str] = []
            def _mask(m):
                link_targets.append(m.group(0))
                return f"@@LINKTARGET{len(link_targets)-1}@@"
            text = re.sub(r"\]\([^)]+\)", _mask, text)

            # Apply keyword highlighting
            for kw in kws:
                # word-boundary-ish, but allow symbols like 510(k)
                pattern = re.compile(re.escape(kw), re.IGNORECASE)
                text = pattern.sub(lambda m: f'<span style="color:{color}">{m.group(0)}</span>', text)

            # Unmask link targets
            for idx, original in enumerate(link_targets):
                text = text.replace(f"@@LINKTARGET{idx}@@", original)

            parts[i] = text
        return "`".join(parts)

    chunks = FENCED_CODE_RE.split(md)
    out = []
    for c in chunks:
        if c.startswith("```") and c.rstrip().endswith("```"):
            out.append(c)
        else:
            out.append(highlight_segment(c))
    return "".join(out)


# =========================
# PDF processing + OCR
# =========================

def parse_page_ranges(rng: str, max_page: int) -> List[int]:
    """
    Input examples: "1-5, 10, 15-20"
    Returns 1-based page numbers sorted unique.
    """
    if not rng or not rng.strip():
        raise ValueError("Empty page range.")
    rng = rng.replace(" ", "")
    items = [x for x in rng.split(",") if x]
    pages: set[int] = set()
    for it in items:
        if "-" in it:
            a, b = it.split("-", 1)
            if not a.isdigit() or not b.isdigit():
                raise ValueError(f"Invalid range token: {it}")
            start, end = int(a), int(b)
            if start <= 0 or end <= 0:
                raise ValueError("Page numbers must be >= 1.")
            if start > end:
                raise ValueError(f"Invalid range (start > end): {it}")
            for p in range(start, end + 1):
                pages.add(p)
        else:
            if not it.isdigit():
                raise ValueError(f"Invalid page token: {it}")
            p = int(it)
            if p <= 0:
                raise ValueError("Page numbers must be >= 1.")
            pages.add(p)

    out = sorted(pages)
    if not out:
        raise ValueError("No pages selected.")
    if out[0] < 1 or out[-1] > max_page:
        raise ValueError(f"Selected pages out of bounds. Document has {max_page} pages.")
    return out

def pdf_page_count(pdf_bytes: bytes) -> int:
    if "PdfReader" not in globals():
        raise RuntimeError("pypdf not installed.")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return len(reader.pages)

def extract_pdf_pages(pdf_bytes: bytes, pages_1based: List[int]) -> bytes:
    if "PdfReader" not in globals():
        raise RuntimeError("pypdf not installed.")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for p in pages_1based:
        writer.add_page(reader.pages[p - 1])
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()

def extract_pdf_text_pages(pdf_bytes: bytes, pages_1based: List[int]) -> str:
    if "PdfReader" not in globals():
        raise RuntimeError("pypdf not installed.")
    reader = PdfReader(io.BytesIO(pdf_bytes))
    texts = []
    for p in pages_1based:
        try:
            texts.append(reader.pages[p - 1].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts).strip()

def render_pdf_pages_to_png_bytes(pdf_bytes: bytes, pages_1based: List[int], dpi: int) -> List[bytes]:
    if "convert_from_bytes" not in globals():
        raise RuntimeError("pdf2image not installed or poppler missing.")
    # pdf2image uses 1-based "first_page"/"last_page" and renders ranges; we render per page for control.
    images_bytes: List[bytes] = []
    for p in pages_1based:
        pil_images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=p, last_page=p)
        if not pil_images:
            continue
        img = pil_images[0]
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        images_bytes.append(buf.getvalue())
    return images_bytes

def local_ocr_images(images_png: List[bytes]) -> str:
    if "pytesseract" not in globals():
        raise RuntimeError("pytesseract not installed.")
    from PIL import Image  # pillow required

    texts = []
    for b in images_png:
        img = Image.open(io.BytesIO(b))
        txt = pytesseract.image_to_string(img)
        texts.append(txt)
    return "\n\n".join(texts).strip()

VISION_OCR_SYSTEM = """\
You are an OCR engine. Transcribe exactly what you see.
Rules:
- Output ONLY valid Markdown.
- Preserve headings/lists/table structure when clear.
- Do NOT add commentary or explanations.
- Do NOT guess unreadable text; use [illegible].
- Do NOT invent missing lines.
"""

VISION_OCR_USER = """\
OCR the provided page image(s) into Markdown.
Preserve tables as Markdown tables when possible; otherwise keep as aligned text blocks.
Return only Markdown.
"""

MD_REWRITE_SYSTEM = """\
You convert raw/OCR text into clean, audit-friendly Markdown.

Rules:
- Do not change meaning.
- Do not add facts.
- Fix hyphenation across line breaks, whitespace, headings, and list formatting.
- Only create Markdown tables when structure is obvious; otherwise keep plain text.
- If text is ambiguous or broken, keep it as-is and annotate minimally with [uncertain].
Output only Markdown.
"""

def local_markdown_cleanup(text: str) -> str:
    # Conservative cleanup: normalize whitespace and obvious hyphenation.
    if not text:
        return ""
    # de-hyphenate line breaks: "inter-\nnational" -> "international"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # normalize newlines: collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # trim trailing spaces
    text = "\n".join([ln.rstrip() for ln in text.splitlines()])
    return text.strip()

def maybe_truncate_with_warning(text: str, max_tokens: int) -> Tuple[str, Optional[str]]:
    tok = estimate_tokens(text)
    if tok <= max_tokens:
        return text, None
    # Hard truncate by characters proportionally; warn.
    ratio = max_tokens / tok
    cut = max(2000, int(len(text) * ratio))
    truncated = text[:cut]
    warning = f"Context too large (~{tok} tokens). Truncated to fit ~{max_tokens} tokens. Consider pre-summarization."
    return truncated, warning


# =========================
# ToC Pipeline
# =========================

def discover_pdfs(base_dir: Path) -> List[Path]:
    return sorted([p for p in base_dir.rglob("*.pdf") if p.is_file()])

def make_toc_markdown(entries: List[Dict[str, Any]], base_dir: Path) -> str:
    # Hierarchical listing by folder path
    by_folder: Dict[str, List[Dict[str, Any]]] = {}
    for e in entries:
        rel = os.path.relpath(e["path"], base_dir)
        folder = os.path.dirname(rel) or "."
        by_folder.setdefault(folder, []).append({**e, "rel": rel})

    lines = ["# Directory ToC", ""]
    for folder in sorted(by_folder.keys()):
        lines.append(f"## {folder}")
        lines.append("")
        for e in sorted(by_folder[folder], key=lambda x: x["rel"].lower()):
            rel = e["rel"]
            blurb = e.get("blurb", "").strip()
            meta = e.get("meta", "")
            lines.append(f"- **[{Path(rel).name}]({rel})**")
            if meta:
                lines.append(f"  - _Metadata_: {meta}")
            if blurb:
                lines.append(f"  - {blurb}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

def throttle_sleep(i: int, base_s: float = 0.2, jitter: float = 0.2) -> None:
    time.sleep(base_s + random.random() * jitter + min(1.0, i * 0.02))


# =========================
# API key precedence + UI masking
# =========================

@dataclass
class KeyStatus:
    value: Optional[str]
    source: str  # "env" | "user" | "missing"

def resolve_key(env_name_candidates: List[str], session_key_name: str) -> KeyStatus:
    user = st.session_state.get(session_key_name)
    if user:
        return KeyStatus(value=user, source="user")
    for env_name in env_name_candidates:
        v = os.getenv(env_name)
        if v:
            return KeyStatus(value=v, source="env")
    return KeyStatus(value=None, source="missing")


# =========================
# Session state init
# =========================

def init_state():
    st.session_state.setdefault("run_log", [])
    st.session_state.setdefault("active_doc_name", "")
    st.session_state.setdefault("active_doc_type", "")  # pdf/txt/md/paste
    st.session_state.setdefault("active_pdf_bytes", None)
    st.session_state.setdefault("active_text_raw", "")
    st.session_state.setdefault("active_text_md", "")
    st.session_state.setdefault("selected_pages", [])
    st.session_state.setdefault("last_usage", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("toc_results", [])
    st.session_state.setdefault("toc_md", "")
    st.session_state.setdefault("custom_keywords", "")
    st.session_state.setdefault("keyword_color", "coral")
    st.session_state.setdefault("agents_doc", None)
    st.session_state.setdefault("skill_md", None)
    st.session_state.setdefault("agents_yaml_text", "")
    st.session_state.setdefault("skill_md_text", "")
    st.session_state.setdefault("note_text", "")
    st.session_state.setdefault("note_md", "")
    st.session_state.setdefault("chain_outputs", {})  # agent_id -> editable output

init_state()


# =========================
# Load defaults (or repo files)
# =========================

REPO_DIR = Path(".")
AGENTS_PATH = REPO_DIR / "agents.yaml"
SKILL_PATH = REPO_DIR / "SKILL.md"

if st.session_state["agents_doc"] is None:
    agents_text = load_text_file_or_default(AGENTS_PATH, DEFAULT_AGENTS_YAML)
    st.session_state["agents_yaml_text"] = agents_text
    try:
        raw = yaml.safe_load(agents_text) or {}
        aligned, warnings = align_agents_schema(raw)
        errs = validate_agents_schema(aligned)
        st.session_state["agents_doc"] = aligned if not errs else aligned
        if warnings:
            for w in warnings:
                add_log(f"agents.yaml alignment note: {w}", "WARN")
        if errs:
            for e in errs:
                add_log(f"agents.yaml validation: {e}", "WARN")
    except Exception as e:
        add_log(f"Failed to load agents.yaml: {e}", "ERROR")
        st.session_state["agents_doc"] = yaml.safe_load(DEFAULT_AGENTS_YAML)

if st.session_state["skill_md"] is None:
    skill_text = load_text_file_or_default(SKILL_PATH, DEFAULT_SKILL_MD)
    st.session_state["skill_md_text"] = skill_text
    st.session_state["skill_md"] = skill_text


# =========================
# Page config + WOW settings
# =========================

st.set_page_config(page_title="OPAL", layout="wide")

# Sidebar WOW controls (theme/lang/style)
with st.sidebar:
    lang = st.selectbox("Language / 語言", ["en", "zh-TW"], index=0, key="lang")
    theme = st.selectbox(t(lang, "theme"), [t(lang, "light"), t(lang, "dark")], index=1 if st.session_state.get("theme") == "dark" else 0)
    theme_val = "dark" if theme == t(lang, "dark") else "light"

    style_names = [s[0] for s in PAINTER_STYLES]
    cols = st.columns([3, 1])
    with cols[0]:
        painter = st.selectbox(t(lang, "style_pack"), style_names, index=style_names.index(st.session_state.get("painter", style_names[0])) if st.session_state.get("painter") in style_names else 0)
    with cols[1]:
        if st.button(t(lang, "jackpot")):
            painter = random.choice(style_names)
            st.session_state["painter"] = painter
    st.session_state["theme"] = theme_val
    st.session_state["painter"] = painter

inject_wow_css(st.session_state["theme"], st.session_state["painter"])

# Title
st.markdown(f"<div class='opal-card'><h2 style='margin:0'>{t(lang,'app_title')}</h2>"
            f"<div class='opal-small'>WOW UI • Theme: <b>{st.session_state['theme']}</b> • Style: <b>{st.session_state['painter']}</b></div></div>",
            unsafe_allow_html=True)
st.write("")

# =========================
# Sidebar: Global settings, keys, models, keywords, advanced
# =========================

MODEL_CATALOG = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini (as requested; availability depends on API/account)
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    # Anthropic (stub)
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
    # XAI (stub)
    "grok-4-fast-reasoning",
    "grok-4-1-fast-non-reasoning",
]

PROVIDER_OPTIONS = ["gemini", "openai", "both"]  # both = choose per action; UI uses selected model to infer

def infer_provider_from_model(model: str) -> str:
    m = model.lower()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini"):
        return "gemini"
    if "claude" in m:
        return "anthropic"
    if m.startswith("grok"):
        return "xai"
    return "openai"

with st.sidebar:
    st.markdown(f"### {t(lang,'sidebar_global')}")
    local_only = st.toggle(t(lang, "local_only"), value=st.session_state.get("local_only", False), help=t(lang, "local_only_help"))
    st.session_state["local_only"] = local_only

    provider_pref = st.selectbox(t(lang, "provider"), PROVIDER_OPTIONS, index=0, help=t(lang, "provider_help"))
    st.session_state["provider_pref"] = provider_pref

    default_model = st.selectbox(t(lang, "model"), MODEL_CATALOG, index=MODEL_CATALOG.index(st.session_state.get("default_model", "gemini-2.5-flash")) if st.session_state.get("default_model") in MODEL_CATALOG else 0)
    st.session_state["default_model"] = default_model

    temperature = st.slider(t(lang, "temperature"), 0.0, 1.0, float(st.session_state.get("temperature", 0.2)), 0.05)
    st.session_state["temperature"] = temperature

    max_tokens = st.number_input(t(lang, "max_tokens"), min_value=256, max_value=32000, value=int(st.session_state.get("max_tokens", 12000)), step=256, help=t(lang, "max_tokens_help"))
    st.session_state["max_tokens"] = int(max_tokens)

    # Keys
    st.markdown(f"### {t(lang,'api_keys')}")

    openai_status = resolve_key(["OPENAI_API_KEY"], "user_openai_key")
    google_status = resolve_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], "user_google_key")
    anthropic_status = resolve_key(["ANTHROPIC_API_KEY"], "user_anthropic_key")
    xai_status = resolve_key(["XAI_API_KEY"], "user_xai_key")

    def key_badge(status: KeyStatus) -> str:
        if status.source == "env":
            return f"<span class='opal-badge opal-status-ok'>{t(lang,'using_env')}</span>"
        if status.source == "user":
            return f"<span class='opal-badge opal-status-warn'>{t(lang,'using_user')}</span>"
        return f"<span class='opal-badge opal-status-bad'>{t(lang,'missing')}</span>"

    # Do not show API key if from env; allow user to override (masked)
    st.markdown(f"<div class='opal-small'>OpenAI: {key_badge(openai_status)}</div>", unsafe_allow_html=True)
    if openai_status.source != "env":
        v = st.text_input(t(lang, "openai_key"), value=mask_key(openai_status.value, True), type="password", placeholder="sk-...", help="Stored in session_state only.")
        # If user types something new (not all dots), accept it
        if v and "●" not in v:
            st.session_state["user_openai_key"] = v

    st.markdown(f"<div class='opal-small'>Gemini: {key_badge(google_status)}</div>", unsafe_allow_html=True)
    if google_status.source != "env":
        v = st.text_input(t(lang, "google_key"), value=mask_key(google_status.value, True), type="password", placeholder="AIza...", help="Stored in session_state only.")
        if v and "●" not in v:
            st.session_state["user_google_key"] = v

    st.markdown(f"<div class='opal-small'>Anthropic: {key_badge(anthropic_status)}</div>", unsafe_allow_html=True)
    if anthropic_status.source != "env":
        v = st.text_input(t(lang, "anthropic_key"), value=mask_key(anthropic_status.value, True), type="password")
        if v and "●" not in v:
            st.session_state["user_anthropic_key"] = v

    st.markdown(f"<div class='opal-small'>XAI: {key_badge(xai_status)}</div>", unsafe_allow_html=True)
    if xai_status.source != "env":
        v = st.text_input(t(lang, "xai_key"), value=mask_key(xai_status.value, True), type="password")
        if v and "●" not in v:
            st.session_state["user_xai_key"] = v

    # Keywords
    st.markdown(f"### {t(lang,'keywords')}")
    st.caption(t(lang, "default_keywords"))
    st.write(", ".join(DEFAULT_REGULATORY_KEYWORDS))
    st.session_state["custom_keywords"] = st.text_area(t(lang, "custom_keywords"), value=st.session_state.get("custom_keywords", ""), height=90)
    st.session_state["keyword_color"] = st.text_input("Highlight color (CSS name/hex)", value=st.session_state.get("keyword_color", "coral"))

    with st.expander(t(lang, "advanced"), expanded=False):
        st.session_state["ocr_dpi"] = st.slider(t(lang, "ocr_dpi"), 100, 400, int(st.session_state.get("ocr_dpi", 250)), 10)
        st.session_state["toc_concurrency"] = st.slider(t(lang, "concurrency"), 1, 12, int(st.session_state.get("toc_concurrency", 4)))
        st.session_state["backoff_max_s"] = st.slider(t(lang, "backoff_max"), 5, 120, int(st.session_state.get("backoff_max_s", 40)))
        if st.button(t(lang, "cleanup_tmp")):
            cleanup_tmp_dir()
            add_log("Cleaned up /tmp artifacts.", "INFO")

show_run_log(lang)


# =========================
# Dashboard (WOW status indicators)
# =========================

def build_status_panel():
    active_name = st.session_state.get("active_doc_name") or "—"
    md_len = len(st.session_state.get("active_text_md") or "")
    raw_len = len(st.session_state.get("active_text_raw") or "")
    approx_md_tokens = estimate_tokens(st.session_state.get("active_text_md") or "")
    last_usage: Optional[LLMUsage] = st.session_state.get("last_usage")

    # Key readiness
    model = st.session_state.get("default_model", "gemini-2.5-flash")
    inferred = infer_provider_from_model(model)
    ready = False
    if st.session_state.get("local_only"):
        ready = True
    else:
        if inferred == "openai":
            ready = resolve_key(["OPENAI_API_KEY"], "user_openai_key").source != "missing"
        elif inferred == "gemini":
            ready = resolve_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], "user_google_key").source != "missing"
        else:
            ready = False

    cols = st.columns(4)
    cols[0].markdown(f"<div class='opal-card'><div class='opal-small'>{t(lang,'active_doc')}</div>"
                     f"<div style='font-weight:700; font-size:16px'>{active_name}</div>"
                     f"<div class='opal-small'>raw chars: {raw_len:,} • md chars: {md_len:,}</div></div>", unsafe_allow_html=True)
    cols[1].markdown(f"<div class='opal-card'><div class='opal-small'>Context</div>"
                     f"<div style='font-weight:700; font-size:16px'>~{approx_md_tokens:,} tokens</div>"
                     f"<div class='opal-small'>max_tokens: {st.session_state.get('max_tokens')}</div></div>", unsafe_allow_html=True)
    cols[2].markdown(f"<div class='opal-card'><div class='opal-small'>LLM Readiness</div>"
                     f"<div style='font-weight:700; font-size:16px'>{'READY' if ready else 'NOT READY'}</div>"
                     f"<div class='opal-small'>model→{model} • provider→{inferred}</div></div>", unsafe_allow_html=True)
    if last_usage:
        cols[3].markdown(f"<div class='opal-card'><div class='opal-small'>Last Call</div>"
                         f"<div style='font-weight:700; font-size:16px'>{last_usage.provider}/{last_usage.model}</div>"
                         f"<div class='opal-small'>in~{last_usage.approx_input_tokens} • out~{last_usage.approx_output_tokens} • {last_usage.elapsed_s:.1f}s</div></div>", unsafe_allow_html=True)
    else:
        cols[3].markdown(f"<div class='opal-card'><div class='opal-small'>Last Call</div>"
                         f"<div style='font-weight:700; font-size:16px'>—</div>"
                         f"<div class='opal-small'>No LLM calls yet</div></div>", unsafe_allow_html=True)

tabs = st.tabs([
    t(lang, "tabs_dashboard"),
    t(lang, "tabs_ingest"),
    t(lang, "tabs_trim_ocr"),
    t(lang, "tabs_workspace"),
    t(lang, "tabs_agents"),
    t(lang, "tabs_toc"),
    t(lang, "tabs_agent_mgmt"),
    t(lang, "tabs_notes"),
])

with tabs[0]:
    build_status_panel()
    st.write("")
    st.markdown("<div class='opal-card'><b>Processing transparency</b><br/>"
                "<span class='opal-small'>This tool tracks steps, approximate token usage, and retains run history in-session. "
                "No API keys are written to disk. Temporary artifacts live under /tmp and can be cleaned up from the sidebar.</span>"
                "</div>", unsafe_allow_html=True)


# =========================
# Tab: Ingest & Preview
# =========================

def render_pdf_preview(pdf_bytes: bytes, height: int = 720):
    # Try streamlit-pdf-viewer; fallback to iframe
    with contextlib.suppress(Exception):
        import streamlit_pdf_viewer
        streamlit_pdf_viewer.pdf_viewer(pdf_bytes, height=height)
        return

    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    html = f"""
    <iframe
      src="data:application/pdf;base64,{b64}"
      width="100%"
      height="{height}"
      style="border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;"
      type="application/pdf"
    ></iframe>
    """
    st.markdown(html, unsafe_allow_html=True)

with tabs[1]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("<div class='opal-card'>", unsafe_allow_html=True)
        uploads = st.file_uploader(t(lang, "upload_files"), type=["pdf", "txt", "md"], accept_multiple_files=True)
        paste = st.text_area(t(lang, "paste_text"), height=220, placeholder="Paste text/markdown here…")

        set_btn = st.button(t(lang, "set_active"))
        st.markdown("</div>", unsafe_allow_html=True)

        if set_btn:
            try:
                if paste and paste.strip():
                    st.session_state["active_doc_name"] = "Pasted text"
                    st.session_state["active_doc_type"] = "paste"
                    st.session_state["active_pdf_bytes"] = None
                    st.session_state["active_text_raw"] = paste.strip()
                    st.session_state["active_text_md"] = paste.strip()
                    add_log("Set active document from pasted text.")
                elif uploads:
                    # Choose first as active (UX focus on one active doc)
                    f = uploads[0]
                    name = f.name
                    data = f.read()
                    ext = name.lower().split(".")[-1]
                    st.session_state["active_doc_name"] = name
                    st.session_state["active_doc_type"] = ext
                    if ext == "pdf":
                        st.session_state["active_pdf_bytes"] = data
                        st.session_state["active_text_raw"] = ""
                        st.session_state["active_text_md"] = ""
                        add_log(f"Set active document PDF: {name}")
                    else:
                        text = data.decode("utf-8", errors="ignore")
                        st.session_state["active_pdf_bytes"] = None
                        st.session_state["active_text_raw"] = text
                        st.session_state["active_text_md"] = text
                        add_log(f"Set active document text: {name}")
                else:
                    st.warning("Nothing to set active.")
            except Exception as e:
                add_log(f"Ingest failed: {e}", "ERROR")
                st.error(str(e))

    with c2:
        st.markdown("<div class='opal-card'>", unsafe_allow_html=True)
        st.subheader(t(lang, "pdf_preview"))
        pdfb = st.session_state.get("active_pdf_bytes")
        if pdfb:
            try:
                pc = pdf_page_count(pdfb)
                st.caption(f"Pages: {pc}")
                render_pdf_preview(pdfb, height=720)
            except Exception as e:
                add_log(f"PDF preview failed: {e}", "ERROR")
                st.error(f"PDF preview failed: {e}")
        else:
            st.caption("No active PDF.")
        st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: Trim & OCR
# =========================

with tabs[2]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    pdfb = st.session_state.get("active_pdf_bytes")
    if not pdfb:
        st.info("Load a PDF in Ingest & Preview to use Trim & OCR.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            pc = pdf_page_count(pdfb)
        except Exception as e:
            add_log(f"Failed to read PDF: {e}", "ERROR")
            st.error(str(e))
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            rng = st.text_input(t(lang, "page_range"), value=st.session_state.get("page_range", "1-1"))
            ocr_mode = st.selectbox(t(lang, "ocr_method"), [t(lang, "ocr_local"), t(lang, "ocr_vision"), t(lang, "ocr_hybrid")], index=0)
            st.session_state["page_range"] = rng

            pages = []
            try:
                pages = parse_page_ranges(rng, pc)
                st.write(f"{t(lang, 'compute_pages')}: {pages}")
                st.session_state["selected_pages"] = pages
            except Exception as e:
                st.session_state["selected_pages"] = []
                st.warning(str(e))

            run = st.button(t(lang, "run_trim_ocr"), disabled=not bool(pages))
            prog = st.progress(0, text="Idle")

            if run:
                tmp_dir = ensure_tmp_dir()
                add_log(f"Trim & OCR started. Pages={pages}, mode={ocr_mode}, dpi={st.session_state['ocr_dpi']}")

                try:
                    prog.progress(10, text="Extracting selected pages…")
                    trimmed_pdf = extract_pdf_pages(pdfb, pages)

                    # Store artifact in /tmp for trace/debug; still avoid writing secrets
                    trimmed_path = tmp_dir / "trimmed.pdf"
                    trimmed_path.write_bytes(trimmed_pdf)

                    raw_text = ""
                    md_text = ""

                    # Quick text extraction (helps avoid OCR if selectable)
                    prog.progress(20, text="Attempting native text extraction…")
                    native_text = extract_pdf_text_pages(trimmed_pdf, list(range(1, len(pages) + 1)))  # trimmed pdf pages are 1..N
                    native_text = (native_text or "").strip()

                    # Map mode selector to internal
                    mode_val = "local" if ocr_mode == t(lang, "ocr_local") else ("vision" if ocr_mode == t(lang, "ocr_vision") else "hybrid")

                    # Local-only constraints
                    if st.session_state["local_only"]:
                        if mode_val == "vision":
                            st.warning("Local-only mode: Vision OCR disabled. Falling back to Local OCR.")
                            mode_val = "local"

                    # If the PDF has good text and mode is local/hybrid, prefer it.
                    # For vision mode, we intentionally do OCR-to-Markdown from images.
                    if native_text and mode_val in ("local", "hybrid"):
                        raw_text = native_text
                        add_log("Used native PDF text extraction (no OCR).")
                        prog.progress(45, text="Cleaning into Markdown (local)…")
                        md_text = local_markdown_cleanup(raw_text)

                        if mode_val == "hybrid" and not st.session_state["local_only"]:
                            # LLM cleanup to reconstruct headings/tables
                            prog.progress(70, text="LLM cleanup (hybrid)…")
                            agent = get_agent_by_id(st.session_state["agents_doc"], "markdown_rewriter") or {}
                            skill_lib = parse_skills(st.session_state["skill_md"])
                            sys = assemble_system_prompt(agent, skill_lib, GLOBAL_SAFETY_RULES)
                            user = f"Clean this text into Markdown. Do not add facts.\n\nTEXT:\n{raw_text}"
                            model = st.session_state["default_model"]
                            provider = infer_provider_from_model(model)
                            key_status = resolve_key(
                                ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                                "user_openai_key" if provider == "openai" else "user_google_key",
                            )
                            client = LLMClient(provider, key_status.value, backoff_max_s=st.session_state["backoff_max_s"])
                            truncated, warn = maybe_truncate_with_warning(user, st.session_state["max_tokens"])
                            if warn:
                                st.warning(warn)
                                add_log(warn, "WARN")
                            out, usage = client.generate_text(
                                model=model,
                                system_prompt=sys,
                                user_prompt=truncated,
                                temperature=0.1,
                                max_tokens=st.session_state["max_tokens"],
                                secrets_to_scrub=[key_status.value or ""],
                            )
                            st.session_state["last_usage"] = usage
                            md_text = out or md_text

                    else:
                        # OCR path
                        prog.progress(35, text="Rendering pages to images…")
                        imgs = render_pdf_pages_to_png_bytes(trimmed_pdf, list(range(1, len(pages) + 1)), dpi=st.session_state["ocr_dpi"])
                        if not imgs:
                            raise RuntimeError("No images rendered from PDF.")

                        if mode_val == "local":
                            prog.progress(55, text="Running Tesseract OCR…")
                            raw_text = local_ocr_images(imgs)
                            prog.progress(75, text="Cleaning into Markdown (local)…")
                            md_text = local_markdown_cleanup(raw_text)

                        elif mode_val == "vision":
                            if st.session_state["local_only"]:
                                raise RuntimeError("Local-only mode: Vision OCR not allowed.")
                            prog.progress(55, text="Vision OCR → Markdown (LLM)…")
                            model = st.session_state["default_model"]
                            provider = infer_provider_from_model(model)
                            key_status = resolve_key(
                                ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                                "user_openai_key" if provider == "openai" else "user_google_key",
                            )
                            if key_status.source == "missing":
                                raise RuntimeError(f"Missing API key for provider={provider}. Add it in sidebar.")
                            client = LLMClient(provider, key_status.value, backoff_max_s=st.session_state["backoff_max_s"])
                            out, usage = client.generate_vision_markdown(
                                model=model,
                                system_prompt=VISION_OCR_SYSTEM,
                                user_prompt=VISION_OCR_USER,
                                images=imgs,
                                temperature=0.0,
                                max_tokens=st.session_state["max_tokens"],
                                secrets_to_scrub=[key_status.value or ""],
                            )
                            st.session_state["last_usage"] = usage
                            md_text = out
                            raw_text = ""  # vision directly returns md

                        else:  # hybrid
                            prog.progress(55, text="Running Tesseract OCR…")
                            raw_text = local_ocr_images(imgs)
                            prog.progress(65, text="Local cleanup…")
                            md_text = local_markdown_cleanup(raw_text)

                            if st.session_state["local_only"]:
                                add_log("Local-only mode: skipped LLM cleanup step for hybrid.", "WARN")
                            else:
                                prog.progress(80, text="LLM cleanup to Markdown…")
                                agent = get_agent_by_id(st.session_state["agents_doc"], "markdown_rewriter") or {}
                                skill_lib = parse_skills(st.session_state["skill_md"])
                                sys = assemble_system_prompt(agent, skill_lib, GLOBAL_SAFETY_RULES)
                                user = f"{MD_REWRITE_SYSTEM}\n\nRAW OCR TEXT:\n{raw_text}"
                                model = st.session_state["default_model"]
                                provider = infer_provider_from_model(model)
                                key_status = resolve_key(
                                    ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                                    "user_openai_key" if provider == "openai" else "user_google_key",
                                )
                                if key_status.source == "missing":
                                    raise RuntimeError(f"Missing API key for provider={provider}. Add it in sidebar.")
                                client = LLMClient(provider, key_status.value, backoff_max_s=st.session_state["backoff_max_s"])
                                truncated, warn = maybe_truncate_with_warning(user, st.session_state["max_tokens"])
                                if warn:
                                    st.warning(warn)
                                    add_log(warn, "WARN")
                                out, usage = client.generate_text(
                                    model=model,
                                    system_prompt=sys,
                                    user_prompt=truncated,
                                    temperature=0.1,
                                    max_tokens=st.session_state["max_tokens"],
                                    secrets_to_scrub=[key_status.value or ""],
                                )
                                st.session_state["last_usage"] = usage
                                md_text = out or md_text

                    st.session_state["active_text_raw"] = raw_text
                    st.session_state["active_text_md"] = md_text
                    add_log("Trim & OCR completed.")
                    prog.progress(100, text="Done.")
                except Exception as e:
                    add_log(f"Trim & OCR failed: {e}", "ERROR")
                    prog.progress(0, text="Failed.")
                    st.error(str(e))

            st.markdown("---")
            st.subheader(t(lang, "raw_text"))
            st.text_area(label="raw", value=st.session_state.get("active_text_raw", ""), height=220, label_visibility="collapsed")

            st.subheader(t(lang, "md_transform"))
            st.text_area(label="md", value=st.session_state.get("active_text_md", ""), height=260, label_visibility="collapsed")

            st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: Markdown Workspace
# =========================

with tabs[3]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    md = st.session_state.get("active_text_md", "")
    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.subheader(t(lang, "md_editor"))
        edited = st.text_area("md_editor", value=md, height=720, label_visibility="collapsed")
        st.session_state["active_text_md"] = edited

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                t(lang, "download_md"),
                data=edited.encode("utf-8"),
                file_name=(Path(st.session_state.get("active_doc_name") or "document").stem + ".md"),
                mime="text/markdown",
            )
        with d2:
            st.download_button(
                t(lang, "download_txt"),
                data=(st.session_state.get("active_text_raw") or edited).encode("utf-8"),
                file_name=(Path(st.session_state.get("active_doc_name") or "document").stem + ".txt"),
                mime="text/plain",
            )

    with c2:
        st.subheader(t(lang, "md_preview"))
        do_hl = st.toggle(t(lang, "toggle_highlight"), value=True)
        custom = [k.strip() for k in (st.session_state.get("custom_keywords") or "").splitlines() if k.strip()]
        kws = DEFAULT_REGULATORY_KEYWORDS + custom
        preview_md = edited
        if do_hl:
            preview_md = highlight_keywords_markdown_safe(preview_md, kws, color=st.session_state.get("keyword_color", "coral"))
        st.markdown(preview_md, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: Agents (edit prompt/model/tokens; chaining with editable intermediate outputs)
# =========================

def push_history(entry: Dict[str, Any]) -> None:
    st.session_state.setdefault("history", [])
    st.session_state["history"].insert(0, entry)

def can_call_provider(provider: str) -> Tuple[bool, str]:
    if st.session_state.get("local_only"):
        return False, "Local-only mode enabled; cloud calls disabled."
    if provider == "openai":
        ks = resolve_key(["OPENAI_API_KEY"], "user_openai_key")
        if ks.source == "missing":
            return False, "Missing OpenAI key."
        return True, ""
    if provider == "gemini":
        ks = resolve_key(["GOOGLE_API_KEY", "GEMINI_API_KEY"], "user_google_key")
        if ks.source == "missing":
            return False, "Missing Gemini key."
        return True, ""
    return False, f"Provider '{provider}' not available in this build."

with tabs[4]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    agents_doc = st.session_state.get("agents_doc") or {"agents": []}
    agents_list = agents_doc.get("agents", [])
    if not agents_list:
        st.error("No agents loaded.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        agent_names = [f"{a.get('name')} ({a.get('id')})" for a in agents_list]
        agent_ids = [a.get("id") for a in agents_list]
        idx = 0
        if st.session_state.get("selected_agent_id") in agent_ids:
            idx = agent_ids.index(st.session_state["selected_agent_id"])

        selected_label = st.selectbox(t(lang, "agents_select"), agent_names, index=idx)
        selected_id = re.search(r"\(([^)]+)\)\s*$", selected_label).group(1)
        st.session_state["selected_agent_id"] = selected_id

        agent = get_agent_by_id(agents_doc, selected_id) or {}
        skill_lib = parse_skills(st.session_state.get("skill_md") or DEFAULT_SKILL_MD)

        st.subheader(t(lang, "agent_details"))
        cA, cB = st.columns([1, 1])
        with cA:
            role = st.text_input("role", value=agent.get("role", ""), disabled=True)
            skills = st.multiselect(t(lang, "skills"), options=sorted(skill_lib.keys()), default=agent.get("skills", []) or [])
        with cB:
            # Model override and tokens per spec
            model_override = st.selectbox("Model override", MODEL_CATALOG, index=MODEL_CATALOG.index(agent.get("default_model", st.session_state["default_model"])) if agent.get("default_model", st.session_state["default_model"]) in MODEL_CATALOG else 0)
            temp_override = st.slider("Temperature override", 0.0, 1.0, float(agent.get("temperature", st.session_state["temperature"])), 0.05)
            max_tokens_override = st.number_input("Max tokens override", min_value=256, max_value=32000, value=int(agent.get("max_tokens", st.session_state["max_tokens"])), step=256)

        sys_prompt = st.text_area(t(lang, "system_prompt"), value=agent.get("system_prompt", ""), height=220)
        user_instruction = st.text_area(t(lang, "user_instruction"), value=st.session_state.get("user_instruction", ""), height=80)

        # Allow user to modify the prompt before execute agents one by one.
        final_agent = {
            **agent,
            "skills": skills,
            "system_prompt": sys_prompt,
            "default_model": model_override,
            "temperature": float(temp_override),
            "max_tokens": int(max_tokens_override),
        }

        # Editable input context for this run (can be agent output of previous step)
        st.markdown("---")
        st.markdown("#### Input to this agent (editable)")
        input_mode = st.radio("Input view", ["Markdown", "Text"], horizontal=True)
        current_context = st.session_state.get("active_text_md", "")

        # If user previously edited chained output for this agent, use it as the input editor content
        input_editor_key = f"agent_input_editor_{selected_id}"
        if input_editor_key not in st.session_state:
            st.session_state[input_editor_key] = current_context

        agent_input = st.text_area(
            "agent_input",
            value=st.session_state[input_editor_key],
            height=220,
            label_visibility="collapsed",
        )
        st.session_state[input_editor_key] = agent_input

        # Run agent
        run = st.button(t(lang, "run_agent"))
        output_area_key = f"agent_output_{selected_id}"
        st.session_state.setdefault(output_area_key, "")

        if run:
            model = final_agent["default_model"]
            provider = infer_provider_from_model(model)

            ok, reason = can_call_provider(provider)
            if not ok:
                st.error(reason)
            else:
                try:
                    ks = resolve_key(
                        ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                        "user_openai_key" if provider == "openai" else "user_google_key",
                    )
                    client = LLMClient(provider, ks.value, backoff_max_s=st.session_state["backoff_max_s"])

                    final_sys = assemble_system_prompt(final_agent, skill_lib, GLOBAL_SAFETY_RULES)
                    doc_ctx = agent_input
                    # Guardrails: avoid silent huge drops
                    user_prompt = f"DOCUMENT CONTEXT (Markdown):\n{doc_ctx}\n\nUSER INSTRUCTION:\n{user_instruction}".strip()
                    user_prompt, warn = maybe_truncate_with_warning(user_prompt, int(final_agent["max_tokens"]))
                    if warn:
                        st.warning(warn)
                        add_log(warn, "WARN")

                    add_log(f"Running agent {selected_id} with {provider}/{model} (temp={final_agent['temperature']}, max_tokens={final_agent['max_tokens']})")
                    with st.spinner("Agent running…"):
                        out, usage = client.generate_text(
                            model=model,
                            system_prompt=final_sys,
                            user_prompt=user_prompt,
                            temperature=float(final_agent["temperature"]),
                            max_tokens=int(final_agent["max_tokens"]),
                            secrets_to_scrub=[ks.value or ""],
                        )
                    st.session_state["last_usage"] = usage
                    st.session_state[output_area_key] = out

                    push_history({
                        "ts": now_ts(),
                        "agent_id": selected_id,
                        "agent_name": agent.get("name"),
                        "provider": provider,
                        "model": model,
                        "temperature": float(final_agent["temperature"]),
                        "max_tokens": int(final_agent["max_tokens"]),
                        "skills": list(skills),
                        "system_prompt": sys_prompt,
                        "user_instruction": user_instruction,
                        "input_preview": (doc_ctx[:800] + "…") if len(doc_ctx) > 800 else doc_ctx,
                        "output": out,
                        "usage": dataclasses.asdict(usage) if usage else None,
                    })
                    add_log(f"Agent {selected_id} completed.")
                except Exception as e:
                    add_log(f"Agent run failed: {e}", "ERROR")
                    st.error(str(e))

        # Output view + editable (as input to next agent)
        st.markdown("---")
        st.subheader(t(lang, "output"))
        out_view = st.radio("Output view", ["Markdown", "Text"], horizontal=True, key=f"out_view_{selected_id}")

        editable_output = st.text_area(
            "editable_output",
            value=st.session_state.get(output_area_key, ""),
            height=260,
            label_visibility="collapsed",
            key=f"editable_output_{selected_id}",
        )
        st.session_state[output_area_key] = editable_output

        if out_view == "Markdown":
            st.markdown(editable_output, unsafe_allow_html=True)
        else:
            st.code(editable_output, language="text")

        st.markdown("#### Send output to next agent")
        cols = st.columns([2, 2, 3])
        with cols[0]:
            next_agent_id = st.selectbox("Next agent", agent_ids, index=(agent_ids.index(selected_id) + 1) % len(agent_ids))
        with cols[1]:
            send_mode = st.selectbox("How to send", [t(lang, "chain_replace"), t(lang, "chain_append")], index=0)
        with cols[2]:
            if st.button("Apply to next agent input"):
                key_next = f"agent_input_editor_{next_agent_id}"
                if send_mode == t(lang, "chain_replace"):
                    st.session_state[key_next] = editable_output
                else:
                    st.session_state[key_next] = (st.session_state.get(key_next, st.session_state.get("active_text_md", "")) + "\n\n---\n\n" + editable_output).strip()
                add_log(f"Applied output of {selected_id} to next agent {next_agent_id} ({'replace' if send_mode==t(lang,'chain_replace') else 'append'}).")

        # Chain Runner
        st.markdown("---")
        st.subheader(t(lang, "chain_runner"))
        chain_ids = st.multiselect("Chain (agent IDs in order)", agent_ids, default=[selected_id])
        chain_mode = st.radio(t(lang, "chain_mode"), [t(lang, "chain_replace"), t(lang, "chain_append")], horizontal=True)
        run_chain = st.button(t(lang, "run_chain"), disabled=not chain_ids)

        if run_chain:
            # Start from the current agent input editor content
            chain_input = st.session_state.get(f"agent_input_editor_{chain_ids[0]}", st.session_state.get("active_text_md", ""))
            st.session_state["chain_outputs"] = {}
            for i, aid in enumerate(chain_ids):
                a = get_agent_by_id(agents_doc, aid) or {}
                # Allow per-agent runtime overrides by reading current UI if selected; otherwise use defaults
                model = a.get("default_model", st.session_state["default_model"])
                provider = infer_provider_from_model(model)
                ok, reason = can_call_provider(provider)
                if not ok:
                    st.error(f"Chain stopped at {aid}: {reason}")
                    break
                ks = resolve_key(
                    ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                    "user_openai_key" if provider == "openai" else "user_google_key",
                )
                client = LLMClient(provider, ks.value, backoff_max_s=st.session_state["backoff_max_s"])
                sys = assemble_system_prompt(a, skill_lib, GLOBAL_SAFETY_RULES)
                user = f"DOCUMENT CONTEXT:\n{chain_input}\n\nINSTRUCTION:\n{user_instruction}".strip()
                user, warn = maybe_truncate_with_warning(user, int(a.get("max_tokens", st.session_state["max_tokens"])))
                if warn:
                    st.warning(f"{aid}: {warn}")
                    add_log(f"{aid}: {warn}", "WARN")

                st.write(f"Running **{aid}** ({provider}/{model}) …")
                with st.spinner(f"Chain step {i+1}/{len(chain_ids)}: {aid}"):
                    out, usage = client.generate_text(
                        model=model,
                        system_prompt=sys,
                        user_prompt=user,
                        temperature=float(a.get("temperature", st.session_state["temperature"])),
                        max_tokens=int(a.get("max_tokens", st.session_state["max_tokens"])),
                        secrets_to_scrub=[ks.value or ""],
                    )
                st.session_state["last_usage"] = usage
                st.session_state["chain_outputs"][aid] = out

                # Let user modify output immediately (stored in chain_outputs)
                edited_step = st.text_area(f"Edit output for {aid} (feeds next)", value=out, height=180, key=f"chain_edit_{aid}")
                st.session_state["chain_outputs"][aid] = edited_step

                if chain_mode == t(lang, "chain_replace"):
                    chain_input = edited_step
                else:
                    chain_input = (chain_input + "\n\n---\n\n" + edited_step).strip()

            st.success("Chain completed (or stopped with an error).")

        # History viewer
        st.markdown("---")
        st.subheader(t(lang, "history"))
        hist = st.session_state.get("history", [])
        if not hist:
            st.caption("—")
        else:
            for i, h in enumerate(hist[:15]):
                with st.expander(f"{h['ts']} • {h['agent_id']} • {h['model']}", expanded=False):
                    st.markdown(f"**Provider:** {h.get('provider')}  \n**Model:** {h.get('model')}  \n**Temp:** {h.get('temperature')}  \n**Max tokens:** {h.get('max_tokens')}")
                    st.markdown("**Input preview:**")
                    st.code(h.get("input_preview", ""), language="markdown")
                    st.markdown("**Output:**")
                    st.markdown(h.get("output", ""), unsafe_allow_html=True)
                    if st.button(f"{t(lang,'restore')} output to workspace", key=f"restore_{i}"):
                        st.session_state["active_text_md"] = h.get("output", "")
                        add_log(f"Restored history item {i} output to Markdown workspace.")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: ToC Pipeline
# =========================

with tabs[5]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    base_dir_str = st.text_input(t(lang, "toc_base_dir"), value=st.session_state.get("toc_base_dir", "."))
    st.session_state["toc_base_dir"] = base_dir_str
    toc_ocr_choice = st.selectbox("ToC OCR fallback", ["auto", "local", "vision"], index=0, help="auto: OCR only if page 1 has little/no text.")
    toc_agent_id = "toc_summarizer"

    run = st.button(t(lang, "run_toc"))
    if run:
        base_dir = Path(base_dir_str).resolve()
        add_log(f"ToC pipeline started. base_dir={base_dir} ocr={toc_ocr_choice}")
        if not base_dir.exists():
            st.error("Base directory does not exist.")
        else:
            pdfs = discover_pdfs(base_dir)
            st.write(f"Discovered {len(pdfs)} PDFs.")
            if not pdfs:
                st.warning("No PDFs found.")
            else:
                # Prepare summarizer agent
                agents_doc = st.session_state["agents_doc"]
                agent = get_agent_by_id(agents_doc, toc_agent_id) or {}
                skill_lib = parse_skills(st.session_state.get("skill_md") or DEFAULT_SKILL_MD)
                sys = assemble_system_prompt(agent, skill_lib, GLOBAL_SAFETY_RULES)

                # Choose model/provider from agent default, else global
                model = agent.get("default_model", st.session_state["default_model"])
                provider = infer_provider_from_model(model)

                ok, reason = (True, "")
                if not st.session_state["local_only"]:
                    ok, reason = can_call_provider(provider)
                else:
                    ok, reason = (False, "Local-only mode: ToC summarization disabled (needs LLM).")

                if not ok:
                    st.error(reason)
                else:
                    ks = resolve_key(
                        ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                        "user_openai_key" if provider == "openai" else "user_google_key",
                    )
                    client = LLMClient(provider, ks.value, backoff_max_s=st.session_state["backoff_max_s"])

                    progress = st.progress(0, text="Starting…")
                    results: List[Dict[str, Any]] = []
                    concurrency = int(st.session_state.get("toc_concurrency", 4))

                    def process_one(i: int, path: Path) -> Dict[str, Any]:
                        throttle_sleep(i)
                        try:
                            pdf_bytes = path.read_bytes()
                            # Extract first page text
                            text = ""
                            with contextlib.suppress(Exception):
                                text = extract_pdf_text_pages(pdf_bytes, [1]).strip()

                            # If little/no text, OCR based on setting
                            need_ocr = (len(text) < 40)
                            if toc_ocr_choice == "local":
                                need_ocr = True
                            elif toc_ocr_choice == "vision":
                                need_ocr = True

                            if need_ocr:
                                # Render first page
                                imgs = render_pdf_pages_to_png_bytes(pdf_bytes, [1], dpi=st.session_state["ocr_dpi"])
                                if toc_ocr_choice in ("auto", "local"):
                                    # local OCR first
                                    with contextlib.suppress(Exception):
                                        text = local_ocr_images(imgs).strip()
                                if (len(text) < 40) and toc_ocr_choice in ("auto", "vision"):
                                    # vision OCR fallback
                                    out_md, _ = client.generate_vision_markdown(
                                        model=model,
                                        system_prompt=VISION_OCR_SYSTEM,
                                        user_prompt=VISION_OCR_USER,
                                        images=imgs,
                                        temperature=0.0,
                                        max_tokens=2000,
                                        secrets_to_scrub=[ks.value or ""],
                                    )
                                    text = out_md.strip()

                            user = f"FIRST PAGE TEXT/MD:\n{text}\n\nReturn ~100-word blurb."
                            user, _ = maybe_truncate_with_warning(user, 1200)
                            blurb, usage = client.generate_text(
                                model=model,
                                system_prompt=sys,
                                user_prompt=user,
                                temperature=float(agent.get("temperature", 0.2)),
                                max_tokens=int(agent.get("max_tokens", 1200)),
                                secrets_to_scrub=[ks.value or ""],
                            )
                            return {"path": str(path), "blurb": blurb.strip(), "meta": f"provider={provider}, model={model}", "usage": dataclasses.asdict(usage)}
                        except Exception as e:
                            add_log(f"ToC item failed for {path}: {e}", "ERROR")
                            return {"path": str(path), "blurb": f"[ERROR] {e}", "meta": "", "usage": None}

                    with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
                        futs = [ex.submit(process_one, i, p) for i, p in enumerate(pdfs)]
                        for j, fut in enumerate(cf.as_completed(futs), start=1):
                            res = fut.result()
                            results.append(res)
                            progress.progress(int(j / len(pdfs) * 100), text=f"Processed {j}/{len(pdfs)}")

                    toc_md = make_toc_markdown(results, base_dir)
                    st.session_state["toc_results"] = results
                    st.session_state["toc_md"] = toc_md
                    add_log("ToC pipeline completed.")
                    st.success("ToC pipeline completed.")

    st.markdown("---")
    st.subheader(t(lang, "toc_preview"))
    st.text_area("toc_md", value=st.session_state.get("toc_md", ""), height=520, label_visibility="collapsed")
    st.download_button(t(lang, "download_toc"), data=(st.session_state.get("toc_md", "") or "").encode("utf-8"), file_name="ToC.md", mime="text/markdown")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: Agent Management (upload/download agents.yaml & SKILL.md, align schema + diff)
# =========================

with tabs[6]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        up_agents = st.file_uploader(t(lang, "upload_agents"), type=["yaml", "yml"], accept_multiple_files=False)
        if up_agents:
            txt = up_agents.read().decode("utf-8", errors="ignore")
            st.session_state["agents_yaml_text"] = txt
            add_log("Uploaded agents.yaml into session.")
        st.text_area("agents_yaml", value=st.session_state.get("agents_yaml_text", ""), height=340)

    with c2:
        up_skill = st.file_uploader(t(lang, "upload_skill"), type=["md", "txt"], accept_multiple_files=False)
        if up_skill:
            txt = up_skill.read().decode("utf-8", errors="ignore")
            st.session_state["skill_md_text"] = txt
            st.session_state["skill_md"] = txt
            add_log("Uploaded SKILL.md into session.")
        st.text_area("skill_md", value=st.session_state.get("skill_md_text", ""), height=340)

    st.markdown("---")
    st.subheader(t(lang, "validation"))
    try:
        raw = yaml.safe_load(st.session_state.get("agents_yaml_text") or "") or {}
        aligned, warnings = align_agents_schema(raw)
        errs = validate_agents_schema(aligned)
        if warnings:
            st.warning("Schema alignment notes:\n- " + "\n- ".join(warnings))
        if errs:
            st.error("Schema validation errors:\n- " + "\n- ".join(errs))
        else:
            st.success("agents.yaml is valid under the standard schema.")
    except Exception as e:
        st.error(f"Failed to parse agents.yaml: {e}")
        aligned = None

    if st.button(t(lang, "align_schema")) and aligned is not None:
        before = (st.session_state.get("agents_yaml_text") or "").splitlines()
        after_text = yaml.safe_dump(aligned, sort_keys=False, allow_unicode=True)
        after = after_text.splitlines()
        st.session_state["agents_doc"] = aligned
        st.session_state["agents_yaml_text"] = after_text
        add_log("Aligned agents.yaml schema (deterministic).")

        st.subheader(t(lang, "diff"))
        diff = "\n".join(difflib.unified_diff(before, after, fromfile="before", tofile="after", lineterm=""))
        st.code(diff or "No diff.", language="diff")

    st.download_button("Download current agents.yaml", data=(st.session_state.get("agents_yaml_text") or DEFAULT_AGENTS_YAML).encode("utf-8"), file_name="agents.yaml", mime="text/yaml")
    st.download_button("Download current SKILL.md", data=(st.session_state.get("skill_md_text") or DEFAULT_SKILL_MD).encode("utf-8"), file_name="SKILL.md", mime="text/markdown")

    # Self-checks
    with st.expander(t(lang, "self_checks"), expanded=False):
        st.write("Page range parsing quick checks:")
        tests = [
            ("1-3,5,9-10", 12),
            ("2,2,3", 3),
        ]
        for rng, mp in tests:
            try:
                got = parse_page_ranges(rng, mp)
                st.code(f"{rng} (max={mp}) -> {got}", language="text")
            except Exception as e:
                st.code(f"{rng} failed: {e}", language="text")

    st.markdown("</div>", unsafe_allow_html=True)


# =========================
# Tab: AI Note Keeper (paste/upload → organized markdown + keywords coral; editable; AI Magics)
# =========================

NOTE_ORG_SYSTEM = """\
You are an AI Note Keeper. Convert messy notes into organized Markdown.

Rules:
- Do not invent facts beyond the note.
- Preserve original meaning.
- Produce a clear hierarchy with headings, bullets, and (if useful) sections:
  ## Summary, ## Key Points, ## Decisions, ## Action Items, ## Risks/Issues, ## Open Questions, ## References
- If something is unclear, put it under "Open Questions".
Output only Markdown.
"""

def extract_text_from_upload(file) -> Tuple[str, str]:
    name = file.name
    ext = name.lower().split(".")[-1]
    data = file.read()
    if ext in ("txt", "md"):
        return name, data.decode("utf-8", errors="ignore")
    if ext == "pdf":
        # attempt first: native text extraction (all pages)
        try:
            reader = PdfReader(io.BytesIO(data))
            txts = []
            for p in reader.pages:
                with contextlib.suppress(Exception):
                    txts.append(p.extract_text() or "")
            joined = "\n\n".join(txts).strip()
            return name, joined
        except Exception:
            return name, ""
    return name, ""

def run_note_llm(md_input: str, system: str, user: str) -> str:
    model = st.session_state["default_model"]
    provider = infer_provider_from_model(model)
    ok, reason = can_call_provider(provider)
    if not ok:
        raise RuntimeError(reason)
    ks = resolve_key(
        ["OPENAI_API_KEY"] if provider == "openai" else ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "user_openai_key" if provider == "openai" else "user_google_key",
    )
    client = LLMClient(provider, ks.value, backoff_max_s=st.session_state["backoff_max_s"])
    user_prompt = user.replace("{{NOTE}}", md_input)
    user_prompt, warn = maybe_truncate_with_warning(user_prompt, st.session_state["max_tokens"])
    if warn:
        st.warning(warn)
        add_log(warn, "WARN")
    out, usage = client.generate_text(
        model=model,
        system_prompt=system,
        user_prompt=user_prompt,
        temperature=0.2,
        max_tokens=st.session_state["max_tokens"],
        secrets_to_scrub=[ks.value or ""],
    )
    st.session_state["last_usage"] = usage
    return out

def magic_keyword_colorize(md: str, keyword_colors: Dict[str, str]) -> str:
    # apply span style per keyword, avoid code blocks
    out = md
    for kw, col in keyword_colors.items():
        out = highlight_keywords_markdown_safe(out, [kw], color=col)
    return out

with tabs[7]:
    st.markdown("<div class='opal-card'>", unsafe_allow_html=True)

    st.subheader(t(lang, "notes_input"))
    note_upload = st.file_uploader("Upload note source", type=["pdf", "txt", "md"], accept_multiple_files=False)
    note_paste = st.text_area("Paste note text/markdown", value=st.session_state.get("note_text", ""), height=200)

    if note_upload:
        name, txt = extract_text_from_upload(note_upload)
        if txt:
            st.session_state["note_text"] = txt
            note_paste = txt
            add_log(f"Loaded note from upload: {name}")
        else:
            st.warning("Could not extract text from the uploaded file. Consider OCR via the main OCR tab, then paste here.")

    organize = st.button(t(lang, "notes_organize"))
    if organize:
        if st.session_state.get("local_only"):
            # Local-only: simple heuristic structure
            raw = note_paste.strip()
            org = "## Summary\n\n" + (raw[:800] + ("…" if len(raw) > 800 else "")) + "\n\n## Key Points\n\n- " + "\n- ".join([ln.strip() for ln in raw.splitlines() if ln.strip()][:20])
            st.session_state["note_md"] = org
            add_log("Organized note using local heuristic (local-only mode).")
        else:
            try:
                out = run_note_llm(note_paste, NOTE_ORG_SYSTEM, "Organize this note into Markdown.\n\nNOTE:\n{{NOTE}}")
                st.session_state["note_md"] = out
                add_log("Organized note using LLM.")
            except Exception as e:
                add_log(f"Note organize failed: {e}", "ERROR")
                st.error(str(e))

    st.markdown("---")
    st.subheader(t(lang, "notes_editor"))
    note_md = st.text_area("note_md", value=st.session_state.get("note_md", ""), height=380)
    st.session_state["note_md"] = note_md

    # Always show preview with coral highlights for default regulatory keywords
    st.markdown("**Preview (with coral keywords):**")
    preview = highlight_keywords_markdown_safe(note_md, DEFAULT_REGULATORY_KEYWORDS, color="coral")
    st.markdown(preview, unsafe_allow_html=True)

    # AI Magics (6 features)
    st.markdown("---")
    st.subheader(t(lang, "ai_magics"))

    magic = st.selectbox(
        "Choose a Magic",
        [
            "1) Extract Action Items",
            "2) Executive Summary (brief)",
            "3) Risks & Issues Finder",
            "4) Generate Q&A for Review",
            "5) AI Keywords (custom colors)",
            "6) Convert to Meeting Minutes",
        ],
        index=0,
    )

    keyword_color_block = None
    if magic.startswith("5)"):
        st.markdown(f"**{t(lang,'magic_keywords')}**")
        kws = st.text_area("Keywords (one per line)", value="Contraindication\nRecall\n510(k)", height=120)
        col = st.color_picker("Color", value="#ff7f50")
        keyword_color_block = (kws, col)

    if st.button(t(lang, "run_magic")):
        try:
            if magic.startswith("5)") and keyword_color_block:
                kws, col = keyword_color_block
                pairs = {k.strip(): col for k in kws.splitlines() if k.strip()}
                colored = magic_keyword_colorize(note_md, pairs)
                st.session_state["note_md"] = colored
                add_log("AI Keywords applied (custom colors).")
            else:
                if st.session_state.get("local_only"):
                    raise RuntimeError("Local-only mode: AI Magics (LLM) are disabled except AI Keywords.")
                prompts = {
                    "1)": ("Extract action items into a Markdown checklist. Do not invent.\n\nNOTE:\n{{NOTE}}"),
                    "2)": ("Write a concise executive summary (3–6 bullets) of the note.\n\nNOTE:\n{{NOTE}}"),
                    "3)": ("Identify risks/issues/unknowns and group them. Cite exact phrases where possible.\n\nNOTE:\n{{NOTE}}"),
                    "4)": ("Create a Q&A set (10–20 questions) a reviewer should ask, based only on the note.\n\nNOTE:\n{{NOTE}}"),
                    "6)": ("Convert into meeting minutes with attendees (if present), agenda, decisions, action items, and open questions.\n\nNOTE:\n{{NOTE}}"),
                }
                k = magic.split(")")[0] + ")"
                user = prompts.get(k, "Improve this note.\n\nNOTE:\n{{NOTE}}")
                out = run_note_llm(note_md, "You are a careful analyst. Output Markdown only.", user)
                st.session_state["note_md"] = out
                add_log(f"Ran Magic: {magic}")
        except Exception as e:
            add_log(f"Magic failed: {e}", "ERROR")
            st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)
