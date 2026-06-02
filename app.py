import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from transformers import pipeline
import re
import datetime
import os
import time
import google.generativeai as genai

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Comment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Design System (Light Theme) ───────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
.stApp { background: #F8FAFC; }
section[data-testid="stSidebar"] {
    background: #EFF6FF !important;
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] * { color: #0F172A !important; }
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li { color: #475569 !important; }
.stTextInput input, .stNumberInput input {
    background: #FFFFFF !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    color: #0F172A !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
}
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"] { background: #2563EB !important; color: white !important; }
.stButton > button[kind="primary"]:hover { background: #1D4ED8 !important; }
.stButton > button[kind="secondary"] {
    background: white !important;
    border-color: #CBD5E1 !important;
    color: #374151 !important;
}
.stButton > button[kind="secondary"]:hover { background: #F1F5F9 !important; }
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 12px rgba(15,23,42,0.05);
}
div[data-testid="stMetric"] label {
    color: #64748B !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 13px !important; }
h1 { color: #0F172A !important; font-weight: 800 !important; }
h2, h3 { color: #1E293B !important; font-weight: 700 !important; }
.cluster-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 10px;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    transition: box-shadow 0.15s;
}
.cluster-card:hover { box-shadow: 0 4px 20px rgba(15,23,42,0.10); }
.cluster-military  { border-left: 4px solid #DC2626; }
.cluster-economic  { border-left: 4px solid #D97706; }
.cluster-electoral { border-left: 4px solid #2563EB; }
button[data-baseweb="tab"] { font-weight: 600 !important; color: #64748B !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    color: #0F172A !important;
    border-bottom-color: #2563EB !important;
}
hr { border-color: #E2E8F0 !important; margin: 24px 0 !important; }
.empty-state { text-align: center; padding: 80px 20px; color: #94A3B8; }
.empty-state h2 { color: #64748B !important; }
.chart-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(15,23,42,0.05);
    margin-bottom: 4px;
}
.report-card {
    background: linear-gradient(135deg, #F0FFF4 0%, #FFFFFF 100%);
    border: 1px solid #BBF7D0;
    border-left: 5px solid #16A34A;
    border-radius: 16px;
    padding: 28px 32px;
    margin-top: 8px;
    box-shadow: 0 4px 24px rgba(22,163,74,0.08);
    line-height: 1.75;
    color: #0F172A;
    font-size: 15px;
}
.report-card h3 { color: #15803D !important; margin-top: 20px; margin-bottom: 6px; }
.report-card h4 { color: #16A34A !important; margin-top: 14px; margin-bottom: 4px; }
.report-meta {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 18px;
    padding-bottom: 14px;
    border-bottom: 1px solid #BBF7D0;
    flex-wrap: wrap;
}
.report-badge {
    background: #16A34A;
    color: white;
    border-radius: 9999px;
    padding: 3px 12px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.03em;
}
.biz-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 10px 0;
    box-shadow: 0 2px 8px rgba(15,23,42,0.04);
}
.biz-card-title { font-weight: 700; color: #0F172A; font-size: 16px; margin-bottom: 6px; }
.biz-card-tag {
    display: inline-block;
    background: #F1F5F9;
    color: #475569;
    border-radius: 9999px;
    padding: 2px 10px;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {
    "Positive": "#16A34A",
    "Negative": "#DC2626",
    "Neutral":  "#94A3B8"
}
CLUSTER_NAMES  = ["🪖 Military/Hawkish", "💰 Economic/Domestic", "🗳️ Electoral/Party"]
CLUSTER_COLORS = {
    "🪖 Military/Hawkish":   "#DC2626",
    "💰 Economic/Domestic":  "#D97706",
    "🗳️ Electoral/Party":    "#2563EB"
}
CLUSTER_CSS    = ["cluster-military", "cluster-economic", "cluster-electoral"]
CLUSTER_LABELS = [
    "military action and war support",
    "economic concerns inflation and cost of living",
    "electoral politics and party loyalty"
]

def styled(fig, height=320):
    fig.update_layout(
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(color="#1E293B", size=13, family="Inter, sans-serif"),
        height=height,
        margin=dict(t=36, b=20, l=16, r=16),
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    xanchor="center", x=0.5, font=dict(size=12))
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)",
                     zeroline=False, linecolor="#E2E8F0", tickfont=dict(size=12))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.18)",
                     zeroline=False, linecolor="#E2E8F0", tickfont=dict(size=12))
    return fig

@st.cache_resource(show_spinner="Loading NLP model...")
def load_sentiment_model():
    return pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    truncation=True, max_length=512)

@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def clean_text(text):
    if not isinstance(text, str): return None
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s'!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:512] if len(text) >= 3 else None

def fetch_comments(api_key, video_id, max_results=200):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        comments, request = [], youtube.commentThreads().list(
            part="snippet", videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText", order="relevance"
        )
        while request and len(comments) < max_results:
            response = request.execute()
            for item in response.get("items", []):
                c = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({"text": c["textDisplay"],
                                  "likes": c["likeCount"],
                                  "date":  c["publishedAt"][:10]})
            request = youtube.commentThreads().list_next(request, response)
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"YouTube API Error: {e}")
        return pd.DataFrame()

def classify_cluster(text, classifier):
    if not text or not text.strip(): return CLUSTER_NAMES[2], 0.0
    try:
        result  = classifier(text, CLUSTER_LABELS, multi_label=False)
        top_idx = result["scores"].index(max(result["scores"]))
        return CLUSTER_NAMES[top_idx], round(result["scores"][top_idx], 3)
    except Exception:
        return CLUSTER_NAMES[2], 0.0

def run_analysis(df, sentiment_model, classifier):
    df["clean"] = df["text"].apply(clean_text)
    before = len(df)
    df = df[df["clean"].notna() & (df["clean"].str.strip() != "")].copy()
    dropped = before - len(df)
    if dropped > 0:
        st.info(f"ℹ️ {dropped} comments were skipped (emoji-only, links, or too short).")
    with st.spinner("Running sentiment analysis..."):
        sentiments = sentiment_model(df["clean"].tolist(), batch_size=16)
        df["sentiment"]       = [s["label"].capitalize() for s in sentiments]
        df["sentiment_score"] = [round(s["score"], 3) for s in sentiments]
    with st.spinner("Classifying semantic clusters..."):
        results            = [classify_cluster(t, classifier) for t in df["clean"]]
        df["cluster"]      = [r[0] for r in results]
        df["cluster_conf"] = [r[1] for r in results]
    return df

# ═══════════════════════════════════════════════════════════════
#  ROOT CAUSE OF 429 & THE FIX
# ───────────────────────────────────────────────────────────────
#  Problem 1 — Wrong model:
#    gemini-2.0-flash-lite free tier = 30 RPM but only 1,000 RPD
#    (requests per day). After a few runs the daily quota hit 0.
#
#  Problem 2 — Prompt too large:
#    Sending 3 full comments (up to 120 chars each) × 3 clusters
#    plus the crosstab string → ~500-800 tokens per call.
#    Combined with Streamlit re-runs this easily burns the quota.
#
#  Fix A — Switch to gemini-1.5-flash:
#    • Free tier: 15 RPM and 1,500,000 TPD (tokens/day)
#    • Far more headroom; rarely hits 429 in practice.
#
#  Fix B — Ultra-compact prompt (< 250 tokens):
#    • No comment text at all — only numeric statistics.
#    • Cluster × Sentiment counts as a compact inline string.
#    • Top keywords per cluster (5 words) instead of raw quotes.
#    • Output capped at 500 words to keep response tokens low.
#
#  Fix C — Smarter retry:
#    • Detect 429 vs other errors separately.
#    • Exponential back-off: 15 s → 30 s → 60 s (3 attempts).
#    • Show a progress bar so the user knows what's happening.
# ═══════════════════════════════════════════════════════════════

def _top_keywords(df, cluster_name, n=5):
    """Return top-n space-joined keywords for a cluster (no NLP lib needed)."""
    sub = df[df["cluster"] == cluster_name]["clean"].dropna()
    if sub.empty:
        return "—"
    all_words = " ".join(sub.tolist()).lower().split()
    stopwords = {"the","a","an","is","it","in","on","of","to","and","we",
                 "i","this","that","for","are","be","with","at","they",
                 "have","not","no","do","so","but","my","our","was","were"}
    freq = {}
    for w in all_words:
        w = re.sub(r"[^a-z]", "", w)
        if len(w) > 3 and w not in stopwords:
            freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:n]
    return ", ".join(top) if top else "—"


def build_expert_report(df, gemini_key, video_context=""):
    total   = len(df)
    pos_pct = round(len(df[df["sentiment"] == "Positive"]) / total * 100, 1)
    neg_pct = round(len(df[df["sentiment"] == "Negative"]) / total * 100, 1)
    neu_pct = round(len(df[df["sentiment"] == "Neutral"])  / total * 100, 1)

    # ── Compact cluster × sentiment line (no table, saves ~60 tokens) ──
    cluster_lines = []
    for cn in CLUSTER_NAMES:
        sub = df[df["cluster"] == cn]
        n   = len(sub)
        if n == 0:
            continue
        p = round(len(sub[sub["sentiment"] == "Positive"]) / n * 100)
        g = round(len(sub[sub["sentiment"] == "Negative"]) / n * 100)
        kw = _top_keywords(df, cn)
        cluster_lines.append(f"{cn}: {n} comments | +{p}% −{g}% | keywords: {kw}")
    cluster_block = "\n".join(cluster_lines)

    date_info    = (f"Date range: {df['date'].min()} → {df['date'].max()}"
                    if "date" in df.columns else "")
    context_note = f"Context: {video_context.strip()}" if video_context.strip() else ""

    dominant_cluster = df["cluster"].value_counts().idxmax() if not df.empty else ""
    if "Military" in dominant_cluster:
        expert_role = "a senior geopolitical and defense policy analyst"
    elif "Economic" in dominant_cluster:
        expert_role = "a senior economic analyst specializing in consumer sentiment"
    elif "Electoral" in dominant_cluster:
        expert_role = "a senior political communications strategist"
    else:
        expert_role = "a senior social media intelligence analyst"

    # ── Ultra-compact prompt — target < 250 input tokens ──────
    prompt = f"""You are {expert_role}. Write an executive-grade analysis (max 500 words) of this YouTube comment dataset.

STATS: {total} comments | Positive {pos_pct}% | Negative {neg_pct}% | Neutral {neu_pct}%
{date_info}
{context_note}

CLUSTERS:
{cluster_block}

Produce exactly 5 markdown sections:
### 1. Executive Summary
### 2. Cluster-by-Cluster Analysis
### 3. Audience Signals
### 4. Risks & Opportunities
### 5. Top 3 Business Opportunities
(For each opportunity: business type · audience insight · one concrete action)"""

    genai.configure(api_key=gemini_key)

    # ── Fix A: gemini-1.5-flash — 15 RPM, 1.5M TPD on free tier ──
    model = genai.GenerativeModel("gemini-1.5-flash")

    # ── Fix C: smarter retry with progress bar ─────────────────
    last_error = None
    waits      = [15, 30, 60]          # seconds between retries
    for attempt, wait in enumerate(waits, start=1):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            last_error = e
            err_str    = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                if attempt < len(waits):
                    st.warning(
                        f"⏳ Gemini rate limit — waiting {wait}s before retry "
                        f"({attempt}/{len(waits)})…"
                    )
                    # Show a live countdown progress bar
                    bar = st.progress(0)
                    for i in range(wait):
                        time.sleep(1)
                        bar.progress((i + 1) / wait,
                                     text=f"Retrying in {wait - i - 1}s…")
                    bar.empty()
                else:
                    # All retries exhausted — surface a clear message
                    raise RuntimeError(
                        "Gemini free-tier daily quota exhausted. "
                        "Wait ~1 minute and try again, or upgrade your API plan at "
                        "https://aistudio.google.com/app/apikey"
                    ) from e
            else:
                raise e   # non-429 error — propagate immediately

    raise last_error


DATA_FILE = "comments_data.csv"

def save_data(df):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        df = pd.concat([existing, df]).drop_duplicates(subset=["text"])
    df.to_csv(DATA_FILE, index=False)

def load_saved_data():
    if os.path.exists(DATA_FILE): return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key      = st.text_input("YouTube API Key", type="password",
                      help="Get your free key from console.cloud.google.com")
    video_url    = st.text_input("YouTube Video URL",
                      placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Max Comments to Fetch", 50, 500, 200, 50)
    fetch_btn    = st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True)
    st.divider()
    st.markdown("### 📁 Or Load Saved Data")
    load_btn     = st.button("📂 Load Previous Results", use_container_width=True)
    st.divider()
    st.markdown("### 🧪 Demo Mode")
    demo_btn     = st.button("▶️ Run with Sample Data", use_container_width=True)
    st.divider()
    st.markdown("### 🤖 Expert Analysis (Gemini)")
    gemini_key   = st.text_input("Google Gemini API Key", type="password",
                      help="Free key from aistudio.google.com/app/apikey")
    video_context = st.text_area(
        "Video / Channel Context (optional)",
        placeholder="e.g. Fox News segment on Iran nuclear talks, June 2026...",
        height=80,
        help="Short description of the video topic to sharpen the analyst's role."
    )
    st.divider()
    st.markdown("""
    **How to use:**
    1. Enter your YouTube API Key
    2. Paste any YouTube video URL
    3. Click Fetch & Analyze
    4. Add Gemini key for expert report
    """)

st.title("📊 YouTube Comment Intelligence")
st.caption("Powered by RoBERTa · Zero-Shot Classification · Gemini 1.5 Flash Expert Analysis")
st.divider()

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()
if "expert_report" not in st.session_state:
    st.session_state.expert_report = ""

DEMO_COMMENTS = [
    {"text": "We need to resume military action and finish the job completely", "likes": 312, "date": "2026-05-10"},
    {"text": "FINISH the job! No more half measures, total victory only",        "likes": 289, "date": "2026-05-10"},
    {"text": "The military option is the only language they understand",          "likes": 201, "date": "2026-05-11"},
    {"text": "Strike hard and fast, no negotiations with terrorists",             "likes": 178, "date": "2026-05-11"},
    {"text": "We should have bombed them back to stone age already",              "likes": 156, "date": "2026-05-12"},
    {"text": "Lower energy cost is what we need, not another war",               "likes": 445, "date": "2026-05-10"},
    {"text": "Nightmare happening in America with the price of oil and gas",      "likes": 398, "date": "2026-05-10"},
    {"text": "My grocery bill doubled, I dont care about foreign wars",           "likes": 356, "date": "2026-05-11"},
    {"text": "Focus on American jobs and economy, not overseas conflicts",        "likes": 321, "date": "2026-05-11"},
    {"text": "Inflation is killing us, this war just makes everything worse",     "likes": 289, "date": "2026-05-12"},
    {"text": "American families are struggling and we are spending billions abroad","likes": 267, "date": "2026-05-12"},
    {"text": "The dems get control back we are so screwed",                       "likes": 534, "date": "2026-05-10"},
    {"text": "I have lost faith in Trump on this issue honestly",                 "likes": 478, "date": "2026-05-10"},
    {"text": "Republicans better get their act together before midterms",          "likes": 412, "date": "2026-05-11"},
    {"text": "If GOP loses the house it is over for America",                     "likes": 389, "date": "2026-05-11"},
    {"text": "This will cost us the election in November mark my words",          "likes": 345, "date": "2026-05-12"},
    {"text": "Trump needs to stop listening to the warmongers in his cabinet",    "likes": 298, "date": "2026-05-12"},
    {"text": "I voted for peace not another endless war in Middle East",          "likes": 267, "date": "2026-05-13"},
    {"text": "Great leadership, this is exactly what America needed to show strength","likes": 189, "date": "2026-05-10"},
    {"text": "Strong response was necessary, Iran had to be stopped",             "likes": 167, "date": "2026-05-11"},
    {"text": "Gas prices are going through the roof because of this conflict",    "likes": 234, "date": "2026-05-11"},
    {"text": "We should protect our borders first before foreign adventures",     "likes": 198, "date": "2026-05-12"},
    {"text": "The base is fracturing, this is dangerous for the party",           "likes": 312, "date": "2026-05-13"},
    {"text": "Complete mission then come home, no nation building this time",     "likes": 145, "date": "2026-05-13"},
    {"text": "Energy independence was promised and now look at these prices",     "likes": 223, "date": "2026-05-14"},
]

if demo_btn:
    st.info("Running in Demo Mode with sample comments...")
    df_demo = pd.DataFrame(DEMO_COMMENTS)
    df_demo = run_analysis(df_demo, load_sentiment_model(), load_classifier())
    st.session_state.df = df_demo
    st.session_state.expert_report = ""

if load_btn:
    saved = load_saved_data()
    if not saved.empty:
        st.session_state.df = saved
        st.session_state.expert_report = ""
        st.success(f"Loaded {len(saved)} saved comments.")
    else:
        st.warning("No saved data found.")

if fetch_btn:
    if not api_key:
        st.sidebar.error("Please enter a YouTube API Key.")
    elif not video_url:
        st.sidebar.error("Please enter a Video URL.")
    else:
        video_id = None
        if "v=" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        if not video_id:
            st.sidebar.error("Could not extract video ID from URL.")
        else:
            with st.spinner(f"Fetching up to {max_comments} comments..."):
                df_new = fetch_comments(api_key, video_id, max_comments)
            if not df_new.empty:
                st.success(f"Fetched {len(df_new)} comments. Running analysis...")
                df_new = run_analysis(df_new, load_sentiment_model(), load_classifier())
                save_data(df_new)
                st.session_state.df = df_new
                st.session_state.expert_report = ""

df = st.session_state.df

if df.empty:
    st.markdown("""
    <div class="empty-state">
        <h2>👈 Get Started</h2>
        <p>Enter a YouTube API Key and video URL in the sidebar,<br>
        or click <strong>Run with Sample Data</strong> for a demo.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    total          = len(df)
    pos            = len(df[df["sentiment"] == "Positive"])
    neg            = len(df[df["sentiment"] == "Negative"])
    neu            = len(df[df["sentiment"] == "Neutral"])
    cluster_counts = df["cluster"].value_counts()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Comments", total)
    k2.metric("Positive 🟢",    pos, f"↑ {pos/total*100:.0f}%")
    k3.metric("Negative 🔴",    neg, f"↑ {neg/total*100:.0f}%")
    k4.metric("Neutral ⚪",     neu, f"↑ {neu/total*100:.0f}%")
    dominant  = cluster_counts.idxmax() if not cluster_counts.empty else "N/A"
    dom_parts = dominant.split(" ", 1)
    dom_label = dom_parts[1] if len(dom_parts) > 1 else dominant
    k5.metric("Dominant Cluster", dom_label)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Sentiment Distribution**")
        fig_sent = px.pie(
            names=["Positive", "Negative", "Neutral"],
            values=[pos, neg, neu],
            color=["Positive", "Negative", "Neutral"],
            color_discrete_map=SENTIMENT_COLORS, hole=0.52
        )
        fig_sent.update_traces(
            textposition="outside", textinfo="percent+label",
            marker=dict(line=dict(color="#FFFFFF", width=2)),
            pull=[0.02, 0.02, 0.02]
        )
        fig_sent.update_layout(showlegend=True)
        fig_sent = styled(fig_sent, height=310)
        st.plotly_chart(fig_sent, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Semantic Cluster Distribution**")
        cc = df["cluster"].value_counts().reset_index()
        cc.columns = ["Cluster", "Count"]
        cc["Pct"]   = (cc["Count"] / cc["Count"].sum() * 100).round(1)
        cc["Label"] = cc["Pct"].astype(str) + "%"
        fig_cluster = px.bar(
            cc, x="Count", y="Cluster", orientation="h",
            color="Cluster", color_discrete_map=CLUSTER_COLORS, text="Label"
        )
        fig_cluster.update_traces(textposition="outside",
                                   marker_line_color="#FFFFFF", marker_line_width=1.5)
        fig_cluster.update_layout(showlegend=False)
        fig_cluster = styled(fig_cluster, height=310)
        fig_cluster.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown("**Cluster x Sentiment Breakdown**")
    cross = pd.crosstab(df["cluster"], df["sentiment"]).reset_index()
    for c in ["Positive", "Negative", "Neutral"]:
        if c not in cross.columns: cross[c] = 0
    long_cross = cross.melt(id_vars="cluster", var_name="sentiment", value_name="count")
    long_cross  = long_cross[long_cross["count"] > 0]
    fig_stack   = px.bar(
        long_cross, x="count", y="cluster",
        color="sentiment", orientation="h", barmode="stack",
        color_discrete_map=SENTIMENT_COLORS, text="count"
    )
    fig_stack.update_traces(textposition="inside", insidetextanchor="middle",
                             marker_line_color="#FFFFFF", marker_line_width=1)
    fig_stack = styled(fig_stack, height=280)
    fig_stack.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_stack, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if "date" in df.columns:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("**Activity Timeline by Cluster**")
        timeline        = df.groupby(["date", "cluster"]).size().reset_index(name="count")
        active_clusters = timeline["cluster"].nunique()
        if active_clusters >= 2:
            fig_time = px.line(timeline, x="date", y="count", color="cluster",
                               color_discrete_map=CLUSTER_COLORS, markers=True)
            fig_time.update_traces(line=dict(width=2.5), marker=dict(size=7))
        else:
            fig_time = px.bar(timeline, x="date", y="count", color="cluster",
                              color_discrete_map=CLUSTER_COLORS)
        fig_time = styled(fig_time, height=320)
        fig_time.update_xaxes(title_text="Date")
        fig_time.update_yaxes(title_text="Comments")
        st.plotly_chart(fig_time, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("🔍 Top Comments by Cluster")
    tab1, tab2, tab3 = st.tabs(CLUSTER_NAMES)
    SENT_ICON = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}
    for tab, cluster_name, css_cls in zip([tab1, tab2, tab3], CLUSTER_NAMES, CLUSTER_CSS):
        with tab:
            sub = df[df["cluster"] == cluster_name].copy()
            if "likes" in sub.columns: sub = sub.sort_values("likes", ascending=False)
            if sub.empty:
                st.info("No comments in this cluster.")
            for _, row in sub.head(8).iterrows():
                icon      = SENT_ICON.get(row.get("sentiment", ""), "⚪")
                likes_str = f"👍 {int(row['likes'])}" if "likes" in row and pd.notna(row["likes"]) else ""
                conf_str  = f"Conf: {row.get('cluster_conf', '')}"
                st.markdown(f"""
                <div class="cluster-card {css_cls}">
                    <p style="color:#0F172A; font-size:15px; margin:0 0 8px 0; line-height:1.5;">{row['text']}</p>
                    <small style="color:#64748B; font-size:12px;">{icon} {row.get('sentiment','')} &nbsp;·&nbsp; {conf_str} &nbsp;·&nbsp; {likes_str}</small>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    st.subheader("📥 Export Results")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full CSV", csv_data,
            "youtube_sentiment_results.csv", "text/csv", use_container_width=True)
    with col_dl2:
        summary     = df.groupby(["cluster", "sentiment"]).size().reset_index(name="count")
        summary_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Summary CSV", summary_csv,
            "cluster_summary.csv", "text/csv", use_container_width=True)

    # ═══════════════════════════════════════════════════════════
    #  EXPERT ANALYSIS SECTION — Gemini 1.5 Flash (Free tier)
    # ═══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🤖 Expert Analyst Report")
    st.caption("Powered by Gemini 1.5 Flash · Compact prompt · Live retry countdown")

    report_col, _ = st.columns([3, 1])
    with report_col:
        generate_btn = st.button(
            "✨ Generate Expert Report",
            type="primary",
            use_container_width=True,
            disabled=(not gemini_key),
            help="Add your Gemini API Key in the sidebar to enable this feature."
        )

    if not gemini_key:
        st.info("🔑 Add your free **Google Gemini API Key** in the sidebar to unlock the expert report.  \nGet it free at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)")

    if generate_btn and gemini_key:
        with st.spinner("🤖 Analyst is reading the data and writing the report..."):
            try:
                report = build_expert_report(df, gemini_key, video_context)
                st.session_state.expert_report = report
            except Exception as e:
                st.error(f"Gemini API error: {e}")

    if st.session_state.expert_report:
        report_text = st.session_state.expert_report
        dominant_cl = df["cluster"].value_counts().idxmax() if not df.empty else "General"
        st.markdown('<div class="report-card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="report-meta">
            <span class="report-badge">✦ Gemini 1.5 Flash</span>
            <span class="report-badge" style="background:#7C3AED;">Senior Analyst</span>
            <span style="color:#64748B; font-size:13px;">{total} comments · {len(df['cluster'].unique())} clusters · dominant: {dominant_cl}</span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(report_text)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=report_text.encode("utf-8"),
            file_name="expert_analyst_report.md",
            mime="text/markdown",
            use_container_width=False
        )
