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
    overflow: hidden;
}
.chart-title {
    font-size: 15px;
    font-weight: 700;
    color: #1E293B;
    margin: 0 0 4px 2px;
    padding: 0;
    line-height: 1.4;
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
        # t=8 because title is now rendered as HTML above the chart
        margin=dict(t=8, b=20, l=16, r=16),
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
#  STRUCTURED-COMPACT PAYLOAD HELPERS
# ═══════════════════════════════════════════════════════════════

def build_cluster_summary(df):
    lines = []
    for cn in CLUSTER_NAMES:
        sub = df[df["cluster"] == cn]
        n = len(sub)
        if n == 0:
            continue
        pos = round(len(sub[sub["sentiment"] == "Positive"]) / n * 100)
        neg = round(len(sub[sub["sentiment"] == "Negative"]) / n * 100)
        neu = 100 - pos - neg
        avg_conf = round(sub["cluster_conf"].mean(), 2) if "cluster_conf" in sub else "N/A"
        share = round(n / len(df) * 100, 1)
        lines.append(
            f"• {cn}: {n} comments ({share}% of total) | "
            f"+{pos}% pos / -{neg}% neg / ~{neu}% neu | avg confidence: {avg_conf}"
        )
    return "\n".join(lines)


def extract_top_keywords_with_counts(df, cluster_name, n=8):
    stopwords = {
        "the","a","an","is","it","in","on","of","to","and","we","i","this",
        "that","for","are","be","with","at","they","have","not","no","do",
        "so","but","my","our","was","were","he","she","his","her","you",
        "your","its","who","what","just","get","got","been","will","can",
        "us","im","dont","its","thats","would","could","should","really",
        "about","more","than","also","like","very","all","from","one","has"
    }
    sub = df[df["cluster"] == cluster_name]["clean"].dropna()
    if sub.empty:
        return "—"
    freq = {}
    for text in sub:
        for w in text.lower().split():
            w = re.sub(r"[^a-z]", "", w)
            if len(w) > 3 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:n]
    return ", ".join(f"{w}({freq[w]})" for w in top) if top else "—"


def select_representative_comments(df, cluster_name, n=3, max_chars=200):
    sub = df[df["cluster"] == cluster_name].copy()
    if sub.empty:
        return []
    selected = []
    for sentiment in ["Positive", "Negative", "Neutral"]:
        pool = sub[sub["sentiment"] == sentiment].sort_values(
            "cluster_conf", ascending=False
        )
        if not pool.empty:
            row = pool.iloc[0]
            text = str(row["clean"])[:max_chars].strip()
            if text:
                selected.append({
                    "text": text,
                    "sentiment": row["sentiment"],
                    "likes": int(row.get("likes", 0))
                })
        if len(selected) >= n:
            break
    if len(selected) < n:
        already = {s["text"] for s in selected}
        extras = sub.sort_values("cluster_conf", ascending=False)
        for _, row in extras.iterrows():
            text = str(row["clean"])[:max_chars].strip()
            if text not in already:
                selected.append({
                    "text": text,
                    "sentiment": row["sentiment"],
                    "likes": int(row.get("likes", 0))
                })
                already.add(text)
            if len(selected) >= n:
                break
    return selected


def select_outlier_signals(df, n=5):
    if "likes" not in df.columns or df.empty:
        return []
    top = df.nlargest(n, "likes")[["clean", "sentiment", "cluster", "likes"]].dropna(subset=["clean"])
    result = []
    for _, row in top.iterrows():
        text = str(row["clean"])[:200].strip()
        if text:
            result.append({
                "text": text,
                "sentiment": row["sentiment"],
                "cluster": row["cluster"],
                "likes": int(row["likes"])
            })
    return result


def build_cross_cluster_signals(df):
    neg_df = df[df["sentiment"] == "Negative"]
    pos_df = df[df["sentiment"] == "Positive"]
    dom_grievance = (
        neg_df["cluster"].value_counts().idxmax()
        if not neg_df.empty else "N/A"
    )
    dom_support = (
        pos_df["cluster"].value_counts().idxmax()
        if not pos_df.empty else "N/A"
    )
    cluster_pos_rates = {}
    for cn in CLUSTER_NAMES:
        sub = df[df["cluster"] == cn]
        if len(sub) > 0:
            cluster_pos_rates[cn] = round(
                len(sub[sub["sentiment"] == "Positive"]) / len(sub) * 100, 1
            )
    if len(cluster_pos_rates) >= 2:
        best = max(cluster_pos_rates, key=cluster_pos_rates.get)
        worst = min(cluster_pos_rates, key=cluster_pos_rates.get)
        gap = cluster_pos_rates[best] - cluster_pos_rates[worst]
        gap_note = (
            f"Sentiment gap: {best} ({cluster_pos_rates[best]}% pos) vs "
            f"{worst} ({cluster_pos_rates[worst]}% pos) — Δ{gap:.1f}pp"
        )
    else:
        gap_note = "Insufficient clusters for gap analysis."
    return {
        "dominant_grievance_cluster": dom_grievance,
        "dominant_support_cluster": dom_support,
        "sentiment_gap": gap_note
    }


def _determine_expert_role(df):
    dominant = df["cluster"].value_counts().idxmax() if not df.empty else ""
    if "Military" in dominant:
        return "a senior geopolitical and defense policy analyst"
    elif "Economic" in dominant:
        return "a senior economic analyst specializing in consumer sentiment"
    elif "Electoral" in dominant:
        return "a senior political communications strategist"
    return "a senior social media intelligence analyst"


def build_expert_report(df, gemini_key, video_context=""):
    total   = len(df)
    pos_pct = round(len(df[df["sentiment"] == "Positive"]) / total * 100, 1)
    neg_pct = round(len(df[df["sentiment"] == "Negative"]) / total * 100, 1)
    neu_pct = round(len(df[df["sentiment"] == "Neutral"])  / total * 100, 1)
    date_range = (
        f"{df['date'].min()} → {df['date'].max()}"
        if "date" in df.columns else "unknown"
    )

    block1 = (
        f"Total comments: {total} | Positive: {pos_pct}% | "
        f"Negative: {neg_pct}% | Neutral: {neu_pct}% | Date range: {date_range}"
    )
    if video_context.strip():
        block1 += f"\nVideo context: {video_context.strip()}"

    block2 = build_cluster_summary(df)

    kw_lines = []
    for cn in CLUSTER_NAMES:
        kw = extract_top_keywords_with_counts(df, cn, n=8)
        kw_lines.append(f"• {cn}: {kw}")
    block3 = "\n".join(kw_lines)

    rep_lines = []
    for cn in CLUSTER_NAMES:
        reps = select_representative_comments(df, cn, n=3, max_chars=200)
        if reps:
            rep_lines.append(f"\n{cn}:")
            for r in reps:
                rep_lines.append(
                    f'  [{r["sentiment"]} | ❤️{r["likes"]}] "{r["text"]}"'
                )
    block4 = "\n".join(rep_lines) if rep_lines else "No representative comments available."

    outliers = select_outlier_signals(df, n=5)
    if outliers:
        out_lines = []
        for o in outliers:
            out_lines.append(
                f'  [{o["cluster"]} | {o["sentiment"]} | ❤️{o["likes"]}] "{o["text"]}"'
            )
        block5 = "\n".join(out_lines)
    else:
        block5 = "No high-engagement outliers detected."

    cross = build_cross_cluster_signals(df)
    block6 = (
        f"Dominant grievance cluster: {cross['dominant_grievance_cluster']}\n"
        f"Dominant support cluster: {cross['dominant_support_cluster']}\n"
        f"{cross['sentiment_gap']}"
    )

    expert_role = _determine_expert_role(df)

    system_prompt = (
        f"You are {expert_role} writing for a senior strategy team.\n"
        "Rules:\n"
        "- Use ONLY the data provided below. Do not invent events, demographics, or motives.\n"
        "- Clearly separate direct observations (what the data shows) from inferences (what it implies).\n"
        "- Prioritize contradictions, emotional intensity, and cross-cluster tensions over majority sentiment.\n"
        "- Focus on actionable market and communications signals.\n"
        "- Write in concise, high-value markdown. Avoid filler phrases.\n"
        "- Do not exceed 600 words in your response."
    )

    user_payload = f"""Analyze this YouTube comment intelligence summary.

[GLOBAL SUMMARY]
{block1}

[CLUSTER SUMMARIES]
{block2}

[TOP KEYWORDS PER CLUSTER]
{block3}

[REPRESENTATIVE COMMENTS]
{block4}

[HIGH-ENGAGEMENT OUTLIERS]
{block5}

[CROSS-CLUSTER SIGNALS]
{block6}

Write exactly these 6 sections in markdown:
### 1. Executive Summary
### 2. What the Audience Cares About
### 3. Cluster-by-Cluster Dynamics
### 4. Risks, Tensions & Opportunity Windows
### 5. Top 3 Business Opportunities
(For each: business type · key audience insight · one concrete action step)
### 6. Recommended Next Actions"""

    full_prompt = f"{system_prompt}\n\n{user_payload}"

    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    waits     = [15, 30, 60]
    last_error = None
    for attempt, wait in enumerate(waits, start=1):
        try:
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            last_error = e
            err_str    = str(e).lower()
            is_rate    = "429" in str(e) or "quota" in err_str or "rate" in err_str
            is_rpd     = "daily" in err_str or "per day" in err_str or "rpd" in err_str

            if is_rate:
                if is_rpd:
                    raise RuntimeError(
                        "🚫 Gemini **daily quota (RPD)** exhausted for this project.\n\n"
                        "This quota resets at **midnight Pacific Time** (not UTC).\n"
                        "Switching API keys will NOT help — quotas are per project.\n"
                        "Options:\n"
                        "• Wait until midnight PT and try again.\n"
                        "• Upgrade your plan at https://aistudio.google.com/app/apikey"
                    ) from e
                if attempt < len(waits):
                    st.warning(
                        f"⏳ Gemini rate limit (RPM/TPM) — waiting {wait}s before retry "
                        f"({attempt}/{len(waits)})…"
                    )
                    bar = st.progress(0)
                    for i in range(wait):
                        time.sleep(1)
                        bar.progress((i + 1) / wait,
                                     text=f"Retrying in {wait - i - 1}s…")
                    bar.empty()
                else:
                    raise RuntimeError(
                        "⏱️ Gemini rate limit persists after 3 retries (RPM/TPM).\n"
                        "Wait ~1 minute and try again, or check your quota at "
                        "https://aistudio.google.com/app/apikey"
                    ) from e
            else:
                raise e

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
    {"text": "We need to resume military action and finish the job completely", "likes": 312, "date": "2026-05-01"},
    {"text": "The ceasefire was a mistake, we should never have agreed to it", "likes": 289, "date": "2026-05-01"},
    {"text": "Our soldiers are heroes and deserve full support from the government", "likes": 445, "date": "2026-05-02"},
    {"text": "Stop the war now, too many innocent people are dying every day", "likes": 178, "date": "2026-05-02"},
    {"text": "The military leadership has no clue what they are doing", "likes": 134, "date": "2026-05-03"},
    {"text": "Inflation is destroying middle class families and nobody cares", "likes": 567, "date": "2026-05-03"},
    {"text": "I cannot afford groceries anymore, this government has failed us", "likes": 612, "date": "2026-05-04"},
    {"text": "Gas prices are through the roof and wages have not moved in years", "likes": 398, "date": "2026-05-04"},
    {"text": "The economy is in shambles and the politicians just keep lying", "likes": 445, "date": "2026-05-05"},
    {"text": "Small businesses are closing every week in my city, its devastating", "likes": 321, "date": "2026-05-05"},
    {"text": "Tax cuts for the rich while working people struggle to survive", "likes": 289, "date": "2026-05-06"},
    {"text": "Housing costs are impossible for young people now, where do we live", "likes": 376, "date": "2026-05-06"},
    {"text": "The opposition party is corrupt and incompetent, vote them all out", "likes": 234, "date": "2026-05-07"},
    {"text": "Both parties are the same, just serving their corporate donors", "likes": 456, "date": "2026-05-07"},
    {"text": "The election was rigged and everyone knows it but nobody says so", "likes": 189, "date": "2026-05-08"},
    {"text": "We need new leadership that actually listens to ordinary citizens", "likes": 312, "date": "2026-05-08"},
    {"text": "Politicians spend all their time on TV instead of solving problems", "likes": 267, "date": "2026-05-09"},
    {"text": "I am proud of our nation and believe we will get through this together", "likes": 198, "date": "2026-05-09"},
    {"text": "The media never reports the truth about what is really happening", "likes": 423, "date": "2026-05-10"},
    {"text": "Young people are leaving the country because there is no future here", "likes": 334, "date": "2026-05-10"},
]

def run_demo():
    demo_df = pd.DataFrame(DEMO_COMMENTS)
    sentiment_model = load_sentiment_model()
    classifier      = load_classifier()
    return run_analysis(demo_df, sentiment_model, classifier)

def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([^&\n?#]+)", url)
    return match.group(1) if match else None

# ─── Trigger Actions ────────────────────────────────────────────
if fetch_btn:
    if not api_key:
        st.error("Please enter your YouTube API Key.")
    elif not video_url:
        st.error("Please enter a YouTube Video URL.")
    else:
        video_id = extract_video_id(video_url)
        if not video_id:
            st.error("Could not parse video ID from URL.")
        else:
            with st.spinner("Fetching comments..."):
                raw_df = fetch_comments(api_key, video_id, max_comments)
            if raw_df.empty:
                st.error("No comments fetched. Check your API key and video URL.")
            else:
                sentiment_model = load_sentiment_model()
                classifier      = load_classifier()
                st.session_state.df = run_analysis(raw_df, sentiment_model, classifier)
                save_data(st.session_state.df)
                st.session_state.expert_report = ""
                st.success(f"✅ Fetched and analyzed {len(st.session_state.df)} comments.")

if load_btn:
    loaded = load_saved_data()
    if loaded.empty:
        st.warning("No saved data found.")
    else:
        st.session_state.df = loaded
        st.session_state.expert_report = ""
        st.success(f"✅ Loaded {len(loaded)} saved comments.")

if demo_btn:
    with st.spinner("Running demo analysis..."):
        st.session_state.df = run_demo()
    st.session_state.expert_report = ""
    st.success("✅ Demo analysis complete!")

# ─── Main Dashboard ─────────────────────────────────────────────
df = st.session_state.df

if df.empty:
    st.markdown("""
    <div class="empty-state">
        <h2>📭 No Data Yet</h2>
        <p>Enter a YouTube URL and API key in the sidebar, or run the demo to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    # ── KPI Row ────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Comments", f"{len(df):,}")
    k2.metric("Positive", f"{round(len(df[df['sentiment']=='Positive'])/len(df)*100,1)}%",
              delta=f"{len(df[df['sentiment']=='Positive'])} comments")
    k3.metric("Negative", f"{round(len(df[df['sentiment']=='Negative'])/len(df)*100,1)}%",
              delta=f"{len(df[df['sentiment']=='Negative'])} comments")
    k4.metric("Neutral",  f"{round(len(df[df['sentiment']=='Neutral'])/len(df)*100,1)}%",
              delta=f"{len(df[df['sentiment']=='Neutral'])} comments")

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🗂️ Clusters", "📈 Trends", "🤖 Expert Report"])

    # ── Tab 1: Overview ────────────────────────────────────────
    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                '<div class="chart-card">'
                '<p class="chart-title">Sentiment Distribution</p>',
                unsafe_allow_html=True
            )
            sent_counts = df["sentiment"].value_counts().reset_index()
            sent_counts.columns = ["sentiment", "count"]
            fig = px.pie(sent_counts, values="count", names="sentiment",
                         color="sentiment", color_discrete_map=SENTIMENT_COLORS)
            st.plotly_chart(styled(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown(
                '<div class="chart-card">'
                '<p class="chart-title">Comments per Cluster</p>',
                unsafe_allow_html=True
            )
            clust_counts = df["cluster"].value_counts().reset_index()
            clust_counts.columns = ["cluster", "count"]
            fig2 = px.bar(clust_counts, x="cluster", y="count",
                          color="cluster", color_discrete_map=CLUSTER_COLORS)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(styled(fig2), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Download ──────────────────────────────────────────
        st.divider()
        st.download_button(
            "⬇️ Download Full Dataset (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            "youtube_comments_analysis.csv", "text/csv"
        )

    # ── Tab 2: Clusters ────────────────────────────────────────
    with tab2:
        for i, cn in enumerate(CLUSTER_NAMES):
            sub = df[df["cluster"] == cn]
            if sub.empty: continue
            css = CLUSTER_CSS[i]
            pos = len(sub[sub["sentiment"] == "Positive"])
            neg = len(sub[sub["sentiment"] == "Negative"])
            top_comment = sub.sort_values("likes", ascending=False).iloc[0]["text"] \
                          if "likes" in sub.columns else sub.iloc[0]["text"]
            st.markdown(f"""
            <div class="cluster-card {css}">
                <b>{cn}</b> — {len(sub)} comments
                &nbsp;&nbsp;✅ {pos} positive &nbsp; ❌ {neg} negative<br>
                <small style="color:#64748B">Top comment: {str(top_comment)[:180]}…</small>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Sentiment × Cluster Heatmap")
        pivot = df.pivot_table(index="cluster", columns="sentiment",
                               values="text", aggfunc="count", fill_value=0)
        fig3 = px.imshow(pivot, text_auto=True, aspect="auto",
                         color_continuous_scale="Blues", title="Comment Count Heatmap")
        st.plotly_chart(styled(fig3, height=280), use_container_width=True)

    # ── Tab 3: Trends ──────────────────────────────────────────
    with tab3:
        if "date" in df.columns:
            daily = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
            fig4  = px.line(daily, x="date", y="count", color="sentiment",
                            color_discrete_map=SENTIMENT_COLORS,
                            title="Daily Comment Volume by Sentiment",
                            markers=True)
            st.plotly_chart(styled(fig4, height=360), use_container_width=True)

            daily_c = df.groupby(["date", "cluster"]).size().reset_index(name="count")
            fig5 = px.area(daily_c, x="date", y="count", color="cluster",
                           color_discrete_map=CLUSTER_COLORS,
                           title="Daily Cluster Volume")
            st.plotly_chart(styled(fig5, height=320), use_container_width=True)
        else:
            st.info("Date information not available for trend analysis.")

    # ── Tab 4: Expert Report ───────────────────────────────────
    with tab4:
        st.markdown("### 🤖 Gemini Expert Analysis")
        st.caption(
            "Uses **gemini-1.5-flash** with a structured 6-block payload (~700–1,200 tokens). "
            "Exponential back-off handles RPM, TPM, and RPD limits automatically."
        )
        gen_btn = st.button("✨ Generate Expert Report", type="primary")

        if gen_btn:
            if not gemini_key:
                st.error("Please enter your Gemini API Key in the sidebar.")
            else:
                with st.spinner("Generating expert analysis…"):
                    try:
                        report = build_expert_report(df, gemini_key, video_context)
                        st.session_state.expert_report = report
                    except RuntimeError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")

        if st.session_state.expert_report:
            st.markdown('<div class="report-card">', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="report-meta">
                <span class="report-badge">AI ANALYSIS</span>
                <span style="color:#64748B;font-size:13px;">
                    Model: gemini-1.5-flash &nbsp;·&nbsp;
                    Dataset: {len(df):,} comments &nbsp;·&nbsp;
                    Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}
                </span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(st.session_state.expert_report)
            st.markdown('</div>', unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download Report (Markdown)",
                st.session_state.expert_report.encode("utf-8"),
                "expert_report.md", "text/markdown"
            )
