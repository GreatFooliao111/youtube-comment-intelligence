import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from googleapiclient.discovery import build
from transformers import pipeline
import re
import datetime
import os

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube Comment Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1c1f2e; border-radius: 10px; padding: 12px; }
    .cluster-card {
        background: #1c1f2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .cluster-military { border-color: #ef4444; }
    .cluster-economic { border-color: #f59e0b; }
    .cluster-electoral { border-color: #3b82f6; }
    h1 { color: #e2e8f0; }
    .stSidebar { background-color: #161b2e; }
</style>
""", unsafe_allow_html=True)

# ─── Cache: Load Sentiment Model ───────────────────────────────
@st.cache_resource(show_spinner="Loading NLP model...")
def load_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        truncation=True,
        max_length=512
    )

# ─── Cache: Load Zero-Shot Classifier ──────────────────────────
@st.cache_resource(show_spinner="Loading classifier...")
def load_classifier():
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli"
    )

# ─── Helper: Clean Text ─────────────────────────────────────────
# FIX: returns None if text becomes empty after cleaning
def clean_text(text):
    if not isinstance(text, str):
        return None
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    # Keep letters, digits, basic punctuation — strip pure-emoji / symbol comments
    text = re.sub(r"[^\w\s\'!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Minimum 3 characters to be useful for NLP
    if len(text) < 3:
        return None
    return text[:512]

# ─── Helper: Fetch YouTube Comments ────────────────────────────
def fetch_comments(api_key, video_id, max_results=200):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_results, 100),
            textFormat="plainText",
            order="relevance"
        )
        while request and len(comments) < max_results:
            response = request.execute()
            for item in response.get("items", []):
                c = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "text": c["textDisplay"],
                    "likes": c["likeCount"],
                    "date": c["publishedAt"][:10]
                })
            request = youtube.commentThreads().list_next(request, response)
        return pd.DataFrame(comments)
    except Exception as e:
        st.error(f"YouTube API Error: {e}")
        return pd.DataFrame()

# ─── Helper: Classify Clusters ─────────────────────────────────
CLUSTER_LABELS = [
    "military action and war support",
    "economic concerns inflation and cost of living",
    "electoral politics and party loyalty"
]
CLUSTER_NAMES  = ["🪖 Military/Hawkish", "💰 Economic/Domestic", "🗳️ Electoral/Party"]
CLUSTER_COLORS = ["#ef4444", "#f59e0b", "#3b82f6"]

def classify_cluster(text, classifier):
    # FIX: guard against None or empty strings
    if not text or not text.strip():
        return CLUSTER_NAMES[2], 0.0  # default fallback bucket
    try:
        result = classifier(text, CLUSTER_LABELS, multi_label=False)
        top_idx = result["scores"].index(max(result["scores"]))
        return CLUSTER_NAMES[top_idx], round(result["scores"][top_idx], 3)
    except Exception:
        return CLUSTER_NAMES[2], 0.0

# ─── Helper: Run Full Analysis ──────────────────────────────────
def run_analysis(df, sentiment_model, classifier):
    df["clean"] = df["text"].apply(clean_text)

    # FIX: drop rows where clean is None/empty — they break the models
    before = len(df)
    df = df[df["clean"].notna() & (df["clean"].str.strip() != "")].copy()
    dropped = before - len(df)
    if dropped > 0:
        st.info(f"ℹ️ {dropped} comments were skipped (emoji-only, links, or too short).")

    with st.spinner("Running sentiment analysis..."):
        sentiments = sentiment_model(df["clean"].tolist(), batch_size=16)
        df["sentiment"] = [s["label"].capitalize() for s in sentiments]
        df["sentiment_score"] = [round(s["score"], 3) for s in sentiments]

    with st.spinner("Classifying semantic clusters..."):
        results = [classify_cluster(t, classifier) for t in df["clean"]]
        df["cluster"] = [r[0] for r in results]
        df["cluster_conf"] = [r[1] for r in results]

    return df

# ─── Helper: Save/Load CSV ─────────────────────────────────────
DATA_FILE = "comments_data.csv"

def save_data(df):
    if os.path.exists(DATA_FILE):
        existing = pd.read_csv(DATA_FILE)
        df = pd.concat([existing, df]).drop_duplicates(subset=["text"])
    df.to_csv(DATA_FILE, index=False)

def load_saved_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    api_key = st.text_input("YouTube API Key", type="password",
        help="Get your free key from console.cloud.google.com")
    video_url = st.text_input("YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("Max Comments to Fetch", 50, 500, 200, 50)
    fetch_btn = st.button("🚀 Fetch & Analyze", type="primary", use_container_width=True)
    st.divider()
    st.markdown("### 📁 Or Load Saved Data")
    load_btn = st.button("📂 Load Previous Results", use_container_width=True)
    st.divider()
    st.markdown("### 🧪 Demo Mode")
    demo_btn = st.button("▶️ Run with Sample Data", use_container_width=True)
    st.divider()
    st.markdown("""
    **How to use:**
    1. Enter your YouTube API Key
    2. Paste any YouTube video URL
    3. Click Fetch & Analyze
    4. Explore the results
    """)

# ═══════════════════════════════════════════════════════════════
#  MAIN TITLE
# ═══════════════════════════════════════════════════════════════
st.title("📊 YouTube Comment Intelligence")
st.caption("Powered by RoBERTa · Zero-Shot Classification · Real-time Semantic Clustering")
st.divider()

if "df" not in st.session_state:
    st.session_state.df = pd.DataFrame()

# ─── Demo Data ─────────────────────────────────────────────────
DEMO_COMMENTS = [
    {"text": "We need to resume military action and finish the job completely", "likes": 312, "date": "2026-05-10"},
    {"text": "FINISH the job! No more half measures, total victory only", "likes": 289, "date": "2026-05-10"},
    {"text": "The military option is the only language they understand", "likes": 201, "date": "2026-05-11"},
    {"text": "Strike hard and fast, no negotiations with terrorists", "likes": 178, "date": "2026-05-11"},
    {"text": "We should have bombed them back to stone age already", "likes": 156, "date": "2026-05-12"},
    {"text": "Lower energy cost is what we need, not another war", "likes": 445, "date": "2026-05-10"},
    {"text": "Nightmare happening in America with the price of oil and gas", "likes": 398, "date": "2026-05-10"},
    {"text": "My grocery bill doubled, I dont care about foreign wars", "likes": 356, "date": "2026-05-11"},
    {"text": "Focus on American jobs and economy, not overseas conflicts", "likes": 321, "date": "2026-05-11"},
    {"text": "Inflation is killing us, this war just makes everything worse", "likes": 289, "date": "2026-05-12"},
    {"text": "American families are struggling and we are spending billions abroad", "likes": 267, "date": "2026-05-12"},
    {"text": "The dems get control back we are so screwed", "likes": 534, "date": "2026-05-10"},
    {"text": "I have lost faith in Trump on this issue honestly", "likes": 478, "date": "2026-05-10"},
    {"text": "Republicans better get their act together before midterms", "likes": 412, "date": "2026-05-11"},
    {"text": "If GOP loses the house it is over for America", "likes": 389, "date": "2026-05-11"},
    {"text": "This will cost us the election in November mark my words", "likes": 345, "date": "2026-05-12"},
    {"text": "Trump needs to stop listening to the warmongers in his cabinet", "likes": 298, "date": "2026-05-12"},
    {"text": "I voted for peace not another endless war in Middle East", "likes": 267, "date": "2026-05-13"},
    {"text": "Great leadership, this is exactly what America needed to show strength", "likes": 189, "date": "2026-05-10"},
    {"text": "Strong response was necessary, Iran had to be stopped", "likes": 167, "date": "2026-05-11"},
    {"text": "Gas prices are going through the roof because of this conflict", "likes": 234, "date": "2026-05-11"},
    {"text": "We should protect our borders first before foreign adventures", "likes": 198, "date": "2026-05-12"},
    {"text": "The base is fracturing, this is dangerous for the party", "likes": 312, "date": "2026-05-13"},
    {"text": "Complete mission then come home, no nation building this time", "likes": 145, "date": "2026-05-13"},
    {"text": "Energy independence was promised and now look at these prices", "likes": 223, "date": "2026-05-14"},
]

# ─── Triggers ──────────────────────────────────────────────────
if demo_btn:
    st.info("Running in Demo Mode with sample comments...")
    df_demo = pd.DataFrame(DEMO_COMMENTS)
    sentiment_model = load_sentiment_model()
    classifier = load_classifier()
    df_demo = run_analysis(df_demo, sentiment_model, classifier)
    st.session_state.df = df_demo

if load_btn:
    saved = load_saved_data()
    if not saved.empty:
        st.session_state.df = saved
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
                sentiment_model = load_sentiment_model()
                classifier = load_classifier()
                df_new = run_analysis(df_new, sentiment_model, classifier)
                save_data(df_new)
                st.session_state.df = df_new

# ═══════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════
df = st.session_state.df

if df.empty:
    st.markdown("""
    <div style="text-align:center; padding: 80px 20px; color: #64748b;">
        <h2>👈 Get Started</h2>
        <p>Enter a YouTube API Key and video URL in the sidebar,<br>
        or click <strong>Run with Sample Data</strong> for a demo.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    total = len(df)
    pos = len(df[df["sentiment"] == "Positive"])
    neg = len(df[df["sentiment"] == "Negative"])
    neu = len(df[df["sentiment"] == "Neutral"])
    cluster_counts = df["cluster"].value_counts()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Comments", total)
    k2.metric("Positive 🟢", pos, f"{pos/total*100:.0f}%")
    k3.metric("Negative 🔴", neg, f"{neg/total*100:.0f}%")
    k4.metric("Neutral ⚪", neu, f"{neu/total*100:.0f}%")
    dominant = cluster_counts.idxmax() if not cluster_counts.empty else "N/A"
    k5.metric("Dominant Cluster", dominant.split(" ", 1)[-1][:20])

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Distribution")
        fig_sent = px.pie(
            names=["Positive", "Negative", "Neutral"],
            values=[pos, neg, neu],
            color_discrete_sequence=["#22c55e", "#ef4444", "#94a3b8"],
            hole=0.4
        )
        fig_sent.update_traces(textposition="inside", textinfo="percent+label")
        fig_sent.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320,
            margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_sent, use_container_width=True)

    with col2:
        st.subheader("Semantic Cluster Distribution")
        cc = df["cluster"].value_counts().reset_index()
        cc.columns = ["Cluster", "Count"]
        fig_cluster = px.pie(cc, names="Cluster", values="Count",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6"], hole=0.4)
        fig_cluster.update_traces(textposition="inside", textinfo="percent+label")
        fig_cluster.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320,
            margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.subheader("Cluster × Sentiment Matrix")
    cross = pd.crosstab(df["cluster"], df["sentiment"])
    fig_heat = px.imshow(cross, color_continuous_scale="RdYlGn", text_auto=True, aspect="auto")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0", height=300, margin=dict(t=20,b=20,l=10,r=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    if "date" in df.columns:
        st.subheader("Activity Timeline by Cluster")
        timeline = df.groupby(["date", "cluster"]).size().reset_index(name="count")
        fig_time = px.line(timeline, x="date", y="count", color="cluster",
            color_discrete_sequence=["#ef4444", "#f59e0b", "#3b82f6"], markers=True)
        fig_time.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0", legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=320, margin=dict(t=30,b=20,l=10,r=10))
        st.plotly_chart(fig_time, use_container_width=True)

    st.divider()
    st.subheader("🔍 Top Comments by Cluster")
    tab1, tab2, tab3 = st.tabs(CLUSTER_NAMES)
    css_classes = ["cluster-military", "cluster-economic", "cluster-electoral"]
    for tab, cluster_name, css_class in zip([tab1, tab2, tab3], CLUSTER_NAMES, css_classes):
        with tab:
            sub = df[df["cluster"] == cluster_name].copy()
            if "likes" in sub.columns:
                sub = sub.sort_values("likes", ascending=False)
            for _, row in sub.head(8).iterrows():
                sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "⚪"}.get(row.get("sentiment", ""), "⚪")
                likes_str = f"👍 {int(row['likes'])}" if "likes" in row and pd.notna(row["likes"]) else ""
                st.markdown(f"""
                <div class="cluster-card {css_class}">
                    <p style="color:#e2e8f0; margin:0 0 6px 0;">{row['text']}</p>
                    <small style="color:#64748b;">{sentiment_color} {row.get('sentiment','')} · Conf: {row.get('cluster_conf','')} {likes_str}</small>
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
        summary = df.groupby(["cluster", "sentiment"]).size().reset_index(name="count")
        summary_csv = summary.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Summary CSV", summary_csv,
            "cluster_summary.csv", "text/csv", use_container_width=True)
