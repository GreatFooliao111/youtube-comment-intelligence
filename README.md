# 📊 YouTube Comment Intelligence

> Semantic sentiment analysis of YouTube comments — powered by **RoBERTa**, **Zero-Shot Classification**, and **Gemini 2.0 Flash** expert reporting.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Feature | Details |
|---|---|
| **Comment Fetching** | YouTube Data API v3 — up to 500 comments per run |
| **Sentiment Analysis** | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) — 99%+ accuracy |
| **Semantic Clustering** | Zero-Shot Classification (BART MNLI) into 3 audience segments |
| **Interactive Dashboard** | Plotly charts: sentiment pie, cluster bar, cross-matrix, activity timeline |
| **🤖 Expert AI Report** | Gemini 2.0 Flash — dynamic analyst role + 3 business opportunity recommendations |
| **Export** | Full CSV + summary CSV + expert report `.md` download |
| **Demo Mode** | Run with built-in sample data — no API keys required |

---

## 🧠 How the Expert Report Works

After sentiment analysis is complete, the app sends a structured data digest to **Gemini 2.0 Flash**, which assumes a **dynamic expert role** based on the dominant comment cluster:

| Dominant Cluster | Analyst Role Assigned |
|---|---|
| 🪖 Military/Hawkish | Senior Geopolitical & Defense Policy Analyst |
| 💰 Economic/Domestic | Senior Economic Analyst (Consumer Sentiment) |
| 🗳️ Electoral/Party | Senior Political Communications Strategist |
| Other | Senior Social Media Intelligence Analyst |

The report includes:
1. **Executive Summary** — dominant mood and key signal
2. **Cluster-by-Cluster Analysis** — tone, themes, notable examples
3. **Audience Intent & Behavior Signals** — what this audience is ready to do
4. **Risk & Opportunity Signals** — narratives gaining or losing momentum
5. **Top 3 Business Opportunities** — specific, data-grounded recommendations

---

## 🚀 Deploy on Streamlit Cloud (Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account and select this repo
4. Set `app.py` as the main file → click **Deploy**

> ⏱️ **First run:** ~5 min to download NLP models (~2 GB). Subsequent runs use cache.

---

## 🔑 API Keys Required

### YouTube Data API (for live comment fetching)
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create credentials → **API Key**
5. Paste into the sidebar

### Google Gemini API (for expert report — free tier)
1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API Key**
4. Paste into the sidebar under *Expert Analysis*

> **Free tier limits:** 15 requests/min · 1,500 requests/day — more than enough for analysis sessions.

---

## 💻 Local Development

```bash
git clone https://github.com/GreatFooliao111/youtube-comment-intelligence.git
cd youtube-comment-intelligence
pip install -r requirements.txt
streamlit run app.py
```

---

## 🏗️ Architecture

```
YouTube Data API v3
        │
        ▼
  Comment Fetcher
  (up to 500 comments)
        │
        ▼
   Text Cleaner
   (URLs, mentions, noise removal)
        │
        ├──────────────────────────┐
        ▼                          ▼
RoBERTa Sentiment         BART Zero-Shot
(Positive/Negative/        Cluster Classifier
 Neutral + score)          (Military / Economic
                            / Electoral)
        │                          │
        └──────────┬───────────────┘
                   ▼
        Streamlit Dashboard
        (Charts · Tables · CSV Export)
                   │
                   ▼
        Gemini 2.0 Flash
        Expert Analyst Report
        (Dynamic Role · Business Opportunities)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI & Dashboard | Streamlit |
| Sentiment Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` (HuggingFace) |
| Cluster Classifier | `facebook/bart-large-mnli` (Zero-Shot, HuggingFace) |
| Expert Report | Google Gemini 2.0 Flash (`google-generativeai`) |
| Charts | Plotly Express |
| Data | YouTube Data API v3 (`google-api-python-client`) |

---

## 📄 License

MIT — free to use, modify, and deploy.
