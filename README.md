# 📊 YouTube Comment Intelligence

A Streamlit app for semantic analysis of YouTube comments using **RoBERTa** and **Zero-Shot Classification**.

## Features

- Fetches YouTube comments via official API
- Sentiment analysis with RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Semantic clustering into 3 groups: **Military/Hawkish**, **Economic/Domestic**, **Electoral/Party**
- Interactive Plotly dashboard: pie charts, heatmap, timeline
- Export results to CSV
- Demo mode with sample data (no API key needed)

## 🚀 Deploy on Streamlit Cloud (Free)

1. Fork this repo to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo and set `app.py` as the main file
5. Click **Deploy**

> **Note:** First run takes ~5 min to download NLP models (~2GB). Subsequent runs use cache.

## 🔑 Get YouTube API Key (Free)

1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a new project
3. Enable **YouTube Data API v3**
4. Create credentials → API Key
5. Paste into the sidebar

## 💻 Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Architecture

```
YouTube API → Comment Fetcher → Text Cleaner
                                      ↓
                              RoBERTa Sentiment
                                      ↓
                         Zero-Shot Cluster Classifier
                                      ↓
                    CSV Storage + Streamlit Dashboard
```
