import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from tabsearch import  HybridVectorizer
from tabsearch.datasets import load_sp500_demo
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Page Configuration & Styling
# --------------------------
st.set_page_config(
    page_title="TabSearch Demo | Mixed-Data Vector Search", 
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for comparison-focused design
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    
    .reference-company {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        position: sticky;
        top: 0;
        z-index: 100;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }
    
    .metric-badge {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        backdrop-filter: blur(10px);
    }
    
    .results-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .result-column {
        flex: 1;
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .column-header {
        padding: 1rem;
        font-weight: bold;
        text-align: center;
        color: white;
    }
    
    .text-header {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
    }
    
    .mixed-header {
        background: linear-gradient(135deg, #00d2d3, #54a0ff);
    }
    
    .result-row {
        padding: 0.75rem;
        border-bottom: 1px solid #f0f0f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .result-row:hover {
        background-color: #f8f9fa;
    }
    
    .company-info {
        flex: 1;
    }
    
    .company-name {
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.25rem;
    }
    
    .company-details {
        font-size: 0.85rem;
        color: #666;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    
    .match-badge {
        background: #27ae60;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .partial-badge {
        background: #f39c12;
        color: white;
        padding: 0.15rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        font-weight: bold;
    }
    
    .similarity-score {
        text-align: right;
        margin-left: 1rem;
    }
    
    .score-high { color: #27ae60; font-weight: bold; }
    .score-medium { color: #f39c12; font-weight: bold; }
    .score-low { color: #e74c3c; font-weight: bold; }
    
    .analysis-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------
# Data Loading & Model Setup
# --------------------------
@st.cache_data
def load_data():
    df = load_sp500_demo()
    return df[["Symbol", "Sector", "Industry", "Currentprice", "Marketcap",
               "Fulltimeemployees", "Longbusinesssummary"]].copy()

@st.cache_resource
def get_hv(df):
    hv = HybridVectorizer(index_column="Symbol")
    hv.fit_transform(df)
    return hv

@st.cache_resource
def get_text_search(df):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = df.astype(str).agg(" ".join, axis=1).tolist()
    embeddings = model.encode(texts, show_progress_bar=False)
    return model, embeddings

# Load data and models
df = load_data()
hv = get_hv(df)
model, embeddings = get_text_search(df)

# --------------------------
# Sidebar Configuration
# --------------------------
with st.sidebar:
    st.header("🔍 Search Configuration")
    
    symbol = st.selectbox(
        "**Reference Company:**",
        df["Symbol"].unique(),
        help="Select the company to find similar matches for"
    )
    
    top_k = st.slider("**Number of Results:**", 3, 10, 5)
    
    st.markdown("---")
    st.subheader("⚖️ Weight Adjustment")
    st.caption("Fine-tune how different data types contribute to similarity")
    
    text_w = st.slider("Text Weight", 0.0, 5.0, 1.0, 0.25)
    num_w = st.slider("Numerical Weight", 0.0, 5.0, 1.0, 0.25)
    cat_w = st.slider("Categorical Weight", 0.0, 5.0, 1.0, 0.25)
    
    if text_w == 1.0 and num_w == 1.0 and cat_w == 1.0:
        weights = None
        st.success("✅ Using balanced weights")
    else:
        weights = {"text": text_w, "numerical": num_w, "categorical": cat_w}
        st.info("🎛️ Using custom weights")

# --------------------------
# Helper Functions
# --------------------------
def format_market_cap(value):
    if value >= 1e12:
        return f"${value/1e12:.1f}T"
    elif value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.0f}M"
    else:
        return f"${value:,.0f}"

def format_employees(value):
    if value >= 1e6:
        return f"{value/1e6:.1f}M"
    elif value >= 1e3:
        return f"{value/1e3:.0f}K"
    else:
        return f"{value:,.0f}"

def get_similarity_class(score):
    if score >= 0.4:
        return "score-high"
    elif score >= 0.25:
        return "score-medium"
    else:
        return "score-low"

def check_matches(ref_company, result_company):
    badges = []
    
    # Sector match
    if ref_company['Sector'] == result_company['Sector']:
        badges.append('<span class="match-badge">SECTOR</span>')
    
    # Industry match
    if ref_company['Industry'] == result_company['Industry']:
        badges.append('<span class="match-badge">INDUSTRY</span>')
    
    # Market cap similarity (within 50%)
    ref_cap = ref_company['Marketcap']
    res_cap = result_company['Marketcap']
    if abs(ref_cap - res_cap) / max(ref_cap, res_cap) < 0.5:
        badges.append('<span class="partial-badge">SIMILAR CAP</span>')
    
    # Price similarity (within 30%)
    ref_price = ref_company['Currentprice']
    res_price = result_company['Currentprice']
    if abs(ref_price - res_price) / max(ref_price, res_price) < 0.3:
        badges.append('<span class="partial-badge">SIMILAR PRICE</span>')
    
    return ' '.join(badges)

# --------------------------
# Get Reference Company Data
# --------------------------
ref_company = df[df["Symbol"] == symbol].iloc[0]

# --------------------------
# Reference Company Display (Sticky)
# --------------------------
st.markdown(f"""
<div class="reference-company">
    <h2>🎯 Reference Company: {ref_company['Symbol']}</h2>
    <div style="font-size: 1.1em; margin-bottom: 0.5rem;"><strong>{ref_company['Industry']}</strong></div>
    <div class="metric-row">
        <div class="metric-badge">📊 {ref_company['Sector']}</div>
        <div class="metric-badge">💰 {format_market_cap(ref_company['Marketcap'])}</div>
        <div class="metric-badge">💵 ${ref_company['Currentprice']:.2f}</div>
        <div class="metric-badge">👥 {format_employees(ref_company['Fulltimeemployees'])}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------
# Run Search Functions
# --------------------------
def run_text_search():
    idx = df.index[df["Symbol"] == symbol][0]
    sims = cosine_similarity([embeddings[idx]], embeddings)[0]
    sims[idx] = -1
    top_ids = sims.argsort()[::-1][:top_k]
    return df.iloc[top_ids].assign(similarity=sims[top_ids])

def run_tabsearch():
    query = df.loc[df["Symbol"] == symbol].iloc[0].to_dict()
    return hv.similarity_search(
        query,
        ignore_exact_matches=True,
        top_n=top_k,
        block_weights=weights if weights else None
    )

# Execute searches
text_results = run_text_search()
tab_results = run_tabsearch()

# --------------------------
# Main Comparison Display
# --------------------------
st.markdown("## 🔬 Side-by-Side Comparison")

# Create two columns for comparison
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="result-column">
        <div class="column-header text-header">
            🔤 Text-Only Embedding Results
        </div>
    """, unsafe_allow_html=True)
    
    for i, (_, row) in enumerate(text_results.iterrows(), 1):
        similarity_class = get_similarity_class(row['similarity'])
        match_badges = check_matches(ref_company, row)
        
        st.markdown(f"""
        <div class="result-row">
            <div class="company-info">
                <div class="company-name">{i}. {row['Symbol']} - {row['Industry']}</div>
                <div class="company-details">
                    <span>📊 {row['Sector']}</span>
                    <span>💰 {format_market_cap(row['Marketcap'])}</span>
                    <span>💵 ${row['Currentprice']:.2f}</span>
                    <span>👥 {format_employees(row['Fulltimeemployees'])}</span>
                </div>
                <div style="margin-top: 0.25rem;">{match_badges}</div>
            </div>
            <div class="similarity-score">
                <div class="{similarity_class}" style="font-size: 1.2em;">{row['similarity']:.3f}</div>
                <div style="font-size: 0.8em; color: #666;">similarity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="result-column">
        <div class="column-header mixed-header">
            📊 Mixed-Data Vectorizer Results
        </div>
    """, unsafe_allow_html=True)
    
    for i, (_, row) in enumerate(tab_results.iterrows(), 1):
        similarity_class = get_similarity_class(row['similarity'])
        match_badges = check_matches(ref_company, row)
        
        st.markdown(f"""
        <div class="result-row">
            <div class="company-info">
                <div class="company-name">{i}. {row['Symbol']} - {row['Industry']}</div>
                <div class="company-details">
                    <span>📊 {row['Sector']}</span>
                    <span>💰 {format_market_cap(row['Marketcap'])}</span>
                    <span>💵 ${row['Currentprice']:.2f}</span>
                    <span>👥 {format_employees(row['Fulltimeemployees'])}</span>
                </div>
                <div style="margin-top: 0.25rem;">{match_badges}</div>
            </div>
            <div class="similarity-score">
                <div class="{similarity_class}" style="font-size: 1.2em;">{row['similarity']:.3f}</div>
                <div style="font-size: 0.8em; color: #666;">similarity</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Quick Insights
# --------------------------
st.markdown("## 🔍 Key Insights")

col1, col2, col3 = st.columns(3)

with col1:
    text_avg = text_results['similarity'].mean()
    st.metric("Text-Only Avg Similarity", f"{text_avg:.3f}")

with col2:
    tab_avg = tab_results['similarity'].mean()
    st.metric("Mixed-Data Avg Similarity", f"{tab_avg:.3f}")
    
with col3:
    # Count sector matches
    text_sector_matches = sum(1 for _, row in text_results.iterrows() if row['Sector'] == ref_company['Sector'])
    tab_sector_matches = sum(1 for _, row in tab_results.iterrows() if row['Sector'] == ref_company['Sector'])
    st.metric("Sector Matches", f"Text: {text_sector_matches} | Mixed: {tab_sector_matches}")

# --------------------------
# Analysis Section
# --------------------------
with st.expander("📊 Detailed Analysis", expanded=False):
    
    # Similarity score comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            x=[f"{row['Symbol']}" for _, row in text_results.iterrows()], 
            y=text_results['similarity'],
            title="Text-Only Similarity Scores",
            color=text_results['similarity'],
            color_continuous_scale="Reds"
        )
        fig1.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(
            x=[f"{row['Symbol']}" for _, row in tab_results.iterrows()], 
            y=tab_results['similarity'],
            title="Mixed-Data Similarity Scores",
            color=tab_results['similarity'],
            color_continuous_scale="Blues"
        )
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Sector distribution
    st.markdown("### Sector Distribution in Results")
    col1, col2 = st.columns(2)
    
    with col1:
        text_sectors = text_results['Sector'].value_counts()
        fig3 = px.pie(values=text_sectors.values, names=text_sectors.index, 
                      title="Text-Only Results")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        tab_sectors = tab_results['Sector'].value_counts()
        fig4 = px.pie(values=tab_sectors.values, names=tab_sectors.index, 
                      title="Mixed-Data Results")
        st.plotly_chart(fig4, use_container_width=True)

# --------------------------
# Raw Data Tables
# --------------------------
with st.expander("📋 Raw Data Comparison", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text-Only Results")
        st.dataframe(
            text_results[['Symbol', 'Sector', 'Industry', 'Currentprice', 'Marketcap', 'similarity']],
            use_container_width=True
        )
    
    with col2:
        st.subheader("Mixed-Data Results")
        st.dataframe(
            tab_results[['Symbol', 'Sector', 'Industry', 'Currentprice', 'Marketcap', 'similarity']],
            use_container_width=True
        )

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p><strong>TabSearch Demo</strong> - Compare text-only vs mixed-data similarity search</p>
    <p>🟢 <strong>SECTOR/INDUSTRY</strong> = Exact match | 🟡 <strong>SIMILAR CAP/PRICE</strong> = Close match</p>
</div>
""", unsafe_allow_html=True)