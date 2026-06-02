import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Evaluation Metrics", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #0891b2 0%, #06b6d4 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1e293b;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    .score-high {
        color: #22c55e;
        font-weight: bold;
    }
    .score-medium {
        color: #eab308;
        font-weight: bold;
    }
    .score-low {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">📊 RAG Evaluation Metrics Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("Evaluation Configuration")

time_range = st.sidebar.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"])
metric_filter = st.sidebar.multiselect(
    "Metrics to Display",
    ["Faithfulness", "Context Precision", "Answer Relevance", "Context Recall", "MRR"],
    default=["Faithfulness", "Context Precision", "Answer Relevance"]
)

# Generate simulated historical data
np.random.seed(42)

def generate_historical_data(days=30):
    """Generate simulated historical evaluation metrics with daily averages."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=days, freq='D')
    
    # Generate data with realistic daily variation and trends
    base_faithfulness = 0.90
    base_context_precision = 0.85
    base_answer_relevance = 0.80
    base_context_recall = 0.75
    base_mrr = 0.85
    
    # Add some trend and noise
    trend = np.linspace(0, 0.05, days)  # Slight improvement over time
    noise = np.random.normal(0, 0.02, days)
    
    data = {
        "date": dates,
        "faithfulness": np.clip(base_faithfulness + trend + noise, 0.70, 0.99),
        "context_precision": np.clip(base_context_precision + trend * 0.8 + noise * 0.9, 0.65, 0.98),
        "answer_relevance": np.clip(base_answer_relevance + trend * 0.9 + noise * 0.85, 0.60, 0.97),
        "context_recall": np.clip(base_context_recall + trend * 0.7 + noise * 0.8, 0.55, 0.95),
        "mrr": np.clip(base_mrr + trend * 0.85 + noise * 0.88, 0.65, 0.98),
        "query_count": np.random.randint(50, 200, days)
    }
    
    df = pd.DataFrame(data)
    
    # Format dates to show only date (not time) for cleaner display
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    
    return df

# Get data based on time range
if time_range == "Last 24 Hours":
    historical_data = generate_historical_data(1)
elif time_range == "Last 7 Days":
    historical_data = generate_historical_data(7)
elif time_range == "Last 30 Days":
    historical_data = generate_historical_data(30)
else:
    historical_data = generate_historical_data(90)

# Current metrics summary
st.subheader("🎯 Current Performance Summary")

latest_data = historical_data.iloc[-1]

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    faithfulness_score = latest_data["faithfulness"]
    score_class = "score-high" if faithfulness_score > 0.9 else "score-medium" if faithfulness_score > 0.7 else "score-low"
    st.metric("Faithfulness", f"{faithfulness_score:.3f}", delta=f"{faithfulness_score - 0.9:+.3f}")

with col2:
    context_precision = latest_data["context_precision"]
    st.metric("Context Precision", f"{context_precision:.3f}", delta=f"{context_precision - 0.85:+.3f}")

with col3:
    answer_relevance = latest_data["answer_relevance"]
    st.metric("Answer Relevance", f"{answer_relevance:.3f}", delta=f"{answer_relevance - 0.8:+.3f}")

with col4:
    context_recall = latest_data["context_recall"]
    st.metric("Context Recall", f"{context_recall:.3f}", delta=f"{context_recall - 0.75:+.3f}")

with col5:
    mrr_score = latest_data["mrr"]
    st.metric("MRR", f"{mrr_score:.3f}", delta=f"{mrr_score - 0.85:+.3f}")

# Historical trends
st.subheader("📈 Historical Trends")

fig = go.Figure()

if "Faithfulness" in metric_filter:
    fig.add_trace(go.Scatter(
        x=historical_data["date"],
        y=historical_data["faithfulness"],
        mode='lines+markers',
        name='Faithfulness',
        line=dict(color='#22c55e', width=2)
    ))

if "Context Precision" in metric_filter:
    fig.add_trace(go.Scatter(
        x=historical_data["date"],
        y=historical_data["context_precision"],
        mode='lines+markers',
        name='Context Precision',
        line=dict(color='#3b82f6', width=2)
    ))

if "Answer Relevance" in metric_filter:
    fig.add_trace(go.Scatter(
        x=historical_data["date"],
        y=historical_data["answer_relevance"],
        mode='lines+markers',
        name='Answer Relevance',
        line=dict(color='#f59e0b', width=2)
    ))

if "Context Recall" in metric_filter:
    fig.add_trace(go.Scatter(
        x=historical_data["date"],
        y=historical_data["context_recall"],
        mode='lines+markers',
        name='Context Recall',
        line=dict(color='#8b5cf6', width=2)
    ))

if "MRR" in metric_filter:
    fig.add_trace(go.Scatter(
        x=historical_data["date"],
        y=historical_data["mrr"],
        mode='lines+markers',
        name='MRR',
        line=dict(color='#ec4899', width=2)
    ))

fig.update_layout(
    title=f"RAG Metrics Over Time ({time_range})",
    xaxis_title="Date",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.1]),
    height=400,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Query volume
st.subheader("📊 Query Volume")

fig_volume = go.Figure()

fig_volume.add_trace(go.Bar(
    x=historical_data["date"],
    y=historical_data["query_count"],
    name='Query Count',
    marker_color='#06b6d4'
))

fig_volume.update_layout(
    title=f"Daily Query Volume ({time_range})",
    xaxis_title="Date",
    yaxis_title="Number of Queries",
    height=300
)

st.plotly_chart(fig_volume, use_container_width=True)

# Metric distribution
st.subheader("📊 Metric Distribution")

fig_dist = go.Figure()

metrics_to_plot = [m for m in metric_filter if m in historical_data.columns]
for metric in metrics_to_plot:
    fig_dist.add_trace(go.Box(
        y=historical_data[metric],
        name=metric,
        boxpoints='outliers'
    ))

fig_dist.update_layout(
    title="Metric Score Distribution",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.1]),
    height=400
)

st.plotly_chart(fig_dist, use_container_width=True)

# Correlation analysis
st.subheader("🔗 Metric Correlation Analysis")

if len(metrics_to_plot) > 1:
    correlation_matrix = historical_data[metrics_to_plot].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix.values],
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig_corr.update_layout(
        title="Metric Correlation Matrix",
        height=400
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)

# Performance thresholds
st.subheader("⚠️ Performance Thresholds")

thresholds = {
    "Faithfulness": 0.90,
    "Context Precision": 0.85,
    "Answer Relevance": 0.80,
    "Context Recall": 0.75,
    "MRR": 0.85
}

threshold_data = []
for metric, threshold in thresholds.items():
    current_value = latest_data[metric.lower().replace(" ", "_")]
    status = "✅ Above Threshold" if current_value >= threshold else "⚠️ Below Threshold"
    threshold_data.append({
        "Metric": metric,
        "Current": current_value,
        "Threshold": threshold,
        "Status": status,
        "Gap": current_value - threshold
    })

threshold_df = pd.DataFrame(threshold_data)

st.dataframe(
    threshold_df.style.format({
        "Current": "{:.3f}",
        "Threshold": "{:.3f}",
        "Gap": "{:+.3f}"
    }).apply(
        lambda row: ['color: red' if row['Gap'] < 0 else 'color: green'] * len(row),
        axis=1
    ),
    use_container_width=True,
    hide_index=True
)

# Detailed analysis
st.subheader("🔍 Detailed Analysis")

with st.expander("View Performance Insights"):
    st.markdown("""
    **Overall System Health:**
    - The RAG system is performing {"excellently" if latest_data["faithfulness"] > 0.9 else "well" if latest_data["faithfulness"] > 0.8 else "adequately"} based on current metrics
    - Faithfulness scores indicate {"low hallucination risk" if latest_data["faithfulness"] > 0.9 else "moderate hallucination risk" if latest_data["faithfulness"] > 0.8 else "high hallucination risk"}
    - Context precision shows {"strong relevance grounding" if latest_data["context_precision"] > 0.85 else "acceptable relevance grounding"}
    
    **Recommendations:**
    """)
    
    if latest_data["faithfulness"] < 0.9:
        st.warning("- Consider improving context quality or adjusting retrieval parameters to reduce hallucinations")
    if latest_data["context_precision"] < 0.85:
        st.warning("- Review retrieval thresholds and consider implementing re-ranking")
    if latest_data["answer_relevance"] < 0.8:
        st.warning("- Optimize prompt engineering and LLM parameters for better answer relevance")
    
    st.success("""
    - Continue monitoring metrics trends over time
    - Set up automated alerts for metrics falling below thresholds
    - Regularly evaluate with new test datasets to ensure robustness
    """)

# Export functionality
st.subheader("📥 Export Data")

if st.button("Export Metrics CSV", use_container_width=True):
    csv = historical_data.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"rag_metrics_{time_range.lower().replace(' ', '_')}.csv",
        mime="text/csv"
    )
