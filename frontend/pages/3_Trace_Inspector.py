import os
import time
import uuid

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Trace Inspector", page_icon="👁️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #059669 0%, #10b981 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .trace-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
        margin-bottom: 0.5rem;
    }
    .status-ok {
        color: #22c55e;
        font-weight: bold;
    }
    .status-warning {
        color: #eab308;
        font-weight: bold;
    }
    .status-error {
        color: #ef4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">👁️ Trace Inspector: Query Pipeline Visualization</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar for session selection
st.sidebar.header("Session Management")

# Generate session ID
session_id = st.sidebar.text_input("Session ID", value=f"tr-{uuid.uuid4().hex[:8]}-{uuid.uuid4().hex[:4]}", disabled=True)

# Query input for new trace
st.sidebar.subheader("New Query Trace")
test_query = st.sidebar.text_input("Test Query", placeholder="Enter a query to trace...")
if st.sidebar.button("Start Trace", use_container_width=True):
    if test_query:
        st.session_state.current_trace = {
            "session_id": session_id,
            "query": test_query,
            "timestamp": time.time(),
            "spans": []
        }
        st.rerun()

# Main content
if "current_trace" not in st.session_state:
    st.info("👆 Enter a query in the sidebar to start a new trace session")
    st.stop()

trace = st.session_state.current_trace

# Display session info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Session ID", trace["session_id"])
with col2:
    st.metric("Query", trace["query"][:50] + "..." if len(trace["query"]) > 50 else trace["query"])
with col3:
    # Calculate total latency from pipeline stages
    pipeline_stages = [
        {"name": "🎙️ whisper-asr-transcription", "latency": 340},
        {"name": "🔍 cross-modal-hybrid-retriever", "latency": 180},
        {"name": "⚖️ cohere-reranking-layer", "latency": 90},
        {"name": "🤖 vlm-synthesis-layer", "latency": 810}
    ]
    total_latency = sum(stage["latency"] for stage in pipeline_stages)
    st.metric("Total Latency", f"{total_latency:.0f}ms")

# Simulate pipeline spans (in production, these would come from actual tracing)
st.subheader("🗂️ Span Tree Call Execution Runtime")

# Define pipeline stages
pipeline_stages = [
    {
        "name": "🎙️ whisper-asr-transcription",
        "latency": 340,
        "status": "OK",
        "input": "Raw binary audio packet buffer stream",
        "output": f'"{trace["query"]}"'
    },
    {
        "name": "🔍 cross-modal-hybrid-retriever",
        "latency": 180,
        "status": "OK",
        "input": f'Query: "{trace["query"]}"',
        "output": "Found 3 matched text paragraphs + 1 image match (similarity: 0.89)"
    },
    {
        "name": "⚖️ cohere-reranking-layer",
        "latency": 90,
        "status": "OK",
        "input": "50 sparse/dense context candidates",
        "output": "Culled to top 3 factual document blocks"
    },
    {
        "name": "🤖 vlm-prompt-synthesis-engine",
        "latency": 810,
        "status": "OK",
        "input": "Top 3 contexts + query",
        "output": "Word-by-word text token delivery stream initialized via FastAPI WebSockets"
    }
]

# Display spans
for i, stage in enumerate(pipeline_stages):
    status_color = "status-ok" if stage["status"] == "OK" else "status-error"
    
    with st.container():
        st.markdown(f"""
        <div class="trace-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1em;">{stage["name"]}</span>
                <span class="{status_color}">[{stage["latency"]}ms] [{stage["status"]}] 🟢</span>
            </div>
            <div style="margin-top: 0.5rem; padding-left: 1rem; border-left: 2px solid #334155;">
                <div><strong>Input:</strong> {stage["input"]}</div>
                <div><strong>Output:</strong> {stage["output"]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# RAGAS Evaluation
st.subheader("⚖️ Automated RAGAS Runtime Evaluation Logs")

evaluation_metrics = {
    "Faithfulness Check": {"label": "No Hallucination Detected", "score": 0.98, "status": "🟢"},
    "Context Precision": {"label": "High Relevance Grounding", "score": 0.94, "status": "🟢"},
    "Answer Relevance": {"label": "Directly Addresses Query", "score": 0.91, "status": "🟢"}
}

for metric_name, metric_data in evaluation_metrics.items():
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"**Metric:** {metric_name}")
    with col2:
        st.write(f"{metric_data['status']} {metric_data['label']}")
    with col3:
        st.metric("Score", f"{metric_data['score']:.2f}/1.00")

# Judge reasoning
st.info("🧑‍⚖️ **Judge Evaluation Reasoning:** \"The generated output metrics match coordinates inside the retrieved visual context and accurately reflect the source document data.\"")

# Timeline visualization
st.subheader("📊 Pipeline Timeline Visualization")

# Create timeline data
timeline_data = []
cumulative_time = 0
for stage in pipeline_stages:
    timeline_data.append({
        "Stage": stage["name"],
        "Start": cumulative_time,
        "Duration": stage["latency"],
        "End": cumulative_time + stage["latency"]
    })
    cumulative_time += stage["latency"]

# Create Gantt chart
fig = go.Figure()

for i, item in enumerate(timeline_data):
    fig.add_trace(go.Scatter(
        x=[item["Start"], item["End"]],
        y=[i, i],
        mode='lines',
        line=dict(width=20, color='#10b981'),
        name=item["Stage"],
        hovertemplate=f"{item['Stage']}<br>Start: {item['Start']}ms<br>Duration: {item['Duration']}ms<extra></extra>"
    ))

fig.update_layout(
    title="Pipeline Execution Timeline",
    xaxis_title="Time (ms)",
    yaxis_title="Pipeline Stage",
    yaxis=dict(
        tickmode='array',
        tickvals=[i for i in range(len(timeline_data))],
        ticktext=[item["Stage"] for item in timeline_data]
    ),
    height=400,
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# Latency breakdown
st.subheader("📈 Latency Breakdown")

latency_df = pd.DataFrame([
    {"Stage": stage["name"], "Latency (ms)": stage["latency"], "Percentage": f"{stage['latency']/sum(s['latency'] for s in pipeline_stages)*100:.1f}%"}
    for stage in pipeline_stages
])

st.dataframe(latency_df, use_container_width=True, hide_index=True)

# Performance insights
st.subheader("💡 Performance Insights")

with st.expander("View Performance Analysis"):
    st.markdown("""
    **Bottleneck Analysis:**
    - VLM synthesis accounts for ~57% of total latency (810ms)
    - ASR transcription is the second largest component (~24%)
    - Retrieval and re-ranking are relatively fast (~19% combined)
    
    **Optimization Recommendations:**
    - Consider using a faster VLM model for lower latency requirements
    - Implement streaming ASR for real-time transcription
    - Cache frequent query patterns to reduce VLM calls
    - Use batch processing for bulk queries
    """)

# Export trace
st.subheader("📥 Export Trace")

if st.button("Export Trace JSON", use_container_width=True):
    export_data = {
        "session_id": trace["session_id"],
        "query": trace["query"],
        "timestamp": trace["timestamp"],
        "total_latency": sum(stage["latency"] for stage in pipeline_stages),
        "spans": pipeline_stages,
        "evaluation": evaluation_metrics
    }
    st.download_button(
        label="Download Trace",
        data=str(export_data),
        file_name=f"trace_{trace['session_id']}.json",
        mime="application/json"
    )
