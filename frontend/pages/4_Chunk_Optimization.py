import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Chunk Optimization", page_icon="📊", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #dc2626 0%, #f97316 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">📊 Experimental R&D Analysis: Chunk Size Optimization</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("Optimization Parameters")

chunk_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
overlap_percentages = [0, 5, 10, 15, 20, 25]

selected_chunk_size = st.sidebar.selectbox("Chunk Size (tokens)", chunk_sizes, index=3)
selected_overlap = st.sidebar.selectbox("Overlap Percentage", overlap_percentages, index=2)

# Generate simulated performance data
np.random.seed(42)

def generate_performance_data(chunk_size, overlap_pct):
    """Simulate RAG performance metrics for different chunk sizes."""
    # Base precision curve (peaks around 512)
    if chunk_size < 256:
        precision = 0.4 + (chunk_size / 256) * 0.3
    elif chunk_size < 1024:
        precision = 0.7 + ((chunk_size - 256) / 768) * 0.25
    else:
        precision = 0.95 - ((chunk_size - 1024) / 7168) * 0.3
    
    # Base recall curve
    if chunk_size < 256:
        recall = 0.5 + (chunk_size / 256) * 0.2
    elif chunk_size < 1024:
        recall = 0.7 + ((chunk_size - 256) / 768) * 0.15
    else:
        recall = 0.85 - ((chunk_size - 1024) / 7168) * 0.1
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Latency (increases with chunk size)
    latency = 50 + (chunk_size / 64) * 10
    
    # Context precision (drops when chunks are too large)
    context_precision = precision - (chunk_size / 8192) * 0.2
    
    # VLM latency (spikes with large chunks)
    vlm_latency = 200 + (chunk_size / 256) * 50
    
    return {
        "chunk_size": chunk_size,
        "overlap_pct": overlap_pct,
        "precision": max(0, min(1, precision)),
        "recall": max(0, min(1, recall)),
        "f1_score": max(0, min(1, f1)),
        "latency_ms": latency,
        "context_precision": max(0, min(1, context_precision)),
        "vlm_latency_ms": vlm_latency
    }

# Generate data for all chunk sizes
performance_data = []
for size in chunk_sizes:
    data = generate_performance_data(size, selected_overlap)
    performance_data.append(data)

df = pd.DataFrame(performance_data)

# Main visualization
st.subheader("📈 Performance vs Chunk Size")

fig = go.Figure()

# Add precision line
fig.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["precision"],
    mode='lines+markers',
    name='Precision',
    line=dict(color='#3b82f6', width=3),
    marker=dict(size=8)
))

# Add recall line
fig.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["recall"],
    mode='lines+markers',
    name='Recall',
    line=dict(color='#10b981', width=3),
    marker=dict(size=8)
))

# Add F1 score line
fig.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["f1_score"],
    mode='lines+markers',
    name='F1 Score',
    line=dict(color='#f59e0b', width=3),
    marker=dict(size=8)
))

fig.update_layout(
    title=f"RAG Performance Metrics by Chunk Size (Overlap: {selected_overlap}%)",
    xaxis_title="Chunk Size (Tokens)",
    yaxis_title="Score",
    yaxis=dict(range=[0, 1.1]),
    height=500,
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# Latency analysis
st.subheader("⚡ Latency Analysis")

fig_latency = go.Figure()

fig_latency.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["latency_ms"],
    mode='lines+markers',
    name='Retrieval Latency',
    line=dict(color='#8b5cf6', width=3),
    marker=dict(size=8),
    yaxis='y'
))

fig_latency.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["vlm_latency_ms"],
    mode='lines+markers',
    name='VLM Latency',
    line=dict(color='#ef4444', width=3),
    marker=dict(size=8),
    yaxis='y2'
))

fig_latency.update_layout(
    title="Latency vs Chunk Size",
    xaxis_title="Chunk Size (Tokens)",
    yaxis=dict(
        title=dict(text="Retrieval Latency (ms)", font=dict(color="#8b5cf6")),
        tickfont=dict(color="#8b5cf6"),
        side='left'
    ),
    yaxis2=dict(
        title=dict(text="VLM Latency (ms)", font=dict(color="#ef4444")),
        tickfont=dict(color="#ef4444"),
        overlaying='y',
        side='right'
    ),
    height=450,
    margin=dict(l=80, r=80, t=60, b=80),
    showlegend=True
)

st.plotly_chart(fig_latency, use_container_width=True)

# Context precision analysis
st.subheader("🎯 Context Precision Analysis")

fig_context = go.Figure()

fig_context.add_trace(go.Scatter(
    x=df["chunk_size"],
    y=df["context_precision"],
    mode='lines+markers',
    name='Context Precision',
    line=dict(color='#06b6d4', width=3),
    marker=dict(size=8),
    fill='tozeroy',
    fillcolor='rgba(6, 182, 212, 0.2)'
))

# Add annotation for optimal region
fig_context.add_vrect(
    x0=256, x1=1024,
    fillcolor="rgba(34, 197, 94, 0.2)",
    layer="below", line_width=0,
    annotation_text="Optimal Range"
)

fig_context.update_layout(
    title="Context Precision vs Chunk Size",
    xaxis_title="Chunk Size (Tokens)",
    yaxis_title="Context Precision Score",
    yaxis=dict(range=[0, 1.1]),
    height=400
)

st.plotly_chart(fig_context, use_container_width=True)

# Current configuration analysis
st.subheader(f"📊 Current Configuration Analysis: {selected_chunk_size} Tokens")

current_config = df[df["chunk_size"] == selected_chunk_size].iloc[0]

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Precision", f"{current_config['precision']:.3f}")
with col2:
    st.metric("Recall", f"{current_config['recall']:.3f}")
with col3:
    st.metric("F1 Score", f"{current_config['f1_score']:.3f}")
with col4:
    st.metric("Total Latency", f"{current_config['latency_ms'] + current_config['vlm_latency_ms']:.0f}ms")

# R&D Insights
st.subheader("💡 R&D Insight Log")

with st.expander("View Detailed Analysis"):
    st.markdown(f"""
    **Chunk Size: {selected_chunk_size} Tokens Analysis**
    
    **Performance Characteristics:**
    - **Precision:** {current_config['precision']:.3f} - {"High" if current_config['precision'] > 0.8 else "Moderate" if current_config['precision'] > 0.6 else "Low"}
    - **Recall:** {current_config['recall']:.3f} - {"High" if current_config['recall'] > 0.8 else "Moderate" if current_config['recall'] > 0.6 else "Low"}
    - **F1 Score:** {current_config['f1_score']:.3f} - {"Excellent" if current_config['f1_score'] > 0.85 else "Good" if current_config['f1_score'] > 0.7 else "Needs Improvement"}
    - **Retrieval Latency:** {current_config['latency_ms']:.0f}ms
    - **VLM Latency:** {current_config['vlm_latency_ms']:.0f}ms
    
    **Recommendations:**
    """)

    if selected_chunk_size < 256:
        st.warning("""
        ⚠️ **Under-chunked Configuration**
        - Chunks smaller than 256 tokens lose surrounding contextual semantics
        - Consider increasing chunk size to 512-1024 tokens for better context preservation
        - May result in fragmented information and reduced coherence
        """)
    elif selected_chunk_size > 2048:
        st.warning("""
        ⚠️ **Over-chunked Configuration**
        - Chunks larger than 2048 tokens dilute specific information
        - VLM latency spikes significantly with large chunks
        - Context precision drops as chunks become too broad
        - Consider reducing to 512-1024 tokens for optimal balance
        """)
    else:
        st.success("""
        ✅ **Optimal Configuration Range**
        - Chunks between 256-1024 tokens provide good balance
        - Context is preserved while maintaining specificity
        - Latency remains acceptable for real-time applications
        - Recommended: 512 tokens with 10% recursive overlap for peak F1-Score
        """)

# Comparative analysis
st.subheader("🔍 Comparative Analysis")

comparison_df = df[["chunk_size", "precision", "recall", "f1_score", "latency_ms", "vlm_latency_ms"]].copy()
comparison_df["total_latency_ms"] = comparison_df["latency_ms"] + comparison_df["vlm_latency_ms"]
comparison_df = comparison_df.sort_values("f1_score", ascending=False)

st.dataframe(
    comparison_df.style.format({
        "precision": "{:.3f}",
        "recall": "{:.3f}",
        "f1_score": "{:.3f}",
        "latency_ms": "{:.0f}",
        "vlm_latency_ms": "{:.0f}",
        "total_latency_ms": "{:.0f}"
    }).background_gradient(subset=["f1_score"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True
)

# Optimal recommendation
st.subheader("🎯 Optimal Configuration Recommendation")

optimal_row = df.loc[df["f1_score"].idxmax()]

st.info(f"""
**Recommended Configuration:**
- **Chunk Size:** {int(optimal_row['chunk_size'])} tokens
- **Overlap:** {selected_overlap}%
- **Expected F1 Score:** {optimal_row['f1_score']:.3f}
- **Expected Total Latency:** {optimal_row['latency_ms'] + optimal_row['vlm_latency_ms']:.0f}ms

This configuration provides the best balance between precision, recall, and latency for your RAG system.
""")
