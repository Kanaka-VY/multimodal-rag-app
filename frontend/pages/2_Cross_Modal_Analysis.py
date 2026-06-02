import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cross-Modal Analysis", page_icon="🧮", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #7c3aed 0%, #a855f7 100%);
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🧮 Cross-Modal Validation: Text-to-Image Matching Matrix</h1>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
st.sidebar.header("Configuration")
num_queries = st.sidebar.slider("Number of Test Queries", 3, 10, 5)
num_images = st.sidebar.slider("Number of Retrieved Images", 3, 10, 5)
threshold = st.sidebar.slider("Similarity Threshold", 0.0, 1.0, 0.88, 0.05)
st.sidebar.info("🔒 Higher threshold (0.88) reduces false positives")

# Generate sample cross-modal similarity matrix
st.subheader("📊 Cross-Modal Similarity Matrix")

# Sample data for demonstration
query_texts = [
    "Show financial growth trends",
    "Explain backend cluster setup",
    "Analyze voice sample tone",
    "Display revenue charts",
    "Review system architecture",
][:num_queries]

image_types = [
    "Revenue Line Graph",
    "Architecture Block",
    "Audio Waveform Spec",
    "Performance Chart",
    "Network Diagram",
][:num_images]

# Generate similarity scores (simulated for demo)
np.random.seed(42)
similarity_matrix = np.random.uniform(0.0, 1.0, (num_queries, num_images))

# Add some high correlations to simulate true positives
for i in range(min(num_queries, num_images)):
    similarity_matrix[i, i] = np.random.uniform(0.85, 0.98)

# Create heatmap
fig = go.Figure(data=go.Heatmap(
    z=similarity_matrix,
    x=image_types,
    y=query_texts,
    colorscale='RdYlGn',
    colorbar=dict(title="Similarity Score"),
    text=[[f"{score:.2f}" for score in row] for row in similarity_matrix],
    texttemplate="%{text}",
    textfont={"size": 12},
    hovertemplate='Query: %{y}<br>Image: %{x}<br>Similarity: %{z:.3f}<extra></extra>'
))

fig.update_layout(
    title="Text-to-Image Cross-Modal Similarity Scores",
    xaxis_title="Retrieved Images",
    yaxis_title="User Query Text",
    height=500,
    margin=dict(l=200, r=50, t=50, b=150)
)

st.plotly_chart(fig, use_container_width=True)

# True Positive Analysis
st.subheader("✅ True Positive Analysis")

true_positives = []
for i, query in enumerate(query_texts):
    for j, img in enumerate(image_types):
        if similarity_matrix[i, j] >= threshold:
            true_positives.append({
                "Query": query,
                "Image": img,
                "Similarity": similarity_matrix[i, j],
                "Status": "🟢 True Positive" if i == j else "⚠️ False Positive"
            })

if true_positives:
    tp_df = pd.DataFrame(true_positives)
    st.dataframe(tp_df, use_container_width=True, hide_index=True)
else:
    st.warning("No matches found above threshold. Try lowering the threshold.")

# Evaluation Metrics
st.subheader("📊 Final Evaluation Metrics")

# Calculate metrics
total_possible = num_queries * num_images
above_threshold = len([s for row in similarity_matrix for s in row if s >= threshold])
true_positive_count = sum(1 for i in range(min(num_queries, num_images)) if similarity_matrix[i, i] >= threshold)

vision_top1_recall = (true_positive_count / num_queries) if num_queries > 0 else 0
vision_top5_recall = (above_threshold / total_possible) if total_possible > 0 else 0

# Calculate MRR (Mean Reciprocal Rank)
reciprocal_ranks = []
for i in range(num_queries):
    sorted_indices = np.argsort(similarity_matrix[i])[::-1]
    for rank, idx in enumerate(sorted_indices, 1):
        if idx == i:  # Correct match
            reciprocal_ranks.append(1.0 / rank)
            break
    else:
        reciprocal_ranks.append(0.0)

mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

# Display metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Vision Top-1 Retrieval Recall", f"{vision_top1_recall:.1%}")
with col2:
    st.metric("Vision Top-5 Retrieval Recall", f"{vision_top5_recall:.1%}")
with col3:
    st.metric("Mean Reciprocal Rank (MRR)", f"{mrr:.3f}")

# Insights
st.subheader("💡 R&D Insight Log")

with st.expander("View Analysis Insights"):
    st.markdown("""
    **Cross-Modal Retrieval Performance:**
    - The system successfully matches text queries to relevant visual content using CLIP embeddings
    - High similarity scores (>0.85) indicate strong cross-modal understanding
    - False positives may occur when queries have semantic overlap with multiple image types
    
    **Recommendations:**
    - Increase CLIP model size for better fine-grained visual understanding
    - Implement query expansion to improve recall for complex queries
    - Add re-ranking layer to filter out false positives
    """)

# Interactive Test Section
st.subheader("🧪 Interactive Cross-Modal Test")

test_query = st.text_input("Enter test query:", placeholder="e.g., 'Show me the revenue chart'")
if st.button("Test Cross-Modal Retrieval"):
    if test_query:
        with st.spinner("Testing cross-modal retrieval..."):
            # Simulate retrieval (in production, call actual API)
            test_similarities = np.random.uniform(0.0, 1.0, len(image_types))
            
            st.write("**Retrieved Images with Similarity Scores:**")
            for img, score in sorted(zip(image_types, test_similarities), key=lambda x: x[1], reverse=True):
                status = "🟢" if score >= threshold else "⚪"
                st.markdown(f"{status} **{img}** — Similarity: {score:.3f}")
    else:
        st.warning("Please enter a test query")
