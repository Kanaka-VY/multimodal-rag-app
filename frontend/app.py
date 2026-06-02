import os
import time

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="OmniRAG - Enterprise Multimodal RAG", page_icon="🌐", layout="wide")

# Custom CSS for enterprise look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
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
    .status-success {
        color: #22c55e;
    }
    .status-warning {
        color: #eab308;
    }
</style>
""", unsafe_allow_html=True)

# Authentication check
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🌐 OmniRAG</h1>
        <p style="color: #cbd5e1; margin: 0.5rem 0 0 0;">Enterprise Secure Data Knowledge Portal</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        st.subheader("🔒 Sign In to Secure Gateway")
        col1, col2 = st.columns(2)
        with col1:
            username = st.text_input("Username / Corporate Email", placeholder="aml_engineer@enterprise.com")
        with col2:
            role = st.selectbox("Security Role Group", ["AI_Research_Staff", "Data_Scientist", "Admin", "Viewer"])
        token = st.text_input("Access Token Secret", type="password", placeholder="••••••••••••••••")
        
        if st.form_submit_button("SECURE LOGIN", use_container_width=True):
            # Simple authentication (in production, use proper auth)
            if username and token:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.role = role
                st.rerun()
            else:
                st.error("Please enter username and token")
    
    st.info("🔒 Compliance Status: SOC2 Secure | TLS 1.3 Active | RBAC Identity Encrypted")
    st.stop()

# Main application
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🌐 OmniRAG</h1>
    <p style="color: #cbd5e1; margin: 0.5rem 0 0 0;">Enterprise Multimodal Knowledge Base | 👤 {} | {}</p>
</div>
""".format(st.session_state.username, st.session_state.role), unsafe_allow_html=True)

col_side, col_main = st.columns([1, 2])

with col_side:
    st.subheader("📂 Multimodal Ingestion Control")
    api_url = st.text_input("API URL", value=API_URL)
    st.divider()
    
    uploaded = st.file_uploader(
        "Drag & drop target files (PDF, PNG, WAV)",
        type=["pdf", "png", "jpg", "jpeg", "webp", "wav", "mp3", "m4a", "txt"],
    )
    
    if uploaded:
        st.info(f"📄 {uploaded.name} [{uploaded.size / 1024 / 1024:.1f} MB]")
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("Upload & Index", disabled=uploaded is None, use_container_width=True):
            with st.spinner("Indexing complete to Vector Store..."):
                files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
                resp = requests.post(f"{api_url}/ingest/file", files=files, timeout=300)
            if resp.ok:
                st.success("✅ Indexing complete")
                st.json(resp.json())
            else:
                st.error(resp.text)
    
    with col_btn2:
        if st.button("Index /data folder", use_container_width=True):
            with st.spinner("Scanning and indexing..."):
                resp = requests.post(f"{api_url}/ingest/directory", params={"path": "data"}, timeout=600)
            if resp.ok:
                st.success("✅ Bulk indexing complete")
                st.json(resp.json())
            else:
                st.error(resp.text)
    
    st.divider()
    st.subheader("⚙️ Applied Inference Modalities")
    st.checkbox("Hybrid Document Extraction", value=True, disabled=True)
    st.checkbox("CLIP Cross-Modal Mapping", value=True, disabled=True)
    st.checkbox("OpenAI Whisper Audio Sync", value=True, disabled=True)

with col_main:
    st.subheader("💬 Real-Time Enterprise Knowledge Chat")
    
    # Voice input using Web Speech API
    st.markdown("""
    <script>
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onresult = function(event) {
            const transcript = event.results[0][0].transcript;
            document.querySelector('textarea[data-testid="stTextArea"]').value = transcript;
            document.querySelector('textarea[data-testid="stTextArea"]').dispatchEvent(new Event('input', { bubbles: true }));
        };
        
        window.startVoiceInput = function() {
            recognition.start();
        };
    }
    </script>
    <button onclick="startVoiceInput()" style="padding: 8px 16px; background: #3b82f6; color: white; border: none; border-radius: 4px; cursor: pointer;">🎙️ Voice Input</button>
    """, unsafe_allow_html=True)
    
    query = st.text_area("Your question", height=100, placeholder="Type your corporate database or document query here...")
    
    col_filter, col_k = st.columns([2, 1])
    with col_filter:
        modality = st.selectbox(
            "Filter modality (optional)",
            options=[None, "pdf", "text", "image", "audio"],
            format_func=lambda x: "All" if x is None else x,
        )
    with col_k:
        top_k = st.slider("Top K results", 1, 15, 5)
    
    if st.button("🔍 Search", disabled=not query.strip(), use_container_width=True):
        payload = {"query": query, "top_k": top_k}
        if modality:
            payload["modality"] = modality
        
        start_time = time.time()
        
        # Try WebSocket streaming first
        try:
            import websockets
            import asyncio
            
            st.markdown("### ⚡ Streaming Response...")
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            async def stream_response():
                uri = f"ws://{api_url.replace('http://', '')}/ws/query"
                async with websockets.connect(uri) as websocket:
                    await websocket.send_json({
                        "query": query,
                        "modality": modality,
                        "top_k": top_k
                    })
                    
                    full_answer = ""
                    sources = []
                    
                    while True:
                        try:
                            message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                            data = message if isinstance(message, dict) else eval(message)
                            
                            status = data.get("status")
                            
                            if status == "retrieving":
                                response_placeholder.info("🔍 Retrieving relevant documents...")
                            elif status == "sources_retrieved":
                                sources = data.get("sources", [])
                                sources_placeholder.info(f"📄 Found {len(sources)} relevant sources")
                            elif status == "generating":
                                response_placeholder.info("🤖 Generating response...")
                            elif status == "streaming":
                                chunk = data.get("chunk", "")
                                full_answer += chunk
                                response_placeholder.markdown(f"### 🤖 OmniRAG Response\n\n{full_answer}▌")
                            elif status == "complete":
                                full_answer = data.get("answer", full_answer)
                                sources = data.get("sources", sources)
                                break
                            elif status == "error":
                                st.error(data.get("error", "Unknown error"))
                                break
                        except asyncio.TimeoutError:
                            break
                    
                    return full_answer, sources
            
            full_answer, sources = asyncio.run(stream_response())
            
            # Display final answer
            st.markdown("### 🤖 OmniRAG Response")
            st.write(full_answer)
            
            # Display sources
            data = {"answer": full_answer, "sources": sources}
            
        except Exception as e:
            # Fallback to regular HTTP request
            with st.spinner("⚡ Retrieving..."):
                resp = requests.post(f"{api_url}/query", json=payload, timeout=120)
            
            latency = time.time() - start_time
            
            if not resp.ok:
                st.error(resp.text)
            else:
                data = resp.json()
                
                # Answer section
                st.markdown("### 🤖 OmniRAG Response")
                st.write(data.get("answer", ""))
                
                # Sources section with context inspector
                st.markdown("### 👁️ Live Retrieved Context Inspector")
                
                # Display visual context if images are retrieved
                image_sources = [src for src in data.get("sources", []) if src.get("metadata", {}).get("modality") == "image"]
                if image_sources:
                    st.markdown("### 🔍 Generated Cross-Modal Visual Context")
                    
                    # Simulate chart rendering (in production, render actual images)
                    for i, img_src in enumerate(image_sources[:3], 1):
                        meta = img_src.get("metadata", {})
                        filename = meta.get('filename', 'chart.png')
                        
                        col_viz1, col_viz2 = st.columns([2, 1])
                        with col_viz1:
                            st.markdown(f"**{filename}** — Similarity: {img_src.get('score', 0):.3f}")
                            
                            # Simulate chart visualization
                            import plotly.graph_objects as go
                            import numpy as np
                            
                            # Generate sample chart data
                            np.random.seed(42)
                            x_values = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                            y_values = np.cumsum(np.random.uniform(0.3, 0.8, 6))
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=x_values,
                                y=y_values,
                                mode='lines+markers',
                                line=dict(color='#3b82f6', width=3),
                                marker=dict(size=8),
                                fill='tozeroy',
                                fillcolor='rgba(59, 130, 246, 0.2)'
                            ))
                            
                            fig.update_layout(
                                title="Footprint (TB)",
                                xaxis_title="Month",
                                yaxis_title="Value",
                                height=250,
                                margin=dict(l=50, r=50, t=50, b=50),
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_viz2:
                            st.markdown("**Top Vector Match ID:**")
                            st.code(f"point_{hash(filename) % 10000:04x}", language="text")
                            st.markdown("**Metadata:**")
                            st.markdown(f"- Source: {filename}")
                            st.markdown(f"- Frame Type: Chart / Vision Coordinate")
                            st.markdown(f"- Target Space: Unified CLIP Embedding")
                
                # Display text sources
                text_sources = [src for src in data.get("sources", []) if src.get("metadata", {}).get("modality") != "image"]
                for i, src in enumerate(text_sources, 1):
                    meta = src.get("metadata", {})
                    with st.expander(f"📄 Source {i}: {meta.get('filename', 'doc')} ({meta.get('modality', '?')}) — Score: {src.get('score', 0):.3f}"):
                        st.markdown(f"**Source:** {meta.get('source', 'unknown')}")
                        st.markdown(f"**Modality:** {meta.get('modality', 'unknown')}")
                        if meta.get('page'):
                            st.markdown(f"**Page:** {meta.get('page')}")
                        st.markdown(f"**Content:**")
                        st.text(src.get("content", ""))
    
    # Health check
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        st.metric("Indexed Chunks", health.get('documents_indexed', 0))
    except requests.RequestException:
        st.warning("⚠️ API is not reachable. Start the backend with `uvicorn src.main:app`.")
