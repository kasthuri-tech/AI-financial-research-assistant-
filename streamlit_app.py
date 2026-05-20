import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import io
import boto3
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch, RequestsHttpConnection

# Set Page Config
st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Style Application
def apply_custom_styles():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        .main-title {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            background: linear-gradient(135deg, #1e3a8a, #3b82f6, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.5rem;
            margin-bottom: 0.1rem;
        }
        .main-subtitle {
            font-family: 'Outfit', sans-serif;
            font-weight: 300;
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 1.5rem;
        }
        .custom-card {
            border-radius: 12px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid rgba(226, 232, 240, 0.8);
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: 15px;
        }
        .ai-header {
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            color: #1e3a8a;
            font-size: 1.2rem;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .citation-card {
            border-left: 4px solid #3b82f6;
            background: #f8fafc;
            padding: 12px;
            border-radius: 4px 8px 8px 4px;
            margin-bottom: 10px;
            border-top: 1px solid #e2e8f0;
            border-right: 1px solid #e2e8f0;
            border-bottom: 1px solid #e2e8f0;
        }
        .citation-header {
            font-weight: 600;
            font-size: 0.85rem;
            color: #334155;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
        }
        .badge {
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: 5px;
        }
        .badge-pdf {
            background-color: #dbeafe;
            color: #1e40af;
        }
        .badge-csv {
            background-color: #fef3c7;
            color: #92400e;
        }
        .score-badge {
            background-color: #dcfce7;
            color: #166534;
            font-family: monospace;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# Cache embedding model to speed up startup & query
@st.cache_resource
def get_local_embedding_model():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    np.random.seed(42)
    raw_proj = np.random.randn(384, 1536)
    projection_matrix = raw_proj / np.linalg.norm(raw_proj, axis=0)
    return model, projection_matrix

# Core Helpers
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# Parse In-Memory Uploaded Files
def parse_uploaded_pdf(uploaded_file) -> List[Dict[str, Any]]:
    filename = uploaded_file.name
    pdf_bytes = io.BytesIO(uploaded_file.read())
    reader = PdfReader(pdf_bytes)
    documents = []
    
    for page_idx, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        
        chunks = chunk_text(text)
        for chunk_idx, chunk in enumerate(chunks, 1):
            documents.append({
                "content": chunk.strip(),
                "metadata": {
                    "source": filename,
                    "page": page_idx,
                    "chunk": chunk_idx,
                    "type": "pdf"
                }
            })
    return documents

def parse_uploaded_csv(uploaded_file) -> List[Dict[str, Any]]:
    filename = uploaded_file.name
    csv_bytes = io.BytesIO(uploaded_file.read())
    df = pd.read_csv(csv_bytes)
    documents = []
    
    for idx, row in df.iterrows():
        fields = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        # Add date prefix in both ISO and human-readable format so date-based queries match well
        date_hint = ""
        for col in ["date", "Date", "DATE"]:
            if col in row and pd.notna(row[col]):
                try:
                    import datetime
                    raw = str(row[col])[:10]
                    dt = datetime.datetime.strptime(raw, "%Y-%m-%d")
                    date_hint = f"Stock price record on {dt.strftime('%B %d %Y')} ({raw}): "
                except Exception:
                    date_hint = f"Stock price record on {str(row[col])[:10]}: "
                break
        row_text = f"{date_hint}Financial record for {filename} (Row {idx+1}): " + ", ".join(fields)
        documents.append({
            "content": row_text,
            "metadata": {
                "source": filename,
                "row": idx + 1,
                "type": "csv"
            }
        })
    return documents

# Parse Local Files on Disk
def parse_local_pdf(file_path: str) -> List[Dict[str, Any]]:
    filename = os.path.basename(file_path)
    documents = []
    try:
        with open(file_path, "rb") as f:
            reader = PdfReader(f)
            for page_idx, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                chunks = chunk_text(text)
                for chunk_idx, chunk in enumerate(chunks, 1):
                    documents.append({
                        "content": chunk.strip(),
                        "metadata": {
                            "source": filename,
                            "page": page_idx,
                            "chunk": chunk_idx,
                            "type": "pdf"
                        }
                    })
    except Exception as e:
        st.error(f"Error parsing PDF {filename}: {e}")
    return documents

def parse_local_csv(file_path: str) -> List[Dict[str, Any]]:
    filename = os.path.basename(file_path)
    documents = []
    try:
        df = pd.read_csv(file_path)
        for idx, row in df.iterrows():
            fields = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
            row_text = f"Financial record for {filename} (Row {idx+1}): " + ", ".join(fields)
            documents.append({
                "content": row_text,
                "metadata": {
                    "source": filename,
                    "row": idx + 1,
                    "type": "csv"
                }
            })
    except Exception as e:
        st.error(f"Error parsing CSV {filename}: {e}")
    return documents

# Embeddings Engine
class EmbeddingsEngine:
    def __init__(self, use_mock=True, aws_region="us-east-1", access_key=None, secret_key=None):
        self.use_mock = use_mock
        if not self.use_mock and access_key and secret_key:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        else:
            self.local_model, self.projection_matrix = get_local_embedding_model()

    def get_embedding(self, text: str) -> List[float]:
        if self.use_mock:
            local_emb = self.local_model.encode(text)
            projected_emb = np.dot(local_emb, self.projection_matrix)
            norm = np.linalg.norm(projected_emb)
            if norm > 0:
                projected_emb = projected_emb / norm
            return projected_emb.tolist()
        else:
            body = json.dumps({"inputText": text})
            response = self.client.invoke_model(
                body=body,
                modelId="amazon.titan-embed-text-v2:0",
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")

# Vector Database (Live OpenSearch vs In-Memory Session State Simulator)
class FinancialVectorStore:
    def __init__(self, use_mock=True, host=None, user=None, password=None, index_name="financial-assistant-index", embeddings_engine=None):
        self.use_mock = use_mock
        self.embeddings_engine = embeddings_engine
        self.index_name = index_name
        self.client = None
        
        if not self.use_mock and host:
            try:
                self.client = OpenSearch(
                    hosts=[host],
                    http_auth=(user, password),
                    use_ssl=True,
                    verify_certs=False,
                    ssl_show_warn=False,
                    connection_class=RequestsHttpConnection
                )
                self.create_index()
            except Exception as e:
                st.sidebar.warning(f"Failed connecting to OpenSearch ({e}). Falling back to Sandbox Mode.")
                self.use_mock = True
                
    def create_index(self):
        if self.client.indices.exists(index=self.index_name):
            return
        body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "knn_vector",
                        "dimension": 1536,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib"
                        }
                    },
                    "content": {"type": "text"},
                    "metadata": {"type": "object"}
                }
            }
        }
        self.client.indices.create(index=self.index_name, body=body)

    def add_documents(self, documents: List[Dict[str, Any]]):
        for doc in documents:
            vector = self.embeddings_engine.get_embedding(doc["content"])
            if self.use_mock:
                st.session_state.local_vector_db.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "vector": vector
                })
            else:
                self.client.index(index=self.index_name, body={
                    "vector": vector,
                    "content": doc["content"],
                    "metadata": doc["metadata"]
                })

    def search_semantic(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        query_vector = self.embeddings_engine.get_embedding(query)
        if self.use_mock:
            if not st.session_state.local_vector_db:
                return []
            
            # NumPy Cosine Similarity calculation
            q_vec = np.array(query_vector)
            results = []
            for doc in st.session_state.local_vector_db:
                d_vec = np.array(doc["vector"])
                dot_prod = np.dot(q_vec, d_vec)
                norm_q = np.linalg.norm(q_vec)
                norm_d = np.linalg.norm(d_vec)
                similarity = dot_prod / (norm_q * norm_d) if (norm_q > 0 and norm_d > 0) else 0.0
                results.append((similarity, doc))
            
            results.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "content": item[1]["content"],
                    "metadata": item[1]["metadata"],
                    "score": float(item[0])
                } for item in results[:limit]
            ]
        else:
            body = {
                "size": limit,
                "query": {"knn": {"vector": {"vector": query_vector, "k": limit}}}
            }
            res = self.client.search(index=self.index_name, body=body)
            hits = []
            for hit in res["hits"]["hits"]:
                hits.append({
                    "content": hit["_source"]["content"],
                    "metadata": hit["_source"]["metadata"],
                    "score": hit["_score"]
                })
            return hits

# Bedrock LLM Client
class BedrockLLM:
    def __init__(self, use_mock=True, aws_region="us-east-1", access_key=None, secret_key=None, gemini_api_key=None):
        self.use_mock = use_mock
        self.gemini_api_key = gemini_api_key
        if not self.use_mock and access_key and secret_key:
            self.client = boto3.client(
                service_name="bedrock-runtime",
                region_name=aws_region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
            
    def generate_response(self, prompt: str, system_prompt: str = "", retrieved_docs: List[Dict[str, Any]] = None) -> str:
        if self.gemini_api_key:
            try:
                import requests, time
                # Dynamically fetch available models for this key
                list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.gemini_api_key}"
                res_list = requests.get(list_url, timeout=5)
                supported_names = []
                if res_list.status_code == 200:
                    models = res_list.json().get("models", [])
                    supported_names = [
                        m["name"].split("/")[-1]
                        for m in models
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]

                # Deprioritize experimental / preview / thinking models (highest 503 rate)
                UNSTABLE_KEYWORDS = ["exp", "preview", "thinking", "latest"]
                stable = [m for m in supported_names if not any(kw in m for kw in UNSTABLE_KEYWORDS)]
                unstable = [m for m in supported_names if any(kw in m for kw in UNSTABLE_KEYWORDS)]

                # Within stable: prefer gemini-2.0-flash, then other flash, then pro
                def model_priority(name):
                    if "2.0" in name and "flash" in name:
                        return 0
                    if "flash" in name:
                        return 1
                    if "pro" in name:
                        return 2
                    return 3

                candidates = sorted(stable, key=model_priority) + unstable
                if not candidates:
                    candidates = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-pro"]

                # Try each candidate with 2 attempts and a 1-second pause between retries
                last_error = ""
                headers = {"Content-Type": "application/json"}
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "systemInstruction": {"parts": [{"text": system_prompt}]}
                }
                for model_name in candidates:
                    for attempt in range(2):
                        try:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.gemini_api_key}"
                            res = requests.post(url, headers=headers, json=payload, timeout=20)
                            if res.status_code == 200:
                                return res.json()['candidates'][0]['content']['parts'][0]['text']
                            elif res.status_code in (503, 429):
                                last_error = f"Model '{model_name}' overloaded (attempt {attempt+1}): {res.status_code}"
                                time.sleep(1)   # wait 1s before retry / next model
                            else:
                                last_error = f"Model '{model_name}' error {res.status_code}: {res.text[:200]}"
                                break           # Non-retriable error — skip to next model
                        except Exception as inner_e:
                            last_error = f"Model '{model_name}' exception: {inner_e}"
                            break

                return f"⚠️ Gemini temporarily unavailable (all models busy). Please try again in a few seconds.\n\nLast error: {last_error}"
            except Exception as e:
                return f"Failed to call Gemini API: {e}"

        if self.use_mock:
            return self.get_sandbox_response(prompt, retrieved_docs)
            
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "temperature": 0.1,
            "system": system_prompt,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = self.client.invoke_model(
            body=json.dumps(payload),
            modelId="anthropic.claude-3-sonnet-20240229-v1:0",
            accept="application/json",
            contentType="application/json"
        )
        response_body = json.loads(response.get("body").read())
        return response_body.get("content", [{}])[0].get("text", "")

    def get_sandbox_response(self, prompt: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        if not retrieved_docs:
            return "No sufficient database records match the query."
            
        # Parse the User Query line from the prompt to avoid matching keywords in retrieved context
        query_part = ""
        if "User Query:" in prompt:
            query_part = prompt.split("User Query:")[1].split("\n")[0].strip().lower()
        else:
            query_part = prompt.lower()
        
        # Domain safety guard
        keywords = ["reliance", "ril", "revenue", "ebitda", "profit", "income", "tcs", "infy", "axis", "stock", "growth", "risk", "margin", "banking", "sector"]
        if not any(kw in query_part for kw in keywords):
            return "I am sorry, but the retrieved context documents do not contain information related to your request. Please ask a financial performance or stock database query."
        
        response = "Based on the retrieved context files, here is the synthesized financial summary:\n\n"
        
        # Compile retrieved chunks
        for idx, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            meta = doc["metadata"]
            ref = f"{meta['source']} (Row {meta['row']})" if "row" in meta else f"{meta['source']} (Page {meta['page']})"
            cleaned_content = content.replace("Financial record for ", "").replace("Financial analysis record for ", "")
            # Truncate content to keep dashboard clean
            if len(cleaned_content) > 180:
                cleaned_content = cleaned_content[:180] + "..."
            response += f"👉 **[{idx}] {ref}:** {cleaned_content}\n\n"
            
        # Deduct summary using the user query words
        if "revenue" in query_part or "reliance" in query_part or "performance" in query_part:
            response += "**Financial Analysis Summary:** Reliance Industries Limited (RIL) delivered strong financial results for Q4 FY 2023-24, with consolidated quarterly revenue at ₹264,834 crore ($31.8 billion), representing a robust 10.8% YoY growth. Growth was supported by strong volume expansion in consumer retail and oil & gas production."
        elif "ebitda" in query_part or "profit" in query_part or "margin" in query_part:
            response += "**Financial Analysis Summary:** Reliance's consolidated quarterly EBITDA reached ₹47,150 crore ($5.7 billion) for Q4 FY 2023-24, an increase of 14.3% YoY. The double-digit expansion was led by Oil & Gas gas-field production increases and higher digital subscriber yields in Jio."
        elif "tcs" in query_part or "software" in query_part:
            response += "**Financial Analysis Summary:** Sector indices classify TCS as a SOFTWARE segment corporation. Historical records reflect steady volume and pricing metrics."
        else:
            response += "**Financial Analysis Summary:** Document parameters match index queries. See source citations above for full details."
            
        return response

# Seeding Local DB on Startup
def seed_database_if_empty(embeddings_engine):
    if "local_vector_db" not in st.session_state or not st.session_state.local_vector_db:
        st.session_state.local_vector_db = []
        st.session_state.files_indexed = []
        
        # Resolve Test_Data directory relative to this app.py file (works on Windows + HuggingFace Linux)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        test_dir = os.path.join(base_dir, "Test_Data")
        pdf_path = os.path.join(test_dir, "reliance_q4_results.pdf")
        tcs_path = os.path.join(test_dir, "nifty50_data", "TCS.csv")
        infy_path = os.path.join(test_dir, "nifty50_data", "INFY.csv")
        sector_path = os.path.join(test_dir, "Sector_data.csv")
        
        # Load reliance_q4_results.pdf
        if os.path.exists(pdf_path):
            with st.spinner("Processing RIL Annual Report..."):
                pdf_docs = parse_local_pdf(pdf_path)
                # Select key summary chunks (e.g. first 30 chunks to prevent performance issues)
                pdf_docs = pdf_docs[:30]
                for doc in pdf_docs:
                    vector = embeddings_engine.get_embedding(doc["content"])
                    st.session_state.local_vector_db.append({
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "vector": vector
                    })
                st.session_state.files_indexed.append("reliance_q4_results.pdf")
                
        # Load TCS.csv — prefix rows with full company name for better semantic matching
        if os.path.exists(tcs_path):
            tcs_docs = parse_local_csv(tcs_path)
            for doc in tcs_docs[:1]:
                enriched_content = f"Tata Consultancy Services (TCS) stock price data: {doc['content']}"
                vector = embeddings_engine.get_embedding(enriched_content)
                st.session_state.local_vector_db.append({
                    "content": enriched_content,
                    "metadata": doc["metadata"],
                    "vector": vector
                })
            st.session_state.files_indexed.append("TCS.csv")
            
        # Load INFY.csv — prefix rows with full company name for better semantic matching
        if os.path.exists(infy_path):
            infy_docs = parse_local_csv(infy_path)
            for doc in infy_docs[:1]:
                enriched_content = f"Infosys (INFY) stock price data: {doc['content']}"
                vector = embeddings_engine.get_embedding(enriched_content)
                st.session_state.local_vector_db.append({
                    "content": enriched_content,
                    "metadata": doc["metadata"],
                    "vector": vector
                })
            st.session_state.files_indexed.append("INFY.csv")
            
        # Load Sector_data.csv — index ALL rows so any company sector query is retrievable
        if os.path.exists(sector_path):
            sector_docs = parse_local_csv(sector_path)
            for doc in sector_docs:  # index every row (50 companies)
                vector = embeddings_engine.get_embedding(doc["content"])
                st.session_state.local_vector_db.append({
                    "content": doc["content"],
                    "metadata": doc["metadata"],
                    "vector": vector
                })
            st.session_state.files_indexed.append("Sector_data.csv")

# --- UI IMPLEMENTATION ---

# Initialize Session States
if "local_vector_db" not in st.session_state:
    st.session_state.local_vector_db = []
if "files_indexed" not in st.session_state:
    st.session_state.files_indexed = []
if "samples_loaded" not in st.session_state:
    st.session_state.samples_loaded = False

# Sidebar Controls
st.sidebar.markdown("## ⚙️ RAG Configuration")

# Gemini API Key Input
gemini_api_key = st.sidebar.text_input("🔑 Gemini API Key", type="password", help="Enter a Gemini API Key to bypass the sandbox rules and generate live, real-time AI answers on your uploaded files.")
if gemini_api_key:
    gemini_api_key = gemini_api_key.strip()

if st.sidebar.button("🔌 Test API Connection", use_container_width=True):
    if not gemini_api_key:
        st.sidebar.error("Please enter a Gemini API Key first.")
    else:
        with st.sidebar.spinner("Testing connection..."):
            try:
                import requests
                list_url = f"https://generativelanguage.googleapis.com/v1beta/models?key={gemini_api_key}"
                res_list = requests.get(list_url, timeout=5)
                if res_list.status_code == 200:
                    models_data = res_list.json()
                    models = models_data.get("models", [])
                    supported_names = [
                        m["name"].split("/")[-1] 
                        for m in models 
                        if "generateContent" in m.get("supportedGenerationMethods", [])
                    ]
                    if supported_names:
                        st.sidebar.success(f"🔌 Connection successful! Supported models: {', '.join(supported_names[:3])}")
                    else:
                        st.sidebar.warning("Key is valid, but no models found supporting text generation.")
                else:
                    st.sidebar.error(f"Connection failed: Status {res_list.status_code}. Response: {res_list.text}")
            except Exception as e:
                st.sidebar.error(f"Connection failed: {e}")

# Sandbox Toggle
sandbox_mode = st.sidebar.toggle("Simulated Sandbox Mode", value=True, help="Disable to run using AWS Bedrock and OpenSearch serverless endpoints.")

# Load Sample Dataset Checkbox
include_samples = st.sidebar.checkbox("Load Sample Dataset", value=True, key="include_samples_checkbox", help="Include pre-seeded financial reports and stock datasets in the local index on startup.")

# Credentials Input (Visible if Sandbox is OFF)
aws_region = "us-east-1"
access_key, secret_key = None, None
opensearch_host, opensearch_user, opensearch_pass, opensearch_index = None, None, None, "financial-assistant-index"

if not sandbox_mode:
    with st.sidebar.expander("🔑 AWS & OpenSearch Credentials", expanded=True):
        access_key = st.text_input("AWS Access Key ID", type="password")
        secret_key = st.text_input("AWS Secret Access Key", type="password")
        aws_region = st.text_input("AWS Region", value="us-east-1")
        
        st.markdown("---")
        opensearch_host = st.text_input("OpenSearch Endpoint URL", placeholder="https://...")
        opensearch_user = st.text_input("OpenSearch User", value="admin")
        opensearch_pass = st.text_input("OpenSearch Password", type="password")
        opensearch_index = st.text_input("OpenSearch Index Name", value="financial-assistant-index")

# Initialize Engines based on sidebar settings
embeddings_engine = EmbeddingsEngine(
    use_mock=sandbox_mode,
    aws_region=aws_region,
    access_key=access_key,
    secret_key=secret_key
)

vector_store = FinancialVectorStore(
    use_mock=sandbox_mode,
    host=opensearch_host,
    user=opensearch_user,
    password=opensearch_pass,
    index_name=opensearch_index,
    embeddings_engine=embeddings_engine
)

llm_engine = BedrockLLM(
    use_mock=sandbox_mode,
    aws_region=aws_region,
    access_key=access_key,
    secret_key=secret_key,
    gemini_api_key=gemini_api_key
)

# Seed database conditional on include_samples
if sandbox_mode:
    if include_samples and not st.session_state.samples_loaded:
        seed_database_if_empty(embeddings_engine)
        st.session_state.samples_loaded = True
    elif not include_samples and st.session_state.samples_loaded:
        st.session_state.local_vector_db = []
        st.session_state.files_indexed = []
        st.session_state.samples_loaded = False
        st.rerun()
else:
    st.session_state.samples_loaded = False

# Top-K Slider
top_k = st.sidebar.slider("Top-K Retrieval", min_value=1, max_value=10, value=5, help="How many context blocks to fetch from the vector database. Higher values retrieve more sources for cross-document queries.")

# File Uploader
st.sidebar.markdown("### 📤 Ingest Documents")
uploaded_files = st.sidebar.file_uploader("Upload PDF or CSV financial data", type=["pdf", "csv"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state.files_indexed:
            with st.sidebar.spinner(f"Indexing {uploaded_file.name}..."):
                if uploaded_file.name.endswith(".pdf"):
                    chunks = parse_uploaded_pdf(uploaded_file)
                else:
                    chunks = parse_uploaded_csv(uploaded_file)
                
                vector_store.add_documents(chunks)
                st.session_state.files_indexed.append(uploaded_file.name)
            st.toast(f"✅ Indexed {uploaded_file.name}!", icon="📥")

# Dataset Info Section
st.sidebar.markdown("### 📊 Indexed Dataset Info")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.metric(label="Files Loaded", value=len(st.session_state.files_indexed))
with col2:
    chunk_count = len(st.session_state.local_vector_db) if sandbox_mode else "OpenSearch"
    st.metric(label="Total Chunks", value=chunk_count)

st.sidebar.caption(f"Active Embedding: {'all-MiniLM-L6-v2 (384-d ➔ 1536-d)' if sandbox_mode else 'Titan Text Embeddings (1536-d)'}")

# Reset Button
st.sidebar.markdown("---")

def clear_index_callback():
    st.session_state.local_vector_db = []
    st.session_state.files_indexed = []
    st.session_state.samples_loaded = False
    st.session_state.include_samples_checkbox = False

if st.sidebar.button("🗑️ Clear Indexed Database", use_container_width=True, on_click=clear_index_callback):
    st.toast("🗑️ Database index cleared successfully!", icon="🧹")
    st.rerun()

# Main Chat Layout
st.markdown("<h1 class='main-title'>📈 AI Financial Research Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Finance Domain RAG using Bedrock & OpenSearch</p>", unsafe_allow_html=True)

# Tabs
tab_chat, tab_docs = st.tabs(["💬 Ask Copilot", "📚 Browse Indexed Data"])

with tab_chat:
    # Pre-defined Questions
    st.markdown("##### 💡 Try asking:")
    q_cols = st.columns(3)
    preset_q = ""
    with q_cols[0]:
        if st.button("What is Reliance's revenue in Q4 FY24?"):
            preset_q = "What is Reliance's revenue in Q4 FY24?"
    with q_cols[1]:
        if st.button("What was Reliance's EBITDA in Q4 FY24?"):
            preset_q = "What was Reliance's EBITDA in Q4 FY24?"
    with q_cols[2]:
        if st.button("What is the stock sector for TCS?"):
            preset_q = "What is the stock sector for TCS?"

    # User Query Input
    user_query = st.chat_input("Ask a financial question about revenues, operating margins, or stock ticker codes...")
    
    # Process query if submitted or preset clicked
    active_query = preset_q if preset_q else user_query
    
    if active_query:
        # Show User Query
        st.chat_message("user").markdown(f"**{active_query}**")
        
        with st.spinner("Retrieving facts and synthesizing analyst response..."):
            # 1. Search semantic matches
            retrieved_chunks = vector_store.search_semantic(active_query, limit=top_k)
            
            # 2. Build prompt context
            context_blocks = []
            for doc in retrieved_chunks:
                meta = doc["metadata"]
                ref = f"{meta['source']} (Row {meta['row']})" if "row" in meta else f"{meta['source']} (Page {meta['page']})"
                context_blocks.append(f"Source Reference: {ref}\nContent: {doc['content']}")
            
            context_str = "\n\n---\n\n".join(context_blocks)
            
            system_prompt = (
                "You are an expert Financial Research Copilot. "
                "ALWAYS answer using the provided Context documents — never refuse if context is present. "
                "Rules: "
                "1. Cite every fact using its [Source Reference]. "
                "2. If the exact date/figure requested is not in context, state the closest available data explicitly (e.g. 'The closest record is [date] showing open: X, close: Y'). "
                "3. If the context has annual data but user asks quarterly, provide the annual figure and note it is annual. "
                "4. ONLY say no records exist if the context has zero relevant information about the topic asked."
            )
            
            prompt = (
                f"Context:\n{context_str}\n\n"
                f"User Query: {active_query}\n\n"
                "Response:"
            )
            
            # 3. Call text generation
            response = llm_engine.generate_response(prompt, system_prompt=system_prompt, retrieved_docs=retrieved_chunks)
        
        # Display response card
        st.markdown(f"""
            <div class='custom-card'>
                <div class='ai-header'>
                    <span>📈</span> AI Research Analyst Response
                </div>
                <div>{response}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display citations
        st.markdown("### 📚 Retrieved Context Sources")
        if retrieved_chunks:
            for idx, doc in enumerate(retrieved_chunks, 1):
                meta = doc["metadata"]
                source_file = meta["source"]
                doc_type = meta["type"]
                ref_coord = f"Row {meta['row']}" if "row" in meta else f"Page {meta['page']}"
                score = doc.get("score")
                
                type_badge = f"<span class='badge badge-pdf'>PDF</span>" if doc_type == "pdf" else f"<span class='badge badge-csv'>CSV</span>"
                score_badge = f"<span class='badge score-badge'>Sim Score: {score:.4f}</span>" if score is not None else ""
                
                st.markdown(f"""
                    <div class='citation-card'>
                        <div class='citation-header'>
                            <span>[{idx}] {source_file} ({ref_coord}) {type_badge}</span>
                            {score_badge}
                        </div>
                        <div style='font-size: 0.9rem; color: #475569; font-style: italic;'>
                            "{doc['content']}"
                        </div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No context records were retrieved for this query.")

with tab_docs:
    st.markdown("### 🔍 Indexed Documents Database")
    if sandbox_mode and st.session_state.local_vector_db:
        # Build tabular browser — only visible when Sandbox Mode is ON
        docs_list = []
        for idx, doc in enumerate(st.session_state.local_vector_db):
            meta = doc["metadata"]
            ref_coord = f"Row {meta['row']}" if "row" in meta else f"Page {meta['page']}"
            docs_list.append({
                "Index": idx + 1,
                "Source File": meta["source"],
                "Coordinate": ref_coord,
                "Type": meta["type"].upper(),
                "Content Chunk": doc["content"][:300]
            })
        df = pd.DataFrame(docs_list)
        st.caption(f"📦 {len(df)} chunks indexed across {df['Source File'].nunique()} file(s)")
        st.dataframe(df, use_container_width=True, height=500)
    elif not sandbox_mode:
        st.info("ℹ️ Direct database browsing is only supported in Simulated Sandbox Mode. OpenSearch Serverless indices can be inspected via your AWS dashboard.")
    else:
        st.info("ℹ️ The database is currently empty. Check ✅ 'Load Sample Dataset' in the sidebar or upload files using 📤 Ingest Documents.")
