# 📈 AI Financial Research Assistant

A beginner-friendly, Python-based Retrieval-Augmented Generation (RAG) project designed to run directly inside Google Colab and as a local web application. 

This project showcases how to connect **Amazon Bedrock** (LLM and Embeddings) with **Amazon OpenSearch** (Vector Database) to build a semantic search assistant for analyzing financial datasets (PDFs and CSV spreadsheets).

## 🚀 Key Learning Objectives
* **Amazon Bedrock**: Utilizing Foundation Models for generating text (Anthropic Claude 3) and embedding vector generation (Titan Embeddings).
* **RAG Architecture**: Retrieving real-time stock context to answer user queries without LLM hallucination.
* **OpenSearch Integration**: Indexing, storing, and performing k-NN semantic search on vector representations of text.
* **Document Processing**: Splitting PDF reports and stock CSV rows into text chunks with metadata citations.
* **Math & Simulators**: Custom mathematical offline simulations (NumPy Cosine Similarity & orthogonal dimension projection) for credentials-free sandbox runs.
* **Interactive Dashboard**: A fully functional Streamlit web application with drag-and-drop document upload and citation analytics.

## 📂 Project Structure
* `app.py` - Streamlit web application implementing the interactive sidebar control, real-time file parsing, and citation layout.
* `financial_research_assistant.ipynb` - The main Google Colab notebook containing all project code, cells, and visual explanations.
* `Test_Data/` - Contains sample financial datasets:
  * `Sector_data.csv` - IT services market trends.
  * `Nifty50_Master_for_PowerBI.csv` - Comprehensive stock dataset.
  * `nifty50_data/` - Stock CSV files (TCS, INFY, AXISBANK).

## 🛡️ Offline Sandbox Mode (Zero-AWS Setup)
If you don't have AWS credentials or an active OpenSearch instance, both the Colab notebook and the Streamlit web app automatically fall back to **Simulated Sandbox Mode**:
1. **Local Embeddings**: Generates real vector embeddings using the open-source `sentence-transformers/all-MiniLM-L6-v2` model inside the environment.
2. **Dimension Projection**: Projects the 384-dimensional local vector into a 1536-dimensional space (to match Amazon Titan's dimension) using a randomized orthogonal projection matrix (inspired by the *Johnson-Lindenstrauss lemma*).
3. **NumPy Vector Database**: Computes cosine similarity scores on vectors in-memory using `numpy` to replicate OpenSearch's vector search engine.
4. **Context Synthesis**: Extracts text matches and programmatically compiles citations to formulate structured analyst responses without calling external LLM APIs.

## 💻 Running the Streamlit Web App
To run the interactive web application:
1. Make sure you have python and streamlit installed:
   ```bash
   pip install streamlit boto3 opensearch-py sentence-transformers pypdf reportlab pandas numpy
   ```
2. Run the application:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` (or `http://localhost:8502`) in your browser.

## 💻 Running the Colab Notebook
1. Upload `financial_research_assistant.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Connect Colab runtime.
3. Configure your AWS credentials in **Step 2** to run in live mode. **Leave them blank to run in offline Sandbox Mode immediately.**
4. Run all cells sequentially. The notebook will automatically generate a sample financial PDF and load CSV records for search.


