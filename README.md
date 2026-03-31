# 📘 Advanced Multi-Agent Research System

An AI-powered research platform that simulates a structured academic workflow using **RAG (Retrieval-Augmented Generation)**, **FAISS vector memory**, and a **multi-agent architecture**.

---

## 🚀 Overview

This system goes beyond traditional chatbots by:

* Planning before searching
* Retrieving real-time web data
* Generating structured research reports
* Critiquing and refining outputs
* Enabling chatbot interaction with generated reports

👉 It works like a **“Chat with PDF” system**, but dynamically generates the document first.

---

## 🧠 Key Features

* 🔍 Real-time web search using SerpAPI
* 🧩 RAG-based architecture
* 🗂️ Vector database using FAISS
* 🤖 Multi-agent system (Planner, Critic, Improver)
* 💬 Chatbot for report interaction
* ⚡ Parallel processing for faster execution
* 🎨 Streamlit-based UI

---

## 🏗️ System Architecture

### Step-by-step pipeline:

1. **User Input** → Enter research topic
2. **Planner Agent** → Generates search queries
3. **Web Retrieval** → Fetches data using SerpAPI
4. **Summarization** → Extracts key insights
5. **Embedding** → Converts text into vectors
6. **Vector Storage** → Stored in FAISS
7. **Context Retrieval** → Top relevant chunks selected
8. **Report Generation** → Structured research report
9. **Critic Agent** → Identifies gaps
10. **Improver Agent** → Refines final output
11. **Chatbot (RAG)** → Answers questions from report

---

## 🧰 Tech Stack

* **Python**
* **Streamlit**
* **OpenRouter API (LLM)**
* **SerpAPI (Web Search)**
* **FAISS (Vector Database)**
* **SentenceTransformers (Embeddings)**
* **NumPy**

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone <your-repo-link>
cd project-folder
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install streamlit faiss-cpu sentence-transformers openai python-dotenv requests numpy
```

### 4️⃣ Add Environment Variables

Create a `.env` file:

```
OPENROUTER_API_KEY=your_key_here
SERPAPI_API_KEY=your_key_here
```

---

## ▶️ Run the Application

```
streamlit run main.py
```

---

## 💡 How It Works

* The system retrieves real-time data from the web
* Converts it into embeddings using SentenceTransformers
* Stores embeddings in FAISS for similarity search
* Generates a research report using LLM
* Allows users to ask questions using a RAG-based chatbot

---

## 🎯 Advantages

* ✔ Reduces hallucination
* ✔ Uses real-time data
* ✔ Structured research output
* ✔ Interactive chatbot support
* ✔ Modular and scalable design

---

## 🔮 Future Scope

* 📚 Integration with academic sources (ArXiv, PubMed)
* 📄 PDF export (research paper format)
* 📑 Automatic citation generation (APA/IEEE)
* 🧠 Advanced RAG (re-ranking, hybrid search)
* ☁️ Cloud deployment


---

## 📜 License

This project is open-source and available under the MIT License.

---

## ❤️ Author

Developed as part of an AI research project.
