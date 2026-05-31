# 🎬 The Cinematic Alchemist Pro

An AI-powered movie discovery, recommendation, and content ideation platform built with **Streamlit**, **LangGraph**, **LLMs**, and **Semantic Search**.

The Cinematic Alchemist Pro transforms a traditional movie catalogue into an intelligent streaming assistant capable of:

* Understanding viewer moods and "vibes"
* Generating personalised recommendations
* Creating watch-party experiences
* Designing original streaming concepts
* Providing conversational movie discovery through Retrieval-Augmented Generation (RAG)

---

## 🚀 Features

### 🎭 Vibe Search

Describe a mood, atmosphere, or viewing preference and receive:

* AI-generated programming analysis
* Semantic movie recommendations
* Ranked content selections
* Personalised results based on user preferences

Example:

> "Dark psychological thrillers with unexpected twists"

The system performs:

1. Semantic vector search
2. Recommendation scoring
3. LLM-powered analysis
4. Personalised ranking

---

### 🚀 Greenlight Studio

Generate entirely new streaming concepts by combining two existing titles.

Example:

> Stranger Things + Black Mirror

Outputs include:

* Original concept
* Series pitch
* Market analysis
* Confidence score
* Downloadable PDF report

This feature simulates a content development and commissioning workflow commonly used by streaming platforms.

---

### 🍿 Watch Planner

Create curated watch-party experiences based on a chosen theme.

Features:

* Intelligent title selection
* Emotional progression planning
* Narrative watch flow generation
* Downloadable PDF watch plan

Example themes:

* Family movie night
* 90s nostalgia
* Mind-bending sci-fi
* Feel-good comedy marathon

---

### 💬 Netflix AI Assistant (RAG)

Conversational movie exploration powered by Retrieval-Augmented Generation.

Capabilities:

* Movie recommendations
* Genre exploration
* Hidden gem discovery
* Question answering against the movie catalogue

The assistant combines:

* Semantic Search
* Vector Embeddings
* Large Language Models

to provide grounded and context-aware responses.

---

## 🏗️ Architecture

### Core Technologies

| Technology        | Purpose                         |
| ----------------- | ------------------------------- |
| Streamlit         | User Interface                  |
| LangGraph         | Workflow orchestration          |
| LLM Provider      | Content generation and analysis |
| Vector Embeddings | Semantic search                 |
| Pandas            | Data processing                 |
| PDF Generator     | Report creation                 |
| Session State     | User memory and personalisation |

---

### Application Flow

```text
User Input
      │
      ▼
Semantic Search
      │
      ▼
Recommendation Ranking
      │
      ▼
LLM Analysis
      │
      ▼
Presentation Layer
      │
      ├── Vibe Recommendations
      ├── Watch Plans
      ├── Pitch Generation
      └── Conversational Assistant
```

---

## 🧠 AI Components

### Semantic Search Engine

The platform uses vector embeddings to retrieve movies based on meaning rather than exact keywords.

Example:

Query:

```text
Movies about hope after loss
```

May retrieve titles that never explicitly contain those words but share similar themes.

---

### Recommendation Ranking

Recommendations are scored using:

* Semantic similarity
* User profile preferences
* Historical interactions
* Genre alignment
* Vibe matching

---

### LangGraph Workflows

#### Vibe Workflow

```text
Vibe Input
    │
    ▼
Recommendation Engine
    │
    ▼
LLM Analysis
```

#### Watch Planner Workflow

```text
Theme Input
    │
    ▼
Recommendation Engine
    │
    ▼
Watch Party Generator
```

---

## 📂 Project Structure

```text
project/
│
├── app.py
│
├── data/
│   └── netflix_sample.xlsx
│
├── utils/
│   ├── llm_provider.py
│   ├── vector_store.py
│   ├── recommender.py
│   ├── data_quality.py
│   ├── pdf_utils.py
│   └── poster_generator.py
│
├── requirements.txt
│
└── README.md
```

---

## 📊 Data Quality Monitoring

The application includes built-in data health monitoring.

Metrics include:

* Column coverage
* Data completeness
* Recommendation confidence scores

Displayed directly within the Streamlit sidebar.

---

## 💾 Session Memory

The application maintains lightweight user memory during a session.

Stored information includes:

```python
{
    "genres": [],
    "actors": [],
    "vibes": []
}
```

This enables progressively personalised recommendations.

---

## 📄 PDF Export

Users can export:

### Pitch Reports

Contains:

* Generated concept
* Market analysis
* Confidence score

### Watch Party Plans

Contains:

* Theme
* Movie selections
* Viewing narrative

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/cinematic-alchemist-pro.git

cd cinematic-alchemist-pro
```

### Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows

```bash
.venv\Scripts\activate
```

Mac/Linux

```bash
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will launch at:

```text
http://localhost:8501
```

---

## 🔧 Configuration

The application requires:

### Movie Dataset

```text
data/netflix_sample.xlsx
```

Expected sheet:

```text
Additional IMDb Data
```

### Embeddings

Pre-generated vector embeddings should be available through:

```python
load_embeddings()
```

### LLM Provider

Configure your preferred LLM in:

```python
utils/llm_provider.py
```

Supported providers can include:

* OpenAI
* Huggingface
* Open Router
* Groq
* Local Models

---

## 📈 Future Enhancements

Potential roadmap items:

* Movie poster generation
* Hybrid semantic + keyword search
* Collaborative filtering
* User authentication
* Persistent user profiles
* Real-time trending content analysis
* Multi-modal recommendations
* Agentic content programming workflows
* Streaming platform integration

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

Built using:

* Streamlit
* LangGraph
* Pandas
* Sentence Transformers
* Large Language Models
* Retrieval-Augmented Generation (RAG)

Inspired by modern streaming recommendation systems and AI-powered content programming platforms.

