# 🤖 AI Data Analysis Agent (In Progress)

An AI-powered data analysis system that automates data profiling, SQL query generation, and insight extraction from raw datasets.

> 🚧 **Status:** In Progress — actively building core modules and expanding functionality

---

## 🚀 Overview

This project aims to simplify the data analysis workflow by allowing users to upload datasets and ask questions in natural language. The system automatically analyzes the data, generates SQL queries, and produces insights with visualizations.

---

## ✨ Features (Planned & In Progress)

- 📊 **Data Profiling**
  - Detect column types (numeric, categorical, date, text)
  - Calculate statistics (mean, min, max, std)
  - Identify missing values and data quality issues

- 🧠 **Natural Language to SQL (LLM-powered)**
  - Convert user questions into SQL queries
  - Validate queries against dataset schema

- ⚡ **Query Execution**
  - Execute SQL queries on structured data
  - Return results as tables

- 📈 **Insight Generation**
  - Generate key insights, anomalies, and recommendations
  - Suggest follow-up questions

- 📉 **Automated Visualization**
  - Auto-select chart types (bar, line, scatter, histogram)
  - Create interactive visualizations

---

## 🏗️ Architecture

The system is designed using a modular architecture:
User → Orchestrator → Modules → Database → Results


### Core Components:

- **Data Profiler** – Analyzes dataset structure and quality  
- **SQL Generator** – Converts natural language to SQL (LLM-based)  
- **Query Executor** – Safely executes SQL queries  
- **Insight Generator** – Produces human-readable insights  
- **Dashboard Builder** – Generates visualizations  
- **Orchestrator** – Coordinates all components  

---

## 🛠️ Tech Stack

- **Python**
- **Pandas, NumPy** – Data processing
- **SQLite / PostgreSQL** – Database
- **Streamlit** – Frontend UI
- **Plotly** – Visualization
- **Claude API (LLM)** – SQL generation & insights

---

## 📂 Project Structure
data-analysis-agent/
│
├── frontend/ # Streamlit UI
├── agent/ # Orchestrator & state management
├── modules/ # Core components
├── llm/ # LLM integration
├── database/ # DB connections & validation
├── utils/ # Helper functions
├── tests/ # Unit tests


---

## 🧪 Current Progress

- [x] Project architecture designed  
- [ ] Data Profiler module (in development)  
- [ ] Query Executor  
- [ ] LLM integration (SQL generation)  
- [ ] Insight generation  
- [ ] Visualization dashboard  

---

## 🎯 Goal

To build an end-to-end AI agent that reduces manual data analysis work by automating:

- Data understanding  
- Query writing  
- Insight generation  
- Visualization  

---

## 📌 Future Improvements

- Advanced SQL validation and safety checks  
- Query caching for performance  
- Handling large-scale datasets  
- Multi-user support with authentication  
- Deployment on cloud (AWS/GCP)

---

## 🤝 Contributing

This is a personal learning project, but suggestions and feedback are welcome!

---

## ⭐ Motivation

Data analysts often repeat the same tasks — this project aims to automate that workflow using AI and build a smarter data analysis experience.
