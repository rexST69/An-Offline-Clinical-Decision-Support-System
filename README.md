# An-Offline-Clinical-Decision-Support-System
Offline-first AI Clinical Decision Support System for rural healthcare. Uses a deterministic "Parallel Triage Engine" (Local RAG) to safely diagnose maternal vs. neonatal symptoms without internet. Built with Python &amp; Streamlit to empower ASHA workers with official NHM Module 6 protocols.
# Project Janani-Raksha 🏥
**AI-Architected Clinical Decision Support System (CDSS) for Rural Maternal Health**

![Status](https://img.shields.io/badge/Status-Prototype-blue)
![Stack](https://img.shields.io/badge/Tech-Streamlit%20%7C%20Python%20%7C%20RAG-green)

## 📌 Overview
**Project Janani-Raksha** is an Information System designed to reduce "Decision Latency" in rural healthcare. It empowers frontline ASHA (Accredited Social Health Activist) workers with a deterministic, offline-ready triage engine.

Unlike standard Generative AI (which can hallucinate), this system uses a **Parallel Triage Architecture** to retrieve exact medical protocols from the **National Health Mission (NHM) Module 6 Guidelines**, ensuring 100% clinical accuracy.

## ⚙️ System Architecture
The system solves the "Context Collapse" problem found in standard LLMs (where mother and child symptoms get confused) using a 3-stage pipeline:

1.  **Multi-Intent Splitter:** Decomposes complex queries (e.g., *"Mother bleeding and baby fever"*) into atomic sub-queries.
2.  **Context Firewall:** A logic layer that routes maternal symptoms to the Maternal Database and neonatal symptoms to the Child Database, preventing cross-contamination.
3.  **Hybrid Retrieval:** Uses `sentence-transformers` (all-MiniLM-L6-v2) for semantic understanding, allowing the system to understand colloquial village language.

## ✨ Key Features
* **Parallel Triage:** Can diagnose multiple patients (Mother + Child) simultaneously in a single query.
* **Vernacular Toggle:** Instant English-to-Hindi translation for rural accessibility.
* **Active Intervention:** Generates pre-filled **WhatsApp Referral Messages** for critical cases, bridging the gap between diagnosis and hospital admission.
* **Offline First:** Runs locally without requiring heavy cloud APIs, suitable for low-connectivity zones.

## 🛠️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/project-janani-raksha.git](https://github.com/your-username/project-janani-raksha.git)
    cd project-janani-raksha
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

## 🧪 Test Scenarios (Red Teaming)
To verify the "Firewall" logic, try these inputs:
* **Multi-Context:** `Mother has high BP and Baby has fever` (Should show two distinct cards).
* **Ambiguity:** `She is vomiting` (Should trigger Maternal Fallback).
* **Critical:** `Bleeding` (Should trigger Red Alert and WhatsApp Link).

## 📄 Disclaimer
This is a student project for the **Social Work (Term 5)** course. While based on official NHM guidelines, it is a prototype and should **not** be used for actual medical treatment without clinical validation.

---
*Built by Rex Yohan Tirkey | Information Systems & Management*
