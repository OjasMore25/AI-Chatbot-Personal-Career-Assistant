# AI-Chatbot-Personal-Career-Assistant

# 👤 Ojas Ganesh More - Personal AI Chatbot

This is a **personal AI chatbot** that represents [Ojas Ganesh More](https://www.linkedin.com/in/ojasganeshmore) and answers questions about his background, experience, and career using information extracted from his **LinkedIn profile (PDF)** and **precomputed FAISS vector store**.

It uses **LangChain**, **FAISS**, **OpenAI GPT-4o-mini**, and **Gradio** for the frontend UI. It also includes Pushover integration to send important session summaries or contact info.

---

## 🔧 Features

- ✅ RAG-based chatbot using FAISS and HuggingFace Embeddings.
- ✅ Gradio-based UI.
- ✅ Automatically fetches and prepares all necessary files on launch.
- ✅ Summarizes session and pushes report via Pushover when the user ends the chat.
- ✅ Supports function-calling tools to:
  - Record user emails.
  - Record unknown questions.

---

## 📁 File Structure

