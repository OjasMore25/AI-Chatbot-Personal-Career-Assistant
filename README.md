# AI-Chatbot-Personal-Career-Assistant

# 🤖 Ojas Ganesh More - Career Chatbot

This intelligent chatbot represents **Ojas Ganesh More** and answers questions related to his career, background, and skills. It uses RAG (Retrieval-Augmented Generation) with FAISS and LangChain, making it capable of accurate, context-aware conversations.

---

## ✨ Features

- 🧠 **RAG-based Chatbot** using FAISS + HuggingFace Embeddings  
- 📄 Context retrieved from Ojas's LinkedIn profile (PDF)  
- 💬 Powered by OpenAI's `gpt-4o-mini` model  
- 🧰 Tool functions to:
  - Record user interest (email + name)
  - Log unknown questions
- 🔔 **Session summarization** and Pushover notification support  
- ⏱️ Auto session timeout after 1 minute or manual "bye" command  

---

## 📁 Project Structure

project/
-│
-├── app.py # Main application script
-├── .env # Environment variables for OpenAI + Pushover
-├── me/
-│ ├── faiss_index/ # Contains index.faiss and index.pkl
-│ ├── linkedin.pdf # LinkedIn profile used for RAG
-│ └── summary.txt # Manually written summary of Ojas
-└── README.md # Project documentation



---

## ⚙️ Setup Instructions

### 1. Clone the Repository

git clone https://github.com/yourusername/ojas-career-chatbot.git
cd ojas-career-chatbot


### 2. Install Required Packages

You can create a `requirements.txt`, but here are the essentials:


### 3. Create `.env` File

In the project root, create a `.env` file with the following contents:

-OPENAI_API_KEY=your-openai-key
-PUSHOVER_TOKEN=your-pushover-token
-PUSHOVER_USER=your-pushover-user-key


---

## 🚀 Run the Application


It will automatically:

- Download `faiss_index.zip` and extract the FAISS index  
- Download the latest `linkedin.pdf`  
- Launch the Gradio Chat UI in your browser  

---

## 📦 Downloads Required

Make sure these two public files are hosted (as they are auto-fetched):

- **FAISS Index Zip:**  
  https://huggingface.co/datasets/OjasMore/convoo/resolve/main/faiss_index.zip

- **LinkedIn PDF:**  
  https://huggingface.co/datasets/OjasMore/convoo/resolve/main/linkedin.pdf

---

## 🧠 How It Works

- Loads a PDF (LinkedIn) and splits it into text chunks  
- FAISS vector store is queried with HuggingFace embeddings  
- Retrieved context + user message is sent to `gpt-4o-mini`  
- AI responds in the tone and persona of Ojas Ganesh More  
- If user types `bye` or if session exceeds 1 minute, a summary is created and pushed via Pushover  

---

## 🛠️ Tool Functions

- `record_user_details(email, name, notes)`  
  Pushes contact details to Pushover  

- `record_unknown_question(question)`  
  Pushes unanswerable questions to Pushover  

---

## ✅ Example Usage

Start a session and chat:

**You:** What kind of projects has Ojas worked on?  
**Bot:** Ojas has worked on machine learning systems, particularly in finance and hedge fund automation...

**You:** bye  

✅ Bot pushes a session summary to Pushover

---

## 💡 

- Added support for more document types (e.g. resume, blog posts)  
- Integrated with Hugging Face Spaces for easy deployment  
- Auto-email session summary to user  

---

## 👨‍💻 Author

**Ojas Ganesh More**  
🔗 [LinkedIn](https://www.linkedin.com/in/ojasmore25/)

---

## 📜 License

MIT License – free to use, modify, and distribute.


