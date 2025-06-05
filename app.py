from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import time

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import requests
import zipfile

load_dotenv(override=True)

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url} ...")
        response = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(response.content)


def setup_files():
    os.makedirs("me", exist_ok=True)

    # Download the FAISS zip
    faiss_url = "https://huggingface.co/datasets/OjasMore/convoo/resolve/main/faiss_index.zip"
    faiss_path = "me/faiss_index.zip"
    with open(faiss_path, "wb") as f:
        f.write(requests.get(faiss_url).content)

    # Extract it
    with zipfile.ZipFile(faiss_path, 'r') as zip_ref:
        zip_ref.extractall("me/faiss_index")

    # Download the PDF
    pdf_url = "https://huggingface.co/datasets/OjasMore/convoo/resolve/main/linkedin.pdf"
    with open("me/linkedin.pdf", "wb") as f:
        f.write(requests.get(pdf_url).content)


setup_files()



# ========== Pushover Integration ==========

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )

# ========== Tool Functions ==========

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Use this tool to record any question that couldn't be answered",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

# ========== Main Class ==========

class Me:
    def __init__(self):
        self.openai = OpenAI()
        self.name = "Ojas Ganesh More"

        # Load LinkedIn PDF
        reader = PdfReader("me/linkedin.pdf")

        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

        # Load Summary Text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        # Setup FAISS
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        if not os.path.exists("me/faiss_index"):
            self.build_vector_store()
        self.vector_store = FAISS.load_local("me/faiss_index", self.embeddings, allow_dangerous_deserialization=True)


        # Session state
        self.session_start_time = None
        self.session_history = []

    def build_vector_store(self):
        loader = PyPDFLoader("me/linkedin.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(docs, self.embeddings)
        vector_store.save_local("me/faiss_index")

    def retrieve_context(self, query):
        docs = self.vector_store.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in docs])

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            print(f"Tool called: {tool_name}", flush=True)
            tool = globals().get(tool_name)
            result = tool(**arguments) if tool else {}
            results.append({
                "role": "tool",
                "content": json.dumps(result),
                "tool_call_id": tool_call.id
            })
        return results

    def system_prompt(self):
        prompt = (
            f"You are acting as {self.name}. You are answering questions on {self.name}'s website, "
            f"particularly questions related to {self.name}'s career, background, skills and experience. "
            f"Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. "
            f"You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. "
            f"Be professional and engaging, as if talking to a potential client or future employer who came across the website. "
            f"If you don't know the answer to any question, use your record_unknown_question tool to record the question. "
            f"If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email "
            f"and record it using your record_user_details tool.\n\n"
            f"## Summary:\n{self.summary}\n\n"
            f"## LinkedIn Profile:\n{self.linkedin}\n\n"
            f"With this context, please chat with the user, always staying in character as {self.name}."
            f"Also at the start of the conversation tell the user that please write 'bye' so that Ojas will get a summerize report of the the coversation  ."
        )
        return prompt

    def summarize_chat(self):
        """Use the OpenAI model to summarize the chat session."""
        if not self.session_history:
            return "No meaningful conversation to summarize."

    # Prepare history as a prompt for summarization
        history_text = ""
        for turn in self.session_history:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    # Summarization prompt
        summarization_prompt = (
            "Summarize the following conversation between a user and a virtual assistant representing Ojas Ganesh More. "
            "Include only key topics, questions asked, and how the assistant responded. Be brief and professional.\n\n"
            f"{history_text}"
        )

    # Call OpenAI to summarize
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
                      {"role": "user", "content": summarization_prompt}],
            max_tokens=300,
            temperature=0.5
        )

        summary = response.choices[0].message.content.strip()
        return summary


    def chat(self, message, history):
        if self.session_start_time is None:
            self.session_start_time = time.time()

        # def clean_messages(messages):
        #     allowed_keys = {"role", "content"}
        #     return [{k: m[k] for k in allowed_keys if k in m} for m in messages]

        # Check for manual end
        if message.strip().lower() in {"end", "end session", "stop", "bye"}:
            summary = self.summarize_chat()
            push(summary)
            self.session_start_time = None
            self.session_history.clear()
            return "Session ended. A summary has been sent."

        # Timeout
        if time.time() - self.session_start_time > 60000:
            summary = self.summarize_chat()
            push(summary)
            self.session_start_time = None
            self.session_history.clear()
            return "Session timed out after 1 minute. A summary has been sent."

        rag_context = self.retrieve_context(message)
        full_system_prompt = f"{self.system_prompt()}\n\n# Retrieved Context:\n{rag_context}"
        messages = [{"role": "system", "content": full_system_prompt}] + history + [{"role": "user", "content": message}]

        done = False
        while not done:
            # messages = clean_messages(messages)
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", messages=messages, tools=tools
            )
            if response.choices[0].finish_reason == "tool_calls":
                tool_calls = response.choices[0].message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(response.choices[0].message)
                messages.extend(results)
            else:
                done = True

        reply = response.choices[0].message.content
        self.session_history.append({"user": message, "assistant": reply})
        return reply

# ========== Gradio UI ==========

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
