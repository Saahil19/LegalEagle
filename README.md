# LegalEagle — Automated Legal Document Review

![LegalEagle Banner](https://img.shields.io/badge/AI-Legal%20Document%20Review-6366f1?style=for-the-badge)

LegalEagle is an end-to-end, enterprise-grade AI legal contract auditor. It allows users to securely upload legal contracts (PDF/TXT) to instantly extract key clauses, detect hidden liabilities, and chat directly with the document using an intelligent RAG (Retrieval-Augmented Generation) agent.

## 🚀 Key Features

- **Deep Clause Extraction:** Uses a fine-tuned BERT Named Entity Recognition (NER) model to accurately extract governing laws, termination clauses, parties, and confidentiality agreements.
- **Automated Risk Scoring:** Evaluates extracted clauses using a Local LLM (Flan-T5) to grade liabilities on a scale of 1-10. High-risk liabilities are instantly flagged for attorney review.
- **Interactive Chat Agent:** Features a Server-Sent Events (SSE) streaming chat interface. Ask questions like *"What is the governing law?"* or *"Are there any non-compete clauses?"* and get instant, context-aware answers.
- **Secure Authentication:** Built-in mock authentication system securing the entire application.
- **Enterprise UI/UX:** A stunning glassmorphism dashboard built with Next.js and Tailwind-inspired custom CSS.

## 🛠️ Technology Stack

- **Frontend:** Next.js (App Router), React, TypeScript
- **Backend:** FastAPI, Python, SQLAlchemy
- **Database:** SQLite (Easily configurable to PostgreSQL via environment variables)
- **AI & NLP:** PyTorch, HuggingFace Transformers, ONNX Runtime (Quantized Models)
- **Vector Database (RAG):** Qdrant

## 📥 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/LegalEagle.git
cd LegalEagle
```

### 2. Backend Setup (FastAPI)
Navigate to the `backend` directory, create a virtual environment, and start the server:
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
*The backend API will be running at `http://localhost:8000`*

### 3. Frontend Setup (Next.js)
Open a new terminal window, navigate to the `frontend` directory, install dependencies, and start the development server:
```bash
cd frontend
npm install
npm run dev
```
*The frontend application will be running at `http://localhost:3000`*

## 💡 Usage Guide

1. Open your browser and navigate to `http://localhost:3000`.
2. **Sign In:** Use any mock credentials to bypass the secure login wall.
3. **Upload a Contract:** Drag and drop a legal contract `.txt` or `.pdf` file (max 10MB). Sample highly-risky contracts are available in `data/sample_contracts/`.
4. **View Audit:** Wait 1-3 minutes for the AI to complete its analysis. You will be redirected to the Risk Dashboard.
5. **Chat:** Use the chat window on the right side of the dashboard to ask specific questions about the document's obligations.

## 🧠 Model Architecture

- **Information Extraction:** `bert-ner-cuad-final` (Fine-tuned on the CUAD dataset)
- **Embeddings:** `all-MiniLM-L6-v2`
- **Generative Scoring:** `google/flan-t5-base`

*Note: The models have been quantized using ONNX Runtime for a 75% reduction in size and 4x faster CPU inference, allowing the entire pipeline to run locally without expensive GPU clusters or external API costs.*

## 📜 License
This project is open-source and available under the MIT License.
