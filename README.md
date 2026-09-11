# 📄 PDF Semantic Search with Chroma

A small Python project demonstrating how to **load a PDF, split it into chunks, generate embeddings, store them in a Chroma vector database, and retrieve the most semantically relevant chunks based on a user's question**.

This project focuses specifically on understanding the **vector storage and retrieval stage of a RAG pipeline**, without involving an LLM.

---

## 🚀 Project Overview

Traditional keyword search looks for exact words.

Semantic search works differently. It converts text into numerical **embeddings** that represent the meaning of the text, allowing the system to find information that is conceptually related to a question.

This project implements the following pipeline:

```text
                PDF
                 │
                 ▼
          PyPDFLoader
                 │
                 ▼
        Document Objects
                 │
                 ▼
       Text Chunking
                 │
                 ▼
       HuggingFace Embeddings
                 │
                 ▼
        Chroma Vector Store
                 │
                 ▼
            Retriever
                 │
                 ▼
          User Question
                 │
                 ▼
      Relevant Text Chunks
```

---

## ✨ Features

* 📄 Load PDF documents using `PyPDFLoader`
* ✂️ Split documents into smaller chunks
* 🧠 Generate semantic embeddings locally using Hugging Face
* 🗄️ Store embeddings in a Chroma vector database
* 🔍 Perform semantic similarity search
* 🎯 Retrieve the top `K` most relevant document chunks
* 💻 Interactive command-line interface
* 🤖 No LLM required

---

## 🛠️ Technologies Used

| Technology                               | Purpose                           |
| ---------------------------------------- | --------------------------------- |
| **Python**                               | Core programming language         |
| **LangChain**                            | Document processing and retrieval |
| **PyPDFLoader**                          | PDF document loading              |
| **RecursiveCharacterTextSplitter**       | Text chunking                     |
| **Hugging Face / Sentence Transformers** | Local text embeddings             |
| **Chroma**                               | Vector database                   |
| **CLI**                                  | User interaction                  |

---

## 📁 Project Structure

```text
pdf-semantic-search/
│
├── data/
│   └── cv.pdf
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

> **Note:** Do not upload a real personal CV to a public GitHub repository. Use a sample or anonymised PDF instead.

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/pdf-semantic-search.git
cd pdf-semantic-search
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you haven't created `requirements.txt` yet, the main packages are:

```text
langchain
langchain-community
langchain-text-splitters
langchain-chroma
langchain-huggingface
sentence-transformers
pypdf
```

---

## ▶️ Running the Project

Place a PDF inside the `data` folder:

```text
data/
└── cv.pdf
```

Then run:

```bash
python chroma_ex.py
```

You will be prompted to enter a question:

```text
Ask a question about the uploaded document:
```

For example:

```text
What programming languages are mentioned?
```

The system retrieves the two most relevant chunks from the vector database.

Example output:

```text
==================================================
Python, Java and SQL are some of the programming
languages used in software development projects.
==================================================

==================================================
Experience includes Python development and machine
learning projects using Python libraries.
==================================================
```

---

## 🧠 How It Works

### 1. Load the PDF

`PyPDFLoader` converts the PDF into LangChain `Document` objects.

```python
loader = PyPDFLoader("data/cv.pdf")
documents = loader.load()
```

Each document contains information such as:

* `page_content`
* `metadata`

---

### 2. Split the document

Large documents are divided into smaller pieces.

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(documents)
```

The overlap helps preserve context between neighbouring chunks.

For example:

```text
Chunk 1
████████████████████

        █████
        overlap

        Chunk 2
        ████████████████████
```

---

### 3. Generate embeddings

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert each text chunk into a numerical vector.

Conceptually:

```text
"Python developer with machine learning experience"
                    ↓
        [0.12, -0.43, 0.87, ...]
```

Similar meanings produce vectors that are closer together in the embedding space.

The embeddings are generated **locally**, so an external LLM API is not required.

---

### 4. Store vectors in Chroma

The chunks and their embeddings are stored in Chroma:

```python
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
```

Chroma allows the application to efficiently search the stored vectors.

---

### 5. Retrieve relevant chunks

A retriever is created from the vector store:

```python
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)
```

The `k=2` parameter means that the system returns the **two most relevant chunks**.

When a question is submitted:

```python
relevant_chunks = retriever.invoke(question)
```

the question is compared semantically against the stored document embeddings.

---

## 🔎 Example

Suppose the document contains:

```text
The student has experience developing machine learning
models using Python, Scikit-learn and XGBoost.
```

The user asks:

```text
What ML technologies have I worked with?
```

Even though the question doesn't exactly match the wording in the document, the embedding model can identify the text as semantically relevant.

```text
User Question
      │
      ▼
Question Embedding
      │
      ▼
Compare with stored vectors
      │
      ▼
Similarity Search
      │
      ▼
Top 2 Relevant Chunks
```

---

## 🎯 Purpose of the Project

This project was built as a **learning project to understand vector databases and semantic retrieval**, which are important components of modern AI applications and Retrieval-Augmented Generation (RAG) systems.

The project intentionally does **not** include an LLM.

The focus is:

```text
Document → Chunks → Embeddings → Vector Store → Retrieval
```

rather than:

```text
Document → Chunks → Embeddings → Vector Store
                                      ↓
                                  Retrieval
                                      ↓
                                     LLM
                                      ↓
                                   Answer
```

---

## 🔮 Future Improvements

Possible extensions include:

* [ ] Add an LLM for answer generation
* [ ] Convert the project into a full RAG application
* [ ] Add a Streamlit user interface
* [ ] Support multiple PDF documents
* [ ] Add document metadata filtering
* [ ] Allow users to change the number of retrieved chunks
* [ ] Persist the Chroma database between sessions
* [ ] Add similarity scores to retrieved results
* [ ] Add conversation history
* [ ] Integrate the retrieval system into a Career Copilot application

---

## 📚 What I Learned

Through this project, I learned how to:

* Work with LangChain `Document` objects
* Load and process PDF files
* Split large documents into chunks
* Generate text embeddings
* Understand semantic similarity
* Store embeddings in a vector database
* Use Chroma for vector storage
* Build a retriever with LangChain
* Perform semantic search without an LLM

---

## 👨‍💻 Author

**Thenul Jayarathna**

BSc (Hons) Computer Science with Artificial Intelligence
Birmingham City University

Interested in:

* Artificial Intelligence & Machine Learning
* Generative AI
* RAG Systems
* AI Agents
* Backend Development
* Python
* LangChain

---

## ⭐ If you found this project useful

Feel free to explore the repository, experiment with different embedding models, chunk sizes, and retrieval settings.
