from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader("data/cv.pdf")

documents = loader.load()

#splitting the large pdf into smaller pieces
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)

for c in chunks:
    print(c.page_content)

#create the embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

#create the chroma vector store
vector_store = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings
)

#create the retriever
retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 2}
)


def get_relevant_chunks(query):
    relevant_chunks = retriever.invoke(query)
    return relevant_chunks

def get_content(results):
    return [result.page_content for result in results]

while True:
    question = input("Ask a question about the uploaded document: (type stop to stop the program.)")
    if question == "stop":
        print("=" * 50)
        print("exiting from the program")
        print("=" * 50)
        break

    relevant_chunks = get_relevant_chunks(question)
    results = get_content(relevant_chunks)

    print("Two best matching results are: ")
    print("\n\n")
    for r in results:
        print("=" * 50)
        print(r)
        print("=" * 50)
        print("\n")        