from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_document(filepath: str) -> list[Document]:
    loader = PyPDFLoader(filepath)
    return loader.load()


def split_documents(docs: list[Document]) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(docs)


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


if __name__ == "__main__":
    path = Path(Path.cwd(), "docs", "dnd-5e-handbook.pdf")
    doc_chunks = load_document(path)
    print(doc_chunks[3])
