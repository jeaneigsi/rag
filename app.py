import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.cache_
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader=PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text

def get_text_chunks(text):
    textsplitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=250,
        length_function=len

    )
    chunks=textsplitter.split_text(text)
    return chunks


def get_vectorestore(text_chunks):
    embedding=OpenAiEmbeddings()
    vectorestore=FAISS.from_embeddings(embedding, text_chunks)
    return vectorestore


def main():
    load_dotenv()
    st.set_page_config("Chat with multiple pdfs", page_icon=":books:")
    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask a question about your documents: ")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs=st.file_uploader("Upload your pdfs",type="pdf",accept_multiple_files=True)
        if st.button("Run"):
            with st.spinner("Loading..."):
                #get pdfs and chuncks of text
                raw_text=get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #get text chunks
                text_chunks=get_text_chunks(raw_text)
                # st.write(text_chunks)

                #vetoriser
                vectorstore=get_vectorstore(text_chunks)




if __name__ == "__main__":
    main()