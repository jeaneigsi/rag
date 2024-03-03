import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

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

def get_conversation_chain(vectorestores):
    llm=ChatGoogleGenerativeAI(model="gemini")
    memory=ConversationBufferMemory(memory_key='chat history', return_messages=True) # initialize memory
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,retriever=vectorestores.as_retriever(),memory=memory
    )
    return conversation_chain



def get_vectorestore(text_chunks):
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    vectorestore=FAISS.from_texts(text_chunks,embedding=embedding)
    return vectorestore


def main():
    load_dotenv()
    st.set_page_config("Chat with multiple pdfs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None




    st.header("Chat with multiple pdfs :books:")
    st.text_input("Ask a question about your documents: ")

    st.write(user_template.replace("{{MSG}}","Hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","Hello human"), unsafe_allow_html=True)

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
                vectorstore=get_vectorestore(text_chunks)

                #conversation
                st.session_state.conversation=get_conversation_chain(vectorstore) #st.state object increase the scope a variable outside st.sidebar

    st.session_state.conversation




if __name__ == "__main__":
    main()