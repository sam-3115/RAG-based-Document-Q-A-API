import streamlit as st
from app import rag
from pypdf import PdfReader


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="centered"
)


# --------------------------------------------------
# Title
# --------------------------------------------------

st.title("📄 RAG Document Q&A")

st.write(
    "Upload a PDF or TXT document and ask questions "
    "about its content using Retrieval-Augmented Generation."
)


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# --------------------------------------------------
# Document Upload
# --------------------------------------------------

st.subheader("1. Upload Document")

uploaded_file = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"]
)


if uploaded_file is not None:

    if st.button("📤 Upload Document", type="primary"):

        try:

            # ------------------------------------------
            # PDF
            # ------------------------------------------

            if uploaded_file.type == "application/pdf":

                reader = PdfReader(uploaded_file)

                text = ""

                for page in reader.pages:

                    page_text = page.extract_text()

                    if page_text:
                        text += page_text + "\n"


            # ------------------------------------------
            # TXT
            # ------------------------------------------

            else:

                text = uploaded_file.read().decode("utf-8")


            # ------------------------------------------
            # Check extracted text
            # ------------------------------------------

            if not text.strip():

                st.error(
                    "❌ Could not extract any text from this document."
                )

            else:

                with st.spinner(
                    "Processing document and creating embeddings..."
                ):

                    doc_id = rag.ingest_document(text)


                # Store document information
                st.session_state.doc_id = doc_id
                st.session_state.document_name = uploaded_file.name


                st.success(
                    f"✅ {uploaded_file.name} uploaded successfully!"
                )


        except Exception as e:

            st.error(f"❌ Upload failed: {e}")


# --------------------------------------------------
# Question Answering
# --------------------------------------------------

if st.session_state.doc_id is not None:

    st.divider()

    st.subheader("2. Ask a Question")

    st.info(
        f"📄 Current document: **{st.session_state.document_name}**"
    )


    # Question input
    question = st.text_input(
        "Enter your question",
        placeholder="What is the main idea of this document?"
    )


    # Number of retrieved chunks
    top_k = st.slider(
        "Number of relevant chunks",
        min_value=1,
        max_value=5,
        value=3
    )


    # Ask button
    if st.button("🔍 Ask Question", type="primary"):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                with st.spinner(
                    "Searching the document and generating answer..."
                ):

                    result = rag.answer_question(
                        st.session_state.doc_id,
                        question,
                        k=top_k
                    )


                # --------------------------------------
                # Answer
                # --------------------------------------

                st.subheader("🤖 Answer")

                st.write(
                    result["answer"]
                )


                # --------------------------------------
                # Sources
                # --------------------------------------

                st.divider()

                st.subheader("📚 Retrieved Sources")


                if result["sources"]:

                    for i, source in enumerate(
                        result["sources"],
                        start=1
                    ):

                        with st.expander(
                            f"Source {i} • Similarity: "
                            f"{source['score']:.4f}"
                        ):

                            st.write(
                                source["text"]
                            )

                else:

                    st.info(
                        "No relevant sources were found."
                    )


            except Exception as e:

                st.error(
                    f"❌ Query failed: {e}"
                )


# --------------------------------------------------
# Initial message
# --------------------------------------------------

else:

    st.info(
        "👆 Upload a PDF or TXT document above "
        "to start asking questions."
    )