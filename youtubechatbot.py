import streamlit as st
import re
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="YouTube Chatbot", page_icon="📺")
st.title("📺 YouTube Video Chatbot")
st.markdown("Ask questions based on any YouTube video with captions!")

# Extract video ID from a YouTube URL
def extract_video_id(url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None

url = st.text_input("🎥 Enter YouTube Video URL:")

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
    else:
        st.success(f"✅ Video ID extracted: {video_id}")

        try:
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
            transcript = " ".join(chunk["text"] for chunk in transcript_list)

            # Display transcript preview
            with st.expander("📜 Transcript Preview"):
                st.write(transcript[:1500] + "..." if len(transcript) > 1500 else transcript)

            # Split transcript into chunks
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.create_documents([transcript])

            # Embed and store in vector store
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(chunks, embeddings)

            # Set up retriever
            retriever = vector_store.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 2, 'lambda_mult': 0.5}
            )

            # Define LLM and prompt
            llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            # Define pipeline
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            chain = (
                RunnableParallel({
                    'context': retriever | RunnableLambda(format_docs),
                    'question': RunnablePassthrough()
                })
                | prompt
                | llm
                | StrOutputParser()
            )

            # Question input
            question = st.text_input("💬 Ask a question about the video:")

            if question:
                with st.spinner("🧠 Generating answer..."):
                    try:
                        docs = retriever.get_relevant_documents(question)
                        if not docs:
                            st.warning("⚠️ No relevant transcript sections found.")
                        else:
                            answer = chain.invoke(question)
                            if not answer.strip():
                                st.error("⚠️ The model returned an empty response.")
                            else:
                                st.subheader("💡 Answer")
                                st.write(answer)
                    except Exception as e:
                        st.error(f"❌ Error during generation: {e}")

        except TranscriptsDisabled:
            st.error("❌ Captions are disabled for this video.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
