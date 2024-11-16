import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import os
from pdf_reader import load_pdf
from typing import List, Dict, Any
import tempfile
import shutil

def get_resume(job_description: str, resume_file, openai_api_key: str):
    """
    Create a conversational resume analyzer that compares a PDF resume against a job description.
    Using existing load_pdf function.
    """
    # Create a temporary directory for Chroma
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Use existing load_pdf function
        resume_docs = load_pdf(resume_file)  # This already returns split documents
        
        # Create document for job description
        job_doc = Document(
            page_content=f"JOB DESCRIPTION:\n{job_description.strip()}",
            metadata={"source": "job_posting"}
        )
        
        # Combine documents (job description and resume docs)
        all_docs = [job_doc] + (resume_docs if isinstance(resume_docs, list) else [resume_docs])
        
        # Create vector database
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
            persist_directory=persist_directory
        )
        
        # Persist the database
        vectordb.persist()
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Template for initial analysis
        analysis_template = """
        You are an expert resume analyzer and career advisor. First, analyze the resume and job description provided in the context.
        
        Perform a detailed analysis:
        1. Calculate a match percentage based on:
           - Required skills match
           - Experience level match
           - Education requirements match
           - Industry alignment
        
        2. Provide a brief explanation of the rating, highlighting:
           - Key matching qualifications
           - Areas where the candidate exceeds requirements
           - Potential gaps or areas for improvement
        
        Present this information in a clear, professional format. End by asking if the user would like to know more details about specific aspects of the match.
        
        Context: {context}
        Question: {question}
        Chat History: {chat_history}
        """
        
        # Template for follow-up conversations
        chat_template = """
        You are a helpful career advisor assistant. Using the provided resume and job description:
        
        1. Answer any questions about:
           - Specific skills or experiences from the resume
           - How well certain qualifications match the job requirements
           - Suggestions for improving the application
           - Career advice related to the position
        
        2. Always maintain context from the previous conversation
        
        3. If you don't find specific information in the resume or job description, acknowledge that and provide general advice instead.
        
        Be conversational but professional. Provide specific examples from the resume or job description when possible.
        
        Context: {context}
        Question: {question}
        Chat History: {chat_history}
        """
        
        # Create separate chains for analysis and chat
        def create_chain(template: str) -> ConversationalRetrievalChain:
            prompt = PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=template
            )
            
            return ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(
                    temperature=0.7,
                    model_name='gpt-4',
                    openai_api_key=openai_api_key
                ),
                retriever=vectordb.as_retriever(search_kwargs={'k': 10}),
                memory=memory,
                combine_docs_chain_kwargs={"prompt": prompt}
            )
        
        analysis_chain = create_chain(analysis_template)
        chat_chain = create_chain(chat_template)
        
        class ResumeAssistant:
            def __init__(self):
                self.initial_analysis_done = False
                self._persist_dir = persist_directory
            
            def chat(self, question: str) -> str:
                try:
                    if not self.initial_analysis_done:
                        # First interaction: Provide analysis and rating
                        result = analysis_chain({"question": "Analyze the resume match for this position"})
                        self.initial_analysis_done = True
                        return result['answer']
                    else:
                        # Subsequent interactions: Regular chat
                        result = chat_chain({"question": question})
                        return result['answer']
                except Exception as e:
                    return f"Error processing question: {str(e)}"
            
            def __del__(self):
                # Cleanup the temporary directory when the assistant is deleted
                if hasattr(self, '_persist_dir') and os.path.exists(self._persist_dir):
                    shutil.rmtree(self._persist_dir)
        
        return ResumeAssistant()
        
    except Exception as e:
        # Clean up the temporary directory in case of errors
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        raise ValueError(f"Error in resume analysis: {str(e)}")