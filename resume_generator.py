from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
import os
from pdf_reader import load_pdf
from typing import List, Dict, Any
import tempfile
import shutil

import sys

# SQLite workaround for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma

def get_resume(job_description: str, resume_file, openai_api_key: str):
    """
    Create a conversational resume analyzer that compares a PDF resume against a job description.
    Using existing load_pdf function.
    """
    persist_directory = tempfile.mkdtemp()
    
    try:
        # Load documents
        resume_docs = load_pdf(resume_file)
        job_doc = Document(
            page_content=f"JOB DESCRIPTION:\n{job_description.strip()}",
            metadata={"source": "job_posting"}
        )
        all_docs = [job_doc] + (resume_docs if isinstance(resume_docs, list) else [resume_docs])
        
        # Initialize OpenAI LLM
        llm = ChatOpenAI(
            temperature=0.4,
            model_name='gpt-4o-mini',
            openai_api_key=openai_api_key
        )
        
        # Create vector database
        vectordb = Chroma.from_documents(
            documents=all_docs,
            embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
            persist_directory=persist_directory
        )
        vectordb.persist()
        
        # Initialize retriever
        retriever = vectordb.as_retriever(search_kwargs={'k': 10})
        
        # Initialize conversation memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        def create_chain(template_str: str) -> ConversationalRetrievalChain:
            # Create the document chain
            doc_prompt = PromptTemplate.from_template(template_str)
            doc_chain = StuffDocumentsChain(
                llm_chain=LLMChain(llm=llm, prompt=doc_prompt),
                document_variable_name="context"
            )
            
            # Create the question generator
            question_template = """
            Combine the chat history and follow up question into a standalone question.
            Chat History: {chat_history}
            Follow up question: {question}
            """
            question_prompt = PromptTemplate.from_template(question_template)
            question_generator = LLMChain(llm=llm, prompt=question_prompt)
            
            # Create the final chain
            return ConversationalRetrievalChain(
                combine_docs_chain=doc_chain,
                retriever=retriever,
                question_generator=question_generator,
                memory=memory,
            )
        
        # Create analysis and chat chains with their respective templates
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
        
        analysis_chain = create_chain(analysis_template)
        chat_chain = create_chain(chat_template)
        
        class ResumeAssistant:
            def __init__(self):
                self.initial_analysis_done = False
                self._persist_dir = persist_directory
            
            def chat(self, question: str) -> str:
                try:
                    if not self.initial_analysis_done:
                        result = analysis_chain({"question": "Analyze the resume match for this position"})
                        self.initial_analysis_done = True
                        return result['answer']
                    else:
                        result = chat_chain({"question": question})
                        return result['answer']
                except Exception as e:
                    return f"Error processing question: {str(e)}"
            
            def __del__(self):
                if hasattr(self, '_persist_dir') and os.path.exists(self._persist_dir):
                    shutil.rmtree(self._persist_dir)
        
        return ResumeAssistant()
        
    except Exception as e:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        raise ValueError(f"Error in resume analysis: {str(e)}")