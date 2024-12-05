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
        
        # Create analysis and chat chains with their respective templates COMMENT

        analysis_template = """
            You are an AI-powered Applicant Tracking System designed to help users optimize their resumes for specific job descriptions. Follow these instructions step by step and provide a single consolidated response:

            ### Step 1: Analyze the Job Description
            1. Analyze the job description in the context and extract the most important **hard skills** and **soft skills** required. 
            2. Format the result as a prioritized list of skills or keywords.

            ---

            ### Step 2: Analyze the Resume
            1. Analyze the resume content in the context and extract all relevant **skills, certifications, and experiences**.
            2. Compare these with the skills extracted from the job description and identify which **keywords** from the job description are **missing** in the resume.

            ---

            ### Step 3: Score the Resume
            Score the resume based on the following criteria:
            1. **STAR Format**: Evaluate if each bullet point in the resume follows the STAR (Situation, Task, Action, Result) format. Highlight bullet points that do not follow this format and provide suggestions to improve them.
            2. **Action Verbs**: Check if all bullet points start with action verbs. Highlight bullet points that do not and suggest corrections.
            3. **Structure**: Evaluate if the resume is well-structured into clear sections such as Education, Work Experience, and Skills. Suggest improvements if any sections are disorganized or missing.
            4. **Grammar and Sentence Formation**: Identify and correct any grammatical errors or sentence structure issues in the resume.

            Provide a **detailed breakdown of the score**, explaining where the resume excels and where it needs improvement.

            ---

            ### Step 4: Recommend Keyword Integration
            1. For each missing keyword identified in Step 2, suggest where it can be incorporated in the resume. Specify the **exact section** and **line number** (if possible).
            2. Draft new or revised bullet points that include the missing keywords. Ensure each revised bullet point:
            - Starts with an **action verb**.
            - Follows the **STAR format**.
            - Fits seamlessly into the relevant section of the resume.

            ---

            ### Step 5: Provide Consolidated Feedback
            Consolidate all the findings into a single, well-structured response. Include:
            1. Extracted skills and keywords from the job description.
            2. Missing keywords from the resume.
            3. The resume score with detailed feedback.
            4. Suggested revisions and newly drafted bullet points with incorporated keywords.

            Provide this response in a professional, clear, and actionable format so that it helps the user improve their resume effectively.

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