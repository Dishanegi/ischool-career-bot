from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from pdf_reader import load_pdf
from splitter import split_text_documents
from langchain.docstore.document import Document
import streamlit as st

def get_cover_letter(job_description, pdf, openai_api_key):
    # Load resume
    pdf_doc = load_pdf(pdf)
    
    # Process job description text directly
    job_text = str(job_description)
    
    # Process resume
    if isinstance(pdf_doc, list):
        resume_text = "\n".join(str(doc) for doc in pdf_doc)
    else:
        resume_text = str(pdf_doc)
    
    # Create separate documents for job description and resume
    job_doc = Document(
        page_content=f"JOB DESCRIPTION:\n{job_text}",
        metadata={"source": "job_posting"}
    )
    
    resume_doc = Document(
        page_content=f"RESUME:\n{resume_text}",
        metadata={"source": "resume"}
    )
    
    # Split documents separately to maintain context
    split_docs = split_text_documents([job_doc, resume_doc])
    
    # Create vector database
    vectordb = Chroma.from_documents(
        split_docs,
        embedding=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )
    
    # Enhanced prompt template with stronger emphasis on job details
    template = """
    You are a professional cover letter writer. Using the provided context that contains both a job description and a resume, create a tailored cover letter.

    STEP 1 - ANALYZE JOB DESCRIPTION:
    Before writing, carefully extract and list out:
    - Company Name: [Extract from job description]
    - Job Title: [Extract from job description]
    - Location: [Extract if available]
    - Department/Team: [Extract if available]
    - Key Requirements: [List 3-4 main requirements]
    - Company Values/Culture: [Note any mentioned]
    
    STEP 2 - ANALYZE RESUME:
    Identify the candidate's:
    - Name and Contact Details
    - Most relevant skills matching job requirements
    - Key achievements that align with the role
    
    STEP 3 - CREATE COVER LETTER:
    
    [Current Date]
    
    [Company Name]
    [Company Location if available]
    
    Dear [Hiring Manager/Appropriate Salutation],
    
    OPENING PARAGRAPH:
    - Mention specific job title and company name
    - Show knowledge of company (use details from job description)
    - State how you learned of the position
    - Brief overview of why you're an excellent fit
    
    BODY PARAGRAPHS:
    - Take 2-3 key requirements from the job description
    - For each, provide specific evidence from resume showing how you meet it
    - Use numbers and concrete examples
    - Mirror language from the job description
    
    CLOSING:
    - Restate enthusiasm for the role and company
    - Request interview
    - Thank them
    - Provide contact information
    
    Sincerely,
    [Name from Resume]
    [Contact Info from Resume]

    Note: Focus on specificity - use exact company name, job title, and requirements from the posting. Make sure to incorporate concrete examples from the resume that directly address the job requirements.
    
    Context: {context}
    """
    
    COVER_LETTER_PROMPT = PromptTemplate(
        input_variables=["context"],
        template=template
    )
    
    # Set up retrieval chain with higher k value
    pdf_qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            temperature=0.7, 
            model_name='gpt-4o-mini',  # Changed to gpt-3.5-turbo as gpt-4o-mini might not be available
            openai_api_key=openai_api_key
        ),
        chain_type="stuff",
        retriever=vectordb.as_retriever(
            search_kwargs={'k': 10}  # Increased k for more context
        ),
        chain_type_kwargs={
            "prompt": COVER_LETTER_PROMPT
        }
    )
    
    # Enhanced query focusing on job details first
    query = """
    First, carefully analyze the job description to extract company name, position, and key requirements.
    Then, match the resume details to these requirements.
    Finally, create a detailed cover letter that demonstrates deep understanding of both the role and the company.
    """
    
    result = pdf_qa.run(query)
    
    return result