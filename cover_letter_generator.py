from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.utilities import SerpAPIWrapper
from pdf_reader import load_pdf
from splitter import split_text_documents
from langchain_core.documents import Document
import sys
import streamlit as st

# SQLite workaround for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma

def extract_company_name(job_description):
    """Extract company name from job description using LLM"""
    print("\n[DEBUG] Starting company name extraction...")
    
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", api_key=st.secrets["openai_api_key"])
        print("[DEBUG] LLM initialized for company name extraction")
        
        response = llm.invoke(f"""
        From the following job description, extract the exact company name:

        {job_description}

        Provide ONLY the company name with no additional text.
        """)
        
        company_name = response.content.strip()
        print(f"[DEBUG] Extracted company name: {company_name}")
        return company_name
        
    except Exception as e:
        print(f"[ERROR] Error in extract_company_name: {str(e)}")
        raise

def get_company_vision_mission(company_name):
    """Retrieve company vision and mission using search"""
    print(f"\n[DEBUG] Starting vision/mission search for {company_name}...")
    
    try:
        print("[DEBUG] Initializing SerpAPI")
        search = SerpAPIWrapper(serpapi_api_key=st.secrets["serp_api_key"])
        
        print("[DEBUG] Executing search query")
        search_results = search.run(f"{company_name} company mission vision values")
        print("[DEBUG] Search results received")
        
        print("[DEBUG] Initializing LLM for summary")
        llm = ChatOpenAI(
            temperature=0, 
            model_name="gpt-4o-mini",
            api_key=st.secrets["openai_api_key"]
        )
        
        print("[DEBUG] Generating summary")
        summary = llm.invoke(f"""
        Based on this search information about {company_name}, provide a concise summary of their mission, vision, and values:
        {search_results}
        """)
        
        print("[DEBUG] Summary generated successfully")
        return summary.content
        
    except Exception as e:
        print(f"[ERROR] Error in get_company_vision_mission: {str(e)}")
        return f"Company vision and mission could not be retrieved. Error: {str(e)}"

def get_cover_letter(job_description, pdf, openai_api_key):
    """Generate cover letter based on job description and resume"""
    print("\n[DEBUG] Starting cover letter generation...")
    
    try:
        # Extract company information
        print("[DEBUG] Extracting company information")
        company_name = extract_company_name(job_description)
        company_vision_mission = get_company_vision_mission(company_name)
        
        # Process resume
        print("[DEBUG] Processing resume")
        pdf_doc = load_pdf(pdf)
        resume_text = "\n".join(str(doc) for doc in pdf_doc) if isinstance(pdf_doc, list) else str(pdf_doc)
        print("[DEBUG] Resume processed successfully")
        
        # Create documents
        print("[DEBUG] Creating documents")
        job_doc = Document(
            page_content=f"JOB DESCRIPTION:\n{str(job_description)}",
            metadata={"source": "job_posting"}
        )
        
        resume_doc = Document(
            page_content=f"RESUME:\n{resume_text}",
            metadata={"source": "resume"}
        )
        
        # Split documents and create vector database
        print("[DEBUG] Creating vector database")
        split_docs = split_text_documents([job_doc, resume_doc])
        vectordb = Chroma.from_documents(
            split_docs,
            embedding=OpenAIEmbeddings(api_key=openai_api_key)
        )
        print("[DEBUG] Vector database created")
        
        # Create retriever
        print("[DEBUG] Setting up retriever")
        retriever = vectordb.as_retriever(search_kwargs={'k': 10})
        
        # Create template and prompt
        print("[DEBUG] Creating prompt template")
        template = f"""
        You are a professional cover letter writer. Using the provided context that contains both a job description and a resume, create a tailored cover letter.

        COMPANY INFORMATION:
        - Company Name: {company_name}
        - Company Purpose:
        {company_vision_mission}

        STEP 1 - ANALYZE JOB DESCRIPTION:
        Before writing, carefully extract and list out:
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
        
        {company_name}
        [Company Location if available]
        
        Dear [Hiring Manager/Appropriate Salutation],
        
        OPENING PARAGRAPH:
        - Mention specific job title and company name
        - Demonstrate knowledge of company's vision and mission
        - Show why you're aligned with company's goals
        - State how you learned of the position
        - Brief overview of why you're an excellent fit
        
        BODY PARAGRAPHS:
        - Take 2-3 key requirements from the job description
        - For each, provide specific evidence from resume showing how you meet it
        - Use numbers and concrete examples
        - Mirror language from the job description
        - Connect your achievements to company's vision
        
        CLOSING:
        - Restate enthusiasm for the role and how you can contribute to company's mission
        - Request interview
        - Thank them
        - Provide contact information
        
        Sincerely,
        [Name from Resume]
        [Contact Info from Resume]

        Note: Focus on specificity - use exact company name, job title, and requirements from the posting.

        Retrieved information: {{context}}

        Human: Please write a cover letter based on the above information.

        Assistant: Let me write a professional cover letter incorporating all the details provided.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        print("[DEBUG] Prompt template created")
        
        # Create LLM and chain
        print("[DEBUG] Initializing LLM and chain")
        llm = ChatOpenAI(
            temperature=0.7,
            model_name='gpt-4o-mini',
            api_key=st.secrets["openai_api_key"]
        )
        
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("[DEBUG] Chain created successfully")
        
        # Generate cover letter
        print("[DEBUG] Generating final cover letter")
        result = chain.invoke("Generate a detailed cover letter that demonstrates understanding of both the role and company.")
        print("[DEBUG] Cover letter generated successfully")
        
        return result
        
    except Exception as e:
        print(f"[ERROR] Error in get_cover_letter: {str(e)}")
        raise