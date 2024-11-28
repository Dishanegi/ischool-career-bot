from openai import OpenAI
import os
import base64
import time
import tempfile
import shutil
from typing import List, Dict, Any, Tuple, Optional, Union
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from pdf_reader import load_pdf

import sys

# SQLite workaround for Streamlit Cloud
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_community.vectorstores import Chroma

class VoiceAssistant:
    def __init__(self, api_key: str):
        """Initialize the voice assistant with OpenAI and LangChain components"""
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.chain, self.memory = self._create_langchain_agent()
        self.vectordb = None
        self.persist_directory = None
        # Create audio_files directory if it doesn't exist
        self.audio_directory = "audio_files"
        os.makedirs(self.audio_directory, exist_ok=True)
        self.interview_state = {
            "in_progress": False,
            "current_question": 0,
            "position": "",
            "job_description": "",
            "questions": [],
            "answers": {},
            "feedback": []
        }
        
    def process_input(self, input_data: Union[str, bytes], input_type: str = "text") -> str:
        """Process either text or voice input and return the text response"""
        if input_type == "voice" and isinstance(input_data, bytes):
            # Save voice input to audio_files directory
            temp_audio = os.path.join(self.audio_directory, f"audio_input_{int(time.time())}.mp3")
            with open(temp_audio, "wb") as f:
                f.write(input_data)
            
            try:
                # Transcribe voice to text
                input_text = self.transcribe(temp_audio)
                os.remove(temp_audio)  # Clean up temp file
                return input_text
            except Exception as e:
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)
                raise Exception(f"Error processing voice input: {str(e)}")
        
        return input_data 
    
    def handle_response(self, response: str, output_type: str = "text") -> Tuple[str, Optional[str]]:
        """Handle the response in either text or voice format"""
        if output_type == "voice":
            audio_file = os.path.join(self.audio_directory, f"audio_response_{int(time.time())}.mp3")
            try:
                self._text_to_audio(response, audio_file)
                return response, audio_file
            except Exception as e:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                raise Exception(f"Error generating voice response: {str(e)}")
        
        return response, None

    def _create_langchain_agent(self) -> Tuple[LLMChain, ConversationBufferMemory]:
        """Create and configure the LangChain conversation agent"""
        llm = ChatOpenAI(
            temperature=0.4,
            model_name='gpt-4o-mini',
            openai_api_key=self.api_key
        )
        
        memory = ConversationBufferMemory(
            return_messages=True,
            input_key="human_input",
            output_key="output"
        )
        
        interview_template = """
        You are an expert interview coach and career advisor. Your role is to help candidates prepare for interviews by:
        
        1. During mock interviews:
           - Ask relevant technical and behavioral questions
           - Provide feedback on responses
           - Suggest improvements for answer structure
           - Help develop better examples
        
        2. For general preparation:
           - Offer interview best practices
           - Help structure responses
           - Provide industry-specific advice
           - Share common interview questions and strategies
        
        Keep responses clear and structured, as they may be converted to speech.
        Be encouraging but professional, offering specific and actionable advice.
        
        Previous conversation:
        {chat_history}
        
        Human: {human_input}
        Assistant:"""
        
        conversation_prompt = PromptTemplate(
            input_variables=["chat_history", "human_input"],
            template=interview_template
        )
        
        conversation_chain = LLMChain(
            llm=llm,
            prompt=conversation_prompt,
            memory=memory,
            verbose=True
        )
        
        return conversation_chain, memory

    def _create_interview_agent(self, vectordb) -> Tuple[ConversationalRetrievalChain, ConversationBufferMemory]:
        """Create specialized interview chain with document context"""
        llm = ChatOpenAI(
            temperature=0.4,
            model_name='gpt-4',
            openai_api_key=self.api_key
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )
        
        doc_template = """
        You are an expert interview coach analyzing the candidate's resume and job requirements.
        Always address the candidate as "you" and never use their name.
        Use the context to provide specific, actionable advice for interview preparation.
        
        Previous conversation:
        {chat_history}
        
        Context: {context}
        Question: {question}
        Assistant:"""
        
        retriever = vectordb.as_retriever(search_kwargs={'k': 10})
        
        doc_prompt = PromptTemplate(
            template=doc_template,
            input_variables=["chat_history", "context", "question"]
        )
        
        doc_chain = StuffDocumentsChain(
            llm_chain=LLMChain(llm=llm, prompt=doc_prompt),
            document_variable_name="context"
        )
        
        question_prompt = PromptTemplate(
            template="Given the chat history and question, generate a standalone question:\nChat History: {chat_history}\nFollow up question: {question}",
            input_variables=["chat_history", "question"]
        )
        
        return ConversationalRetrievalChain(
            combine_docs_chain=doc_chain,
            retriever=retriever,
            question_generator=LLMChain(llm=llm, prompt=question_prompt),
            memory=memory,
            return_source_documents=False
        ), memory
    
    def initialize_interview_prep(self, resume_file, job_description: str) -> str:
        """Initialize interview preparation with resume and job description"""
        self.persist_directory = tempfile.mkdtemp()
        
        try:
            # Extract job position from description
            position = job_description.split('\n')[0].strip()
            
            # Store in interview state
            self.interview_state["position"] = position
            self.interview_state["job_description"] = job_description
            
            # Process documents
            resume_docs = load_pdf(resume_file)
            job_doc = Document(
                page_content=f"JOB DESCRIPTION:\n{job_description.strip()}",
                metadata={"source": "job_posting"}
            )
            all_docs = [job_doc] + (resume_docs if isinstance(resume_docs, list) else [resume_docs])
            
            # Create vector database
            self.vectordb = Chroma.from_documents(
                documents=all_docs,
                embedding=OpenAIEmbeddings(openai_api_key=self.api_key),
                persist_directory=self.persist_directory
            )
            self.vectordb.persist()
            
            # Create specialized interview agent
            self.chain, self.memory = self._create_interview_agent(self.vectordb)
            
            # Initial analysis prompt
            analysis_prompt = f"""
            Analyze this resume for the {position} position and provide:
            1. A brief overview of the key matching points
            2. Areas that align well with the job requirements
            3. Any potential gaps or areas to focus on
            
            After the analysis, end with:
            'I can help you prepare for the {position} interview with some practice questions. Would you like to start the mock interview? Say yes or no.'
            """
            
            result = self.chain({
                "question": analysis_prompt
            })
            
            return result['answer']
            
        except Exception as e:
            if self.persist_directory and os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            raise Exception(f"Error in interview preparation: {str(e)}")
    
    def _generate_technical_questions(self, job_description: str) -> List[str]:
        """Generate technical questions based on job description requirements"""
        example_questions = {
            "data_analysis": [
                "Can you explain how you would use pandas to handle missing data in a large dataset?",
                "Walk me through your process of creating a dashboard using PowerBI/Tableau.",
                "How would you approach A/B testing for a new feature?"
            ],
            "software_development": [
                "Explain how you would optimize a slow-performing SQL query.",
                "How would you implement error handling in a REST API?",
                "Describe your experience with containerization using Docker."
            ],
            "machine_learning": [
                "How do you handle imbalanced datasets?",
                "Explain your approach to feature selection.",
                "How do you validate your machine learning models?"
            ]
        }

        prompt = f"""
        Here are some example technical interview questions for different roles:
        {str(example_questions)}

        Using these as examples, generate 3 technical questions based on these requirements from the job description:
        {job_description}

        The questions should:
        1. Focus on technical skills specifically mentioned in the job description
        2. Be similar in style to the example questions
        3. Test both theoretical knowledge and practical experience
        4. Be specific to the required technologies/tools

        Format: Return just the questions, one per line.
        """
        
        result = self.chain({"question": prompt})
        return result['answer'].strip().split('\n')[:3]

    def _prepare_interview_questions(self, position: str, job_description: str) -> List[str]:
        """Prepare the complete list of interview questions"""
        behavioral_questions = [
            "Tell me about yourself and your experience.",
            "Why are you interested in this position?",
            "What makes you the best candidate for this role?"
        ]

        technical_questions = self._generate_technical_questions(job_description)
        closing_question = ["Do you have any questions for us?"]

        return behavioral_questions + technical_questions + closing_question

    def _get_relevant_context(self, question: str, answer: str) -> Dict[str, str]:
        """Get highly relevant context from resume and job description"""
        # Create specific queries based on question type
        queries = {
            "technical": f"""
                Find technical skills and experience related to:
                Question: {question}
                Answer topics: {answer[:100]}
            """,
            "experience": f"""
                Find relevant work experience and achievements related to:
                Question: {question}
                Answer topics: {answer[:100]}
            """,
            "requirements": f"""
                Find specific job requirements related to:
                Question: {question}
                Answer topics: {answer[:100]}
            """
        }
        
        context = {}
        for query_type, query in queries.items():
            results = self.vectordb.similarity_search(
                query,
                k=3  # Get top 3 most relevant chunks
            )
            context[query_type] = "\n".join([doc.page_content for doc in results])
        
        return context
        
        context = {}
        for query_type, query in queries.items():
            results = self.vectordb.similarity_search(
                query,
                k=2,  # Get top 2 most relevant chunks
                fetch_k=4  # Consider top 4 before selecting
            )
            context[query_type] = "\n".join([doc.page_content for doc in results])
        
        return context

    def _generate_answer_feedback(self, question: str, answer: str) -> str:
        """Generate brutally honest feedback for an interview answer"""
        # Get targeted context for this specific Q&A
        context = self._get_relevant_context(question, answer)
        
        feedback_prompt = f"""
        You are a brutally honest interview coach. Analyze this response based on concrete evidence from 
        the candidate's background and job requirements.

        TECHNICAL BACKGROUND FROM RESUME:
        {context['technical']}

        RELEVANT EXPERIENCE:
        {context['experience']}

        JOB REQUIREMENTS:
        {context['requirements']}

        QUESTION: {question}
        CANDIDATE'S ANSWER: {answer}

        Provide ruthlessly honest feedback focused on:

        1. Factual Accuracy:
        - Compare claimed technical knowledge against their actual background
        - Identify any misalignment between stated experience and resume
        - Point out any technical inaccuracies or oversimplifications

        2. Requirements Alignment:
        - How well does their answer match the job's specific needs?
        - Are they missing crucial requirements mentioned in the job description?
        - Are they emphasizing irrelevant skills or experiences?

        3. Evidence-Based Critique:
        - What claims in their answer are NOT supported by their resume?
        - Which required skills are they failing to demonstrate?
        - Where are they overselling vs. underselling their actual experience?

        4. Specific Deficiencies:
        - List exact points where the answer falls short
        - Identify specific missing technical details they should know
        - Point out any vague or evasive parts of their response

        Format as:
        CLAIMS VS REALITY: (compare answer claims against actual background)
        CRITICAL GAPS: (specific missing elements from job requirements)
        REQUIRED IMPROVEMENTS: (exact changes needed with examples)

        Be merciless with accuracy but back all criticism with specific evidence from their background or the job requirements.
        """
        
        result = self.chain({"question": feedback_prompt})
        return result['answer']

    def _generate_final_feedback(self) -> str:
        """Generate comprehensive brutally honest final feedback"""
        # Get complete context from all documents
        all_context = self._get_relevant_context(
            "overall performance",
            str(self.interview_state['answers'])
        )
        
        final_feedback_prompt = f"""
        You are a brutally honest interview coach providing final feedback based on concrete evidence.

        CANDIDATE'S TECHNICAL BACKGROUND:
        {all_context['technical']}

        CANDIDATE'S EXPERIENCE:
        {all_context['experience']}

        JOB REQUIREMENTS:
        {all_context['requirements']}

        FULL INTERVIEW PERFORMANCE:
        Questions and Answers:
        {str(self.interview_state['answers'])}

        Previous Feedback:
        {str(self.interview_state['feedback'])}

        Provide a ruthlessly honest final evaluation:

        1. Requirements Gap Analysis:
        - List each key job requirement and rate their demonstrated competency
        - Identify critical missing skills or experiences
        - Compare their level against expected seniority

        2. Technical Competency Evaluation:
        - Analyze depth of technical knowledge against claimed expertise
        - List specific technical areas where they fell short
        - Compare technical skills against job requirements

        3. Experience Verification:
        - Cross-reference interview claims against actual resume experience
        - Identify any credibility issues or exaggerations
        - Assess relevance of their experience to this role

        4. Critical Issues and Risks:
        - List potential red flags for hiring managers
        - Identify gaps that could prevent success in role
        - Point out any concerning patterns in responses

        5. Required Preparation Plan:
        - Prioritize specific areas requiring immediate improvement
        - List exact topics they need to study/practice
        - Identify what experience they need to gain

        Conclude with:
        1. Clear hiring readiness assessment
        2. Estimate of time needed to become fully qualified
        3. Direct statement about pursuing this role now vs. later

        Base all feedback on specific evidence from their resume, answers, and job requirements.
        Be brutally honest about their current readiness for this role.
        """
        
        result = self.chain({"question": final_feedback_prompt})
        return result['answer']

    def chat(self, input_data: Union[str, bytes], input_type: str = "text", output_type: str = "text") -> Tuple[str, Optional[str]]:
        """Process a message and handle both voice and text input/output"""
        try:
            # Process input (voice or text)
            input_text = self.process_input(input_data, input_type)
            response = ""
            
            if isinstance(self.chain, ConversationalRetrievalChain):
                if input_text.lower().strip() in ['yes', 'y'] and not self.interview_state["in_progress"]:
                    # Start mock interview
                    self.interview_state["in_progress"] = True
                    self.interview_state["current_question"] = 0
                    self.interview_state["questions"] = self._prepare_interview_questions(
                        self.interview_state["position"],
                        self.interview_state["job_description"]
                    )
                    
                    response = (
                        "Great! Let's begin the mock interview. I'll ask you questions one by one. "
                        "Take your time to answer each question thoroughly.\n\n"
                        "First question: " + self.interview_state["questions"][0]
                    )
                
                elif self.interview_state["in_progress"]:
                    # Store the answer
                    current_q = self.interview_state["questions"][self.interview_state["current_question"]]
                    self.interview_state["answers"][current_q] = input_text
                    
                    # Get feedback
                    feedback = self._generate_answer_feedback(current_q, input_text)
                    self.interview_state["feedback"].append(feedback)
                    
                    # Move to next question or conclude
                    self.interview_state["current_question"] += 1
                    if self.interview_state["current_question"] < len(self.interview_state["questions"]):
                        response = (
                            f"{feedback}\n\n"
                            f"Next question: {self.interview_state['questions'][self.interview_state['current_question']]}"
                        )
                    else:
                        final_feedback = self._generate_final_feedback()
                        response = f"Interview completed! Here's your comprehensive feedback:\n\n{final_feedback}"
                        self.interview_state["in_progress"] = False
                
                elif input_text.lower().strip() in ['no', 'n']:
                    response = self.chain({
                        "question": "No problem. Feel free to ask any specific questions, or let me know when you're ready to practice interviewing."
                    })['answer']
                else:
                    response = self.chain({"question": input_text})['answer']
            else:
                response = self.chain.predict(human_input=input_text)
            
            # Handle output (voice or text)
            return self.handle_response(response, output_type)
            
        except Exception as e:
            raise Exception(f"Error processing message: {str(e)}")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text"""
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return transcript.text
        except Exception as e:
            raise Exception(f"Error transcribing audio: {str(e)}")

    def _text_to_audio(self, text: str, audio_path: str):
        """Convert text to speech and save to file"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=text
            )
            response.stream_to_file(audio_path)
        except Exception as e:
            raise Exception(f"Error converting text to speech: {str(e)}")

    def get_base64_audio(self, audio_file: str) -> Optional[str]:
        """Convert audio file to base64 encoding"""
        if os.path.exists(audio_file):
            try:
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                return base64.b64encode(audio_bytes).decode("utf-8")
            except Exception as e:
                raise Exception(f"Error encoding audio to base64: {str(e)}")
        return None

    def cleanup(self):
        """Clean up temporary files"""
        if self.persist_directory and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            
        try:
            # Clean up audio files from the audio_files directory
            if os.path.exists(self.audio_directory):
                for file in os.listdir(self.audio_directory):
                    if (file.startswith("audio_input_") or 
                        file.startswith("audio_response_")) and file.endswith(".mp3"):
                        try:
                            os.remove(os.path.join(self.audio_directory, file))
                        except Exception as e:
                            print(f"Could not remove audio file {file}: {str(e)}")
        except Exception as e:
            print(f"Error during audio cleanup: {str(e)}")

    def get_conversation_history(self) -> str:
        """Get the current conversation history"""
        return self.memory.load_memory_variables({})["chat_history"]

    def reset_conversation(self):
        """Reset the conversation history"""
        self.memory.clear()