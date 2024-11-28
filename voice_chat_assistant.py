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
            model_name='gpt-4o-mini',
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
    
    def _get_role_specific_questions(self, role: str) -> List[str]:
        """Get role-specific questions from the question pool"""
        
        # Question pools for different roles
        question_pools = {
            "software_engineer": {
                "technical": [
                    "Can you explain how you would design a scalable microservices architecture?",
                    "Walk me through your experience with CI/CD pipelines and deployment automation.",
                    "How do you ensure code quality and what testing strategies do you implement?",
                    "Describe a challenging performance issue you encountered and how you resolved it.",
                    "How do you approach database design and optimization?"
                ],
                "system_design": [
                    "How would you design a real-time notification system?",
                    "Explain how you would design a URL shortening service.",
                    "How would you architect a distributed caching system?"
                ]
            },
            "cloud_engineer": {
                "technical": [
                    "Describe your experience with AWS/Azure/GCP service implementation.",
                    "How do you handle cloud infrastructure security and compliance?",
                    "Explain your approach to cloud cost optimization.",
                    "How do you implement disaster recovery in cloud environments?",
                    "Walk me through your experience with Infrastructure as Code."
                ],
                "system_design": [
                    "How would you design a multi-region cloud architecture?",
                    "Explain your approach to implementing auto-scaling.",
                    "How would you design a zero-downtime deployment system?"
                ]
            },
            "data_engineer": {
                "technical": [
                    "How do you design and optimize data pipelines?",
                    "Explain your experience with real-time data processing.",
                    "How do you handle data quality and validation?",
                    "Describe your experience with data warehouse design.",
                    "How do you approach data migration projects?"
                ],
                "system_design": [
                    "How would you design a real-time analytics system?",
                    "Explain your approach to building a data lake.",
                    "How would you implement a data quality monitoring system?"
                ]
            }
        }

        # Common questions for all roles
        common_questions = {
            "leadership": [
                "Describe a situation where you had to lead a team through a difficult project.",
                "How do you handle conflicts within your team?",
                "Tell me about a time when you had to influence stakeholders without direct authority.",
                "How do you approach mentoring junior team members?",
                "Describe how you've helped improve team processes or practices."
            ],
            "behavioral": [
                "Tell me about yourself and your experience.",
                "What interests you about this position and our company?",
                "Where do you see yourself in 5 years?",
                "What are your greatest professional achievements?",
                "How do you handle pressure and tight deadlines?"
            ],
            "technical_challenges": [
                "Describe the most challenging technical problem you've solved. What was your approach?",
                "Tell me about a time when you had to make a difficult technical decision. How did you approach it?",
                "Share an example of a project that failed. What did you learn from it?",
                "How do you stay updated with the latest technologies in your field?"
            ],
            "strengths_weaknesses": [
                "What would you consider your technical strengths? How have you applied them?",
                "What areas of your technical expertise do you feel need improvement?",
                "How do you overcome technical limitations or knowledge gaps?",
                "What's your approach to learning new technologies?"
            ],
            "closing": [
                "Do you have any questions about the role or company?",
                "What are your salary expectations?",
                "When would you be available to start?"
            ]
        }

        # Get role-specific questions
        role = role.lower().replace(" ", "_")
        role_questions = []
        if role in question_pools:
            role_questions.extend(question_pools[role]["technical"])
            role_questions.extend(question_pools[role]["system_design"])

        # Combine with common questions
        selected_questions = (
            common_questions["behavioral"][:2] +
            common_questions["leadership"][:2] +
            role_questions[:3] +
            common_questions["technical_challenges"][:2] +
            common_questions["strengths_weaknesses"][:2] +
            common_questions["closing"][:1]
        )

        return selected_questions

    def _prepare_interview_questions(self, position: str, job_description: str) -> List[str]:
        """Prepare the complete list of interview questions based on the position"""
        return self._get_role_specific_questions(position)

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


    def _generate_answer_feedback(self, question: str, answer: str) -> str:
        """Generate feedback for an interview answer"""
        feedback_prompt = f"""
        Analyze this interview response and provide constructive feedback:

        Question: {question}
        Answer: {answer}

        Provide feedback on:
        1. Content relevance and completeness
        2. Structure and clarity of response
        3. Technical accuracy (if technical question)
        4. Specific improvement suggestions

        Keep feedback professional and actionable. Be honest but encouraging.
        """
        
        result = self.chain({"question": feedback_prompt})
        return result['answer']

    def _generate_final_feedback(self) -> str:
        """Generate comprehensive feedback after all questions"""
        final_feedback_prompt = f"""
        Review the entire interview performance:

        Questions and Answers:
        {str(self.interview_state['answers'])}

        Individual Feedback:
        {str(self.interview_state['feedback'])}

        Provide a comprehensive evaluation including:
        1. Overall interview performance
        2. Key strengths demonstrated
        3. Specific areas needing improvement
        4. Technical competency assessment
        5. Communication style feedback
        6. Actionable recommendations for future interviews

        Be direct and specific in your feedback, highlighting both positives and areas for growth.
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