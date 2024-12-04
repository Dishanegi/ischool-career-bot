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
            temperature=0.2,
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
            temperature=0.2,
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

    def _prepare_interview_questions(self, position: str, job_description: str) -> List[str]:
        """Prepare a mixed list of questions with behavioral first, then technical"""
        
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
        
        # Must-ask behavioral questions first
        behavioral_questions = [
            "Tell me about yourself and your experience.",
            "What interests you about this position and our company?",
            "Describe a situation where you had to lead a team through a difficult project.",
            "Tell me about a challenging technical problem you've solved. What was your approach?"
        ]
        
        # Get role-specific technical questions
        role = position.lower().replace(" ", "_")
        technical_questions = []
        if role in question_pools:
            # Get both technical and system design questions
            technical_questions = (
                question_pools[role]["technical"][:3] +  # 3 technical questions
                question_pools[role]["system_design"][:2]  # 2 system design questions
            )
        
        # One strength/weakness question
        evaluation_question = [
            "What would you consider your technical strengths and areas for improvement?"
        ]
        
        # Closing question
        closing_question = [
            "Do you have any questions about the role or company?"
        ]
        
        # Combine in specific order
        final_questions = (
            behavioral_questions +  # First 4 questions are behavioral
            technical_questions +   # Then technical based on role
            evaluation_question +   # Then strengths/weaknesses
            closing_question       # End with closing question
        )
        
        return final_questions

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
        """Generate brutally honest, ultra-concise feedback"""
        resume_context = self.vectordb.similarity_search(question, k=3)
        resume_text = "\n".join([doc.page_content for doc in resume_context])
        
        # Prefix enforces the input format and rules
        prefix = f"""
        Question: {question}
        Answer: {answer}
        Resume: {resume_text}
        Job: {self.interview_state['job_description']}

        RULES:
        - Each point must show [resume proof] vs [answer given] vs [job need]
        - BRUTAL HONESTY: Call out any weak/missing points immediately
        - ULTRA CONCISE: Maximum 50 words per point
        - NO SOFT LANGUAGE: Use direct, harsh feedback
        - EVIDENCE ONLY: Only provable points from resume/answer
        """

        # Suffix enforces brutal feedback format
        suffix = """
        FORMAT RULES:
        - GOOD points: "Used [exact skill] for [specific need]"
        - MISS points: "Have [proven skill] but [failed requirement]"
        - Maximum 2 GOOD, 2 MISS
        - No explanation, no context, no advice
        - BRUTAL HONESTY REQUIRED
        """

        # Behavioral Questions
        if "tell me about yourself" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Used [role/skill from resume] for [exact job requirement]
            • Showed [past experience] solving [company's current need]

            MISS:
            • Have [key experience] but omitted [critical requirement]
            • Resume shows [achievement] but didn't mention impact

            {suffix}
            """

        elif "interests you" in question.lower() or "why this position" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Connected [resume skill] to [company's exact need]
            • Matched [project experience] with [current challenge]

            MISS:
            • Have [background] but ignored [tech stack match]
            • Resume shows [experience] but missed [role fit]

            {suffix}
            """

        elif "strengths" in question.lower() and "improvement" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Proved [strength] with [concrete project result]
            • Named [weakness] matching [non-critical skill]

            MISS:
            • Claimed [weak skill] despite [stronger resume evidence]
            • Admitted weakness in [critical job requirement]

            {suffix}
            """

        elif "difficult project" in question.lower() or "challenging project" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Used [specific project] proving [required skill]
            • Showed [technical win] solving [relevant challenge]

            MISS:
            • Ignored [stronger project] for [key requirement]
            • Skipped [measurable outcome] from resume

            {suffix}
            """

        # Technical Questions
        elif "design" in question.lower() or "architecture" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Applied [architecture experience] to [design need]
            • Used [technical skill] for [scaling requirement]

            MISS:
            • Have [design experience] but skipped [system requirement]
            • Resume shows [tech skill] but missed [design challenge]

            {suffix}
            """

        elif "technical problem" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Used [problem-solving method] for [technical need]
            • Applied [technical skill] to [specific challenge]

            MISS:
            • Have [relevant solution] but no methodology shown
            • Resume proves [skill] but impact not demonstrated

            {suffix}
            """

        elif "team" in question.lower() or "lead" in question.lower():
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Showed [leadership win] matching [team need]
            • Used [project example] proving [management skill]

            MISS:
            • Have [team experience] but skipped [leadership requirement]
            • Resume shows [management win] without metrics

            {suffix}
            """

        elif any(keyword in question.lower() for keyword in ["implement", "develop", "code", "programming", "testing"]):
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Applied [coding skill] to [implementation need]
            • Showed [technical ability] matching [job requirement]

            MISS:
            • Have [technical experience] but no practical example
            • Resume shows [dev skill] without methodology

            {suffix}
            """

        else:
            feedback_prompt = f"""
            {prefix}

            GOOD:
            • Used [technical skill] for [exact requirement]
            • Proved [experience] matching [job need]

            MISS:
            • Have [relevant skill] but didn't apply to [requirement]
            • Resume shows [capability] without proof

            {suffix}
            """

        result = self.chain({
            "question": feedback_prompt
        })

        # Get the response and clean it
        response = result['answer'].strip()
        
        # Ensure GOOD section exists
        if not response.startswith('GOOD:'):
            response = "GOOD:\n" + response
        
        # Only include MISS if it exists in the response
        if "MISS:" in response:
            # Split into sections and clean up
            parts = response.split("MISS:")
            good_section = parts[0].strip()
            miss_section = parts[1].strip()
            
            # Only include MISS section if it has actual content
            if miss_section and any(line.strip() for line in miss_section.split('\n')):
                return f"{good_section}\n\nMISS:\n{miss_section}"
            else:
                return good_section
        
        return response

    def _generate_closing_question_feedback(self, questions_asked: str) -> str:
        """Special feedback for questions asked by candidate about role/company"""
        resume_context = self.vectordb.similarity_search(self.interview_state['job_description'], k=3)
        resume_text = "\n".join([doc.page_content for doc in resume_context])
        
        feedback_prompt = f"""
        Based on:
        - Questions Asked: {questions_asked}
        - Resume: {resume_text}
        - Job Requirements: {self.interview_state['job_description']}

        Evaluate the candidate's questions about the role/company. Provide ultra-compact feedback in exactly this format:

        STRONG QUESTIONS:
        • Connected [specific resume skill/experience] to question about [specific job/project requirement]
        • Demonstrated knowledge of [company/project aspect] relevant to [specific job requirement]

        MISSING OPPORTUNITIES:
        • Could ask about [specific job/project requirement] based on [resume experience]
        • Should explore [specific technical aspect] given [resume background]

        Rules:
        - Max 2 strong points, max 2 missing points
        - Each point must connect questions to resume and job requirements
        - Focus on technical depth and project/role relevance
        - One line per point only
        """
        
        result = self.chain({"question": feedback_prompt})
        return result['answer']

    def _generate_final_feedback(self) -> str:
        resume_context = self.vectordb.similarity_search(
            self.interview_state['job_description'], 
            k=5
        )
        resume_text = "\n".join([doc.page_content for doc in resume_context])
        
        final_feedback_prompt = f"""
        Resume: {resume_text}
        Job Needs: {self.interview_state['job_description']}
        Answers: {str(self.interview_state['answers'])}

        One-line verdict on readiness.

        PROVED: (2 max)
        "Showed [X experience] for [Y requirement]"

        MISSED: (3 max)
        "Have [X] but showed [Y weakness]"

        FIX: (3 max)
        "Use [X experience] for [Y requirement]"

        Connect every point to resume + job needs. Brutal honesty.
        """
        
        result = self.chain({"question": final_feedback_prompt})
        return result['answer']

    def chat(self, input_data: Union[str, bytes], input_type: str = "text", output_type: str = "text") -> Tuple[str, Optional[str]]:
        """Process a message and handle both voice and text input/output"""
        try:
            input_text = self.process_input(input_data, input_type)
            response = ""
            print("\n=== Debug Info ===")
            print(f"Input received: {input_text[:100]}...")
            
            if isinstance(self.chain, ConversationalRetrievalChain):
                if input_text.lower().strip() in ['yes', 'y'] and not self.interview_state["in_progress"]:
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
                    print(f"Starting interview with question: {self.interview_state['questions'][0]}")
                
                elif self.interview_state["in_progress"]:
                    current_q = self.interview_state["questions"][self.interview_state["current_question"]]
                    print(f"\nCurrent question ({self.interview_state['current_question']}): {current_q}")
                    
                    # Store the answer
                    self.interview_state["answers"][current_q] = input_text
                    print(f"Stored answer: {input_text[:100]}...")
                    
                    # Generate feedback
                    try:
                        if current_q == "Do you have any questions about the role or company?":
                            print("Generating closing feedback...")
                            feedback = self._generate_closing_question_feedback(input_text)
                        else:
                            print("Generating standard feedback...")
                            feedback = self._generate_answer_feedback(current_q, input_text)
                        
                        print(f"\nGenerated feedback: {feedback}")
                        
                        if not feedback:
                            print("Warning: Empty feedback generated")
                            feedback = "GOOD:\n• Point not generated\n\nMISS:\n• Feedback generation failed"
                        
                        self.interview_state["feedback"].append(feedback)
                        
                        # Move to next question
                        self.interview_state["current_question"] += 1
                        print(f"Advanced to question {self.interview_state['current_question']}")
                        
                        # Format response
                        if self.interview_state["current_question"] < len(self.interview_state["questions"]):
                            next_q = self.interview_state["questions"][self.interview_state["current_question"]]
                            response = f"FEEDBACK:\n{feedback}\n\nNEXT QUESTION:\n{next_q}"
                            print(f"Prepared next question: {next_q}")
                        else:
                            print("Generating final feedback...")
                            # Clean up any existing audio files before generating final feedback
                            self.cleanup_audio_files()
                            final_feedback = self._generate_final_feedback()
                            response = f"Interview completed!\n\nFINAL FEEDBACK:\n{final_feedback}"
                            self.interview_state["in_progress"] = False
                    
                    except Exception as e:
                        print(f"Error generating feedback: {str(e)}")
                        response = "Error generating feedback. Please try again."
                        raise
                
                elif input_text.lower().strip() in ['no', 'n']:
                    response = self.chain({"question": "No problem. Feel free to ask specific questions."})['answer']
                else:
                    response = self.chain({"question": input_text})['answer']
            else:
                response = self.chain.predict(human_input=input_text)
            
            print(f"\nFinal response: {response[:100]}...")
            print("=== End Debug ===\n")
            
            # Clean up previous audio files before generating new response
            if output_type == "voice":
                self.cleanup_audio_files()
                
            return self.handle_response(response, output_type)
            
        except Exception as e:
            print(f"Error in chat method: {str(e)}")
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

    def cleanup_audio_files(self):
        """Clean up audio files from the audio_files directory"""
        try:
            if os.path.exists(self.audio_directory):
                for file in os.listdir(self.audio_directory):
                    if (file.startswith("audio_input_") or 
                        file.startswith("audio_response_")) and file.endswith(".mp3"):
                        file_path = os.path.join(self.audio_directory, file)
                        try:
                            os.remove(file_path)
                            print(f"Removed audio file: {file}")
                        except Exception as e:
                            print(f"Could not remove audio file {file}: {str(e)}")
        except Exception as e:
            print(f"Error during audio cleanup: {str(e)}")

    def cleanup(self):
        """Clean up temporary files and audio files"""
        # Clean up vector store directory
        if self.persist_directory and os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        
        # Clean up audio files
        self.cleanup_audio_files()

    def get_conversation_history(self) -> str:
        """Get the current conversation history"""
        return self.memory.load_memory_variables({})["chat_history"]

    def reset_conversation(self):
        """Reset the conversation history"""
        self.memory.clear()