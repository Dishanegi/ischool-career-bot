import streamlit as st
from cover_letter_generator import get_cover_letter
from resume_generator import get_resume
from enum import Enum
from typing import Callable, Tuple
import time
import pathlib
from audio_recorder_streamlit import audio_recorder
from voice_chat_assistant import VoiceAssistant
from collections import deque
import os

class PageType(Enum):
    HOME = "Home"
    COVER_LETTER = "Cover Letter Generator"
    RESUME = "Resume Generator"
    VOICE_CHAT = "Voice Chat Assistant" 

def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.set_page_config(
    page_title="CareerForge AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
css_path = pathlib.Path(__file__).parent / "styles.css"
load_css(css_path)

openai_api_key = st.secrets["openai_api_key"]

def get_page_config() -> dict:
    return {
        PageType.COVER_LETTER: {
            "title": "Cover Letter Generator",
            "form_id": "cover_letter_form",
            "desc_label": "Enter Job Description",
            "file_label": "Upload your CV",
            "submit_label": "Generate Cover Letter ✨",
            "generator_func": get_cover_letter
        },
        PageType.RESUME: {
            "title": "Resume Generator",
            "form_id": "resume_form",
            "desc_label": "Enter Target Job Description",
            "file_label": "Upload your Current Resume",
            "submit_label": "Generate Tailored Resume ✨",
            "generator_func": get_resume
        }
    }

def render_home_page():
    """Render the main landing page."""
    # Hero Section with enhanced title container
    st.markdown("""
        <div class="title-container">
            <h1 class="main-title">SU iBot</h1>
            <p class="subtitle">Forge Your Future with AI-Powered Career Tools</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Add spacing
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)

    # Stats Section
    st.markdown("""
        <h2 class="section-header">Our Impact</h2>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">90%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">10K+</div>
                <div class="stat-label">Documents Generated</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">24/7</div>
                <div class="stat-label">AI Assistance</div>
            </div>
        """, unsafe_allow_html=True)

    # Features Section
    st.markdown("""
        <h2 class="section-header">🚀 Our Tools</h2>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <h3 class="feature-title">📝 AI Cover Letter Generator</h3>
                <p style="font-family: 'Poppins', sans-serif;">Create compelling, personalized cover letters that highlight your unique value proposition. Our AI analyzes job descriptions to craft perfect matches.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h3 class="feature-title">📄 Smart Resume Optimizer</h3>
                <p style="font-family: 'Poppins', sans-serif;">Transform your resume with AI-powered optimization. Get tailored suggestions and formatting that align with industry standards.</p>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <h3 class="feature-title">🎙️ Voice Chat Interview Bot</h3>
                <p style="font-family: 'Poppins', sans-serif;">Practice your interview skills with our AI-powered voice chat assistant. Get real-time feedback and improve your interview confidence.</p>
            </div>
        """, unsafe_allow_html=True)

    # How It Works Section
    st.markdown("""
        <h2 class="section-header">🔍 How It Works</h2>
    """, unsafe_allow_html=True)
    
    steps = {
        "1️⃣ Upload": "Share your existing resume or job description",
        "2️⃣ Analyze": "Our AI analyzes your content and requirements",
        "3️⃣ Generate": "Receive tailored documents within seconds",
        "4️⃣ Review": "Make final adjustments and download"
    }
    
    for step, description in steps.items():
        st.markdown(f"""
            <div class="step-card feature-card">
                <h3 class="feature-title">{step}</h3>
                <p style="font-family: 'Poppins', sans-serif;">{description}</p>
            </div>
        """, unsafe_allow_html=True)

def check_inputs(api_key: str, description: str, file) -> Tuple[bool, str]:
    if not api_key:
        return False, 'OpenAI API key is missing in secrets.toml!'
    if not description:
        return False, 'Please enter a job description!'
    if not file:
        return False, 'Please upload your file!'
    return True, ''

def generate_document(generator_func: Callable, description: str, file, api_key: str):
    # Only used for Cover Letter Generator now
    if generator_func.__name__ == 'get_cover_letter':
        progress_text = "Operation in progress. Please wait..."
        my_bar = st.progress(0, text=progress_text)
        
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        with st.spinner('Finalizing your document...'):
            output = generator_func(description, file, api_key)
            st.balloons()
            st.success('Document generated successfully!')
            st.write(output)

def render_generator_page(page_type: PageType):
    config = get_page_config()[page_type]
    
    # Different descriptions for each page type
    descriptions = {
        PageType.COVER_LETTER: "Create a compelling cover letter tailored to your target job description",
        PageType.RESUME: "Optimize your resume to match the job requirements and stand out from the crowd"
    }
    
    # Add timestamp to force animation refresh
    timestamp = int(time.time() * 1000)
    
    st.markdown(f"""
        <div class="page-container animation-{timestamp}">
            <h1 class="page-title animated-title animation-{timestamp}">{config["title"]}</h1>
            <p class="page-subtitle animation-{timestamp}">{descriptions[page_type]}</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for chat
    if 'assistant' not in st.session_state:
        st.session_state.assistant = None
        st.session_state.analysis_done = False
        st.session_state.file_processed = False
        st.session_state.chat_history = []
    
    # Different handling for Resume Generator
    if page_type == PageType.RESUME:
        with st.form(config["form_id"]):
            description = st.text_area(
                config["desc_label"],
                height=200,
                placeholder="Paste the job description here..."
            )
            
            file = st.file_uploader(
                config["file_label"],
                type=["pdf"],
                accept_multiple_files=False
            )
            
            submitted = st.form_submit_button(
                config["submit_label"],
                use_container_width=True
            )
            
            if submitted:
                is_valid, error_message = check_inputs(openai_api_key, description, file)
                if not is_valid:
                    st.error(error_message, icon='⚠')
                else:
                    # Create a new assistant instance if not already created
                    if st.session_state.assistant is None and not st.session_state.file_processed:
                        try:
                            assistant = get_resume(description, file, openai_api_key)
                            st.session_state.assistant = assistant
                            st.session_state.file_processed = True
                            
                            # Get initial analysis
                            if not st.session_state.analysis_done:
                                with st.spinner('Analyzing resume...'):
                                    initial_response = assistant.chat("Start")
                                    st.session_state.chat_history.append(("assistant", initial_response))
                                    st.session_state.analysis_done = True
                                
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                            return
        
        # Chat interface (outside the form)
        if st.session_state.assistant is not None:
            st.markdown("### Interactive Resume Analysis Chat")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                for role, message in st.session_state.chat_history:
                    if role == "assistant":
                        st.markdown(f"""
                            <div class="assistant-message">
                                <i class="fas fa-robot"></i> <b>AI Assistant:</b><br>{message}
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div class="user-message">
                                <i class="fas fa-user"></i> <b>You:</b><br>{message}
                            </div>
                        """, unsafe_allow_html=True)
                
                # Always show prompt for more questions
                st.markdown("""
                    <div class="assistant-message">
                        Would you like to know anything else about the resume match? Feel free to ask another question!
                    </div>
                """, unsafe_allow_html=True)
            
            # Chat Input area - Always visible
            col1, col2 = st.columns([3, 1])
            with col1:
                user_question = st.text_input(
                    "Ask a specific question about the resume match:",
                    key=f"user_input_{len(st.session_state.chat_history)}"  # Unique key for each interaction
                )
            with col2:
                st.markdown("<div style='height: 32px'></div>", unsafe_allow_html=True)  # Spacing
                ask_more = st.button(
                    "Ask Question",
                    key=f"ask_button_{len(st.session_state.chat_history)}",  # Unique key for each interaction
                    use_container_width=True
                )
            
            if ask_more and user_question:
                try:
                    with st.spinner('Getting response...'):
                        # Add user question to history
                        st.session_state.chat_history.append(("user", user_question))
                        
                        # Get AI response
                        response = st.session_state.assistant.chat(user_question)
                        st.session_state.chat_history.append(("assistant", response))
                        
                        # Clear the input by rerunning
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Error during chat: {str(e)}")
            
            # Add some spacing at the bottom
            st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    else:
        # Original handling for Cover Letter Generator
        with st.form(config["form_id"]):
            description = st.text_area(
                config["desc_label"],
                height=200,
                placeholder="Paste the job description here..."
            )
            
            file = st.file_uploader(
                config["file_label"],
                type=["pdf"],
                accept_multiple_files=False
            )
            
            submitted = st.form_submit_button(
                config["submit_label"],
                use_container_width=True
            )
            
            if submitted:
                is_valid, error_message = check_inputs(openai_api_key, description, file)
                if not is_valid:
                    st.error(error_message, icon='⚠')
                else:
                    generate_document(config["generator_func"], description, file, openai_api_key)

def render_voice_chat_page():
    """Render the voice chat assistant page."""
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_recorded_audio' not in st.session_state:
        st.session_state.last_recorded_audio = None
    if 'awaiting_response' not in st.session_state:
        st.session_state.awaiting_response = False
    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = deque(maxlen=5)
    if 'voice_assistant' not in st.session_state:
        st.session_state.voice_assistant = VoiceAssistant(st.secrets["openai_api_key"])
    if 'cleanup_on_start' not in st.session_state:
        st.session_state.voice_assistant.cleanup()
        st.session_state.cleanup_on_start = True
    if 'resume_analyzed' not in st.session_state:
        st.session_state.resume_analyzed = False
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'input_mode' not in st.session_state:  # New state for tracking input mode
        st.session_state.input_mode = "text"

    timestamp = int(time.time() * 1000)
    
    st.markdown(f"""
        <div class="page-container animation-{timestamp}">
            <h1 class="page-title animated-title animation-{timestamp}">Interview Preparation Assistant</h1>
            <p class="page-subtitle animation-{timestamp}">Upload your resume and start practicing for your interview</p>
        </div>
    """, unsafe_allow_html=True)

    # Resume Analysis Section (if analysis hasn't been done)
    if not st.session_state.resume_analyzed:
        st.markdown("### Step 1: Resume Analysis")
        with st.form("resume_analysis_form"):
            description = st.text_area(
                "Enter Job Description",
                height=200,
                placeholder="Paste the job description here..."
            )
            
            file = st.file_uploader(
                "Upload your Resume",
                type=["pdf"],
                accept_multiple_files=False
            )
            
            submitted = st.form_submit_button(
                "Start Interview Prep ✨",
                use_container_width=True
            )
            
            if submitted:
                is_valid, error_message = check_inputs(openai_api_key, description, file)
                if not is_valid:
                    st.error(error_message, icon='⚠')
                else:
                    try:
                        with st.spinner('Analyzing your resume...'):
                            initial_response = st.session_state.voice_assistant.initialize_interview_prep(
                                file,
                                description
                            )
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": "📊 Initial Analysis:\n\n" + initial_response,
                            })
                            
                            st.session_state.resume_analyzed = True
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error in resume analysis: {str(e)}")

     # Chat Interface
    if st.session_state.resume_analyzed:
        st.markdown("### Interactive Interview Practice")
        
        # Display current interview progress if interview is in progress
        if hasattr(st.session_state.voice_assistant, 'interview_state') and \
        st.session_state.voice_assistant.interview_state["in_progress"]:
            current_q = st.session_state.voice_assistant.interview_state["current_question"]
            total_q = len(st.session_state.voice_assistant.interview_state["questions"])
            st.progress(current_q/total_q, text=f"Question {current_q + 1} of {total_q}")

        # Chat history container
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        # For user messages, display the transcribed text without the 🎤 emoji
                        display_text = message["content"]
                        if display_text.startswith("🎤 "):
                            display_text = display_text[2:].strip()
                        st.markdown(display_text)
                    else:
                        # For assistant messages, display the content as before
                        st.markdown(message["content"])
                    
                    # Handle audio playback if present
                    if "audio" in message:
                        audio_base64 = st.session_state.voice_assistant.get_base64_audio(message["audio"])
                        if audio_base64:
                            st.markdown(
                                f'<audio src="data:audio/mp3;base64,{audio_base64}" controls autoplay>',
                                unsafe_allow_html=True
                            )

        # Combined input area
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Text input with unique key
            text_input = st.chat_input(
                "Respond to the interview question or type 'yes' to start the interview...",
                key="chat_input_unique"
            )
            
        with col2:
            # Voice recorder
            st.markdown('<div class="voice-recorder-container" style="margin-top: 10px;">', unsafe_allow_html=True)
            recorded_audio = audio_recorder(
                text="",  # Remove text to show just the button
                recording_color="#e74c3c",
                neutral_color="#95a5a6",
                key="voice_recorder"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Handle text input
        if text_input and not st.session_state.awaiting_response:
            st.session_state.awaiting_response = True
            try:
                # First add the user's message
                st.session_state.messages.append({
                    "role": "user",
                    "content": text_input
                })
                
                # Then get and add the assistant's response
                response, audio_file = st.session_state.voice_assistant.chat(
                    text_input,
                    input_type="text",
                    output_type="voice"
                )
                
                message = {
                    "role": "assistant",
                    "content": response
                }
                if audio_file:
                    message["audio"] = audio_file
                
                st.session_state.messages.append(message)
                
            except Exception as e:
                st.error(f"Error processing text input: {str(e)}")
            
            st.session_state.awaiting_response = False
            st.rerun()

        # Handle voice input
        if recorded_audio is not None and recorded_audio != st.session_state.last_recorded_audio:
            st.session_state.awaiting_response = True
            st.session_state.last_recorded_audio = recorded_audio
            
            try:
                # Get the transcription and response
                input_text = st.session_state.voice_assistant.process_input(recorded_audio, input_type="voice")
                
                # Add the transcribed user message with the 🎤 emoji
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"🎤 {input_text}"
                })
                
                # Get assistant's response
                response, audio_file = st.session_state.voice_assistant.chat(
                    input_text,
                    input_type="text",  # Already transcribed, so treat as text
                    output_type="voice"
                )
                
                # Add the assistant's response
                message = {
                    "role": "assistant",
                    "content": response,
                }
                if audio_file:
                    message["audio"] = audio_file
                
                st.session_state.messages.append(message)
                
            except Exception as e:
                st.error(f"Error processing voice input: {str(e)}")
            
            st.session_state.awaiting_response = False
            st.rerun()

def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
        page = PageType(st.radio("Select Page:", [page.value for page in PageType]))
    
    # Main content
    if page == PageType.HOME:
        render_home_page()
    elif page == PageType.VOICE_CHAT:  # New condition for voice chat page
        render_voice_chat_page()
    else:
        render_generator_page(page)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <p style='color: #666; font-family: Poppins, sans-serif; animation: fadeIn 1s ease-out;'>
                SU iBot
            </p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()