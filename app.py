import streamlit as st
from cover_letter_generator import get_cover_letter
from resume_generator import get_resume
from enum import Enum
from typing import Callable, Tuple
import time
import pathlib

class PageType(Enum):
    HOME = "Home"
    COVER_LETTER = "Cover Letter Generator"
    RESUME = "Resume Generator"
    INTERVIEW_PREP = "Interview Preparation"

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
    
    col1, col2 = st.columns(2)
    
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
    
    st.markdown(f"""
        <div class="title-container">
            <h1 class="main-title" style="font-size: 3rem !important;">{config["title"]}</h1>
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
                        st.experimental_rerun()
                        
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

def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-title">Navigation</h2>', unsafe_allow_html=True)
        page = PageType(st.radio("Select Page:", [page.value for page in PageType]))
    
    # Main content
    if page == PageType.HOME:
        render_home_page()
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