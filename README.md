# SU Career iBot

## Overview

**SU Career iBot** is an AI-powered chatbot designed to enhance career preparation for Syracuse University iSchool students and streamline operations for the Career Services team. By leveraging advanced AI models, the bot provides personalized guidance on resumes, cover letters, and interview preparation, while also offering data-driven insights to Career Services.

---

## Features

### For Students:
1. **Resume and Cover Letter Guidance**:
   - Upload resumes and job descriptions for tailored feedback.
   - Generate personalized cover letters aligned with iSchool templates.
   - Rewrite bullet points using the STAR format for enhanced impact.
2. **Interview Preparation**:
   - Simulate interviews with audio-based questions using TTS and Whisper API.
   - Provide feedback on tone, content, and delivery style.
3. **Career Insights**:
   - Analyze employment trends of Syracuse alumni to identify skill demands and top employers.

### For Career Services:
1. **Alumni Employment Data Analysis**:
   - Visualize job placement trends, skill demands, and employer engagement.
2. **Resource Optimization**:
   - Utilize data insights for curriculum updates and strategic planning.
3. **Employer Engagement**:
   - Support targeted outreach to employers based on analyzed data.

---

## System Components

### 1. Resume and Cover Letter Bot
- **Functionality**:
  - Provides feedback on resumes and generates tailored cover letters.
  - Integrates RAG (Retrieval-Augmented Generation) for contextual responses.
- **Key APIs and Libraries**:
  - OpenAI GPT-4 API for language processing.
  - Chroma for vector database storage and retrieval.
  - LangChain for chaining AI tasks.
  - Streamlit for user interface.

### 2. Career Analytics Bot
- **Functionality**:
  - Visualizes alumni employment trends and skill demands.
  - Generates interactive graphs and charts using natural language queries.
- **Key Tools**:
  - Pandas and Matplotlib for data processing and visualization.
  - Streamlit for interactive user experience.

### 3. Interview Preparation Bot
- **Functionality**:
  - Simulates interviews with audio questions and real-time feedback.
  - Analyzes responses for tone, relevance, and alignment with company culture.
- **Key APIs and Libraries**:
  - OpenAI GPT-4 API for conversational analysis.
  - TTS and Whisper APIs for audio processing.

---

## Strengths and Challenges

### Strengths:
- Personalized and automated career guidance for students.
- Data-driven decision-making for Career Services.
- User-friendly interface built with Streamlit.
- Scalable architecture using modular bots.

### Challenges:
- Integration of multiple bots into a unified system.
- Backend optimization for large datasets.
- Providing advanced customization options for niche use cases.

---

## Ethical Considerations
- **Data Privacy**: Alumni data is anonymized and securely stored.
- **Bias Mitigation**: AI models are trained on diverse datasets to minimize bias.
- **Inclusive Access**: Democratizes career preparation resources for students of varied backgrounds.

---

## Lessons Learned
- **Technical**: Integration of APIs, database optimization, and AI prompt engineering.
- **Collaboration**: The importance of teamwork and communication for managing complex interdependencies.
- **Problem-Solving**: Iterative testing and refinement to address technical challenges.

---

## How to Use

1. **Resume and Cover Letter Assistance**:
   - Upload your resume and a job description.
   - Receive feedback and a personalized cover letter.
2. **Interview Preparation**:
   - Simulate interviews with audio-based questions.
   - Get detailed feedback on your responses.
3. **Career Insights**:
   - Analyze trends in alumni employment data for actionable insights.

---

## Future Enhancements
- AI-powered job recommendations for students.
- Expanded analytics for broader career insights.
- Integration with additional career platforms.

---

## Technical Stack
- **Programming Language**: Python
- **Frameworks**: Streamlit, LangChain
- **APIs**: OpenAI GPT-4, TTS, Whisper, SerpAPI
- **Database**: Chroma for vector storage

---

## Links
- **Live Demo**: [SU Career iBot Application]([https://your-app-link.com](https://su-career-ibot.streamlit.app/))
