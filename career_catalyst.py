import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.prompts import PromptTemplate

openai_api_key = st.secrets["openai_api_key"]

@st.cache_resource
def get_llm():
    """Initialize and return the LLM instance"""
    return OpenAI(temperature=0, openai_api_key=openai_api_key)

@st.cache_resource
def get_pandas_agent(_llm, df):
    """Create and return the pandas dataframe agent"""
    return create_pandas_dataframe_agent(
        _llm,
        df,
        verbose=True,
        allow_dangerous_code=True,
        max_iterations=1000,
        max_execution_time=300,
        handle_parsing_errors=True
    )

def create_visualization(df, column_name, viz_type, title=None):
    """Create different types of visualizations based on the specified type"""
    try:
        if viz_type == "bar":
            fig = px.bar(df, x=column_name, title=title)
        elif viz_type == "line":
            fig = px.line(df, y=column_name, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(df, x=df.index, y=column_name, title=title)
        elif viz_type == "histogram":
            fig = px.histogram(df, x=column_name, title=title)
        elif viz_type == "box":
            fig = px.box(df, y=column_name, title=title)
        elif viz_type == "pie":
            value_counts = df[column_name].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        st.plotly_chart(fig)
        return True
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

@st.cache_data(show_spinner="Analyzing categorical data...")
def analyze_categorical_data(_pandas_agent, column_name, df):
    st.write(f"**Analysis of {column_name}**")
    
    if column_name not in df.columns:
        st.error(f"Column '{column_name}' not found in the dataset")
        return
    
    try:
        if isinstance(df[column_name], pd.DataFrame):
            value_counts = df[column_name].iloc[:, 0].value_counts()
        else:
            value_counts = df[column_name].value_counts()
            
        st.write("Category Distribution:")
        st.write(value_counts)
        
        st.write("Distribution Visualization:")
        chart_data = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        distribution = _pandas_agent.run(f"""Analyze the distribution of {column_name} and provide insights. 
        Include:
        1. Most common categories
        2. Least common categories
        3. Any interesting patterns
        4. Potential business implications
        """)
        st.write("**Key Insights:**")
        st.write(distribution)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("Raw data for debugging:")
        st.write(df[column_name].head())

@st.cache_resource
def get_templates():
    """Create and return the prompt templates"""
    simple_template = PromptTemplate(
        input_variables=['question'],
        template='Calculate and return {question} in a clear, direct way using the given dataset.'
    )
    
    moderate_template = PromptTemplate(
        input_variables=['question'],
        template='''Analyze {question} by:
        1. Calculating relevant metrics
        2. Identifying any obvious patterns
        3. Consider if visualization would be helpful'''
    )
    
    complex_template = PromptTemplate(
        input_variables=['question'],
        template='''Perform a thorough analysis of {question}:
        1. Calculate key metrics and statistics
        2. Identify significant patterns and trends
        3. Consider relationships between different variables
        4. Suggest appropriate visualizations
        5. Provide actionable insights'''
    )
    
    return {
        'SIMPLE': simple_template,
        'MODERATE': moderate_template,
        'COMPLEX': complex_template
    }

@st.cache_data
def classify_question(_llm, question):
    """Classify the complexity of the question"""
    classification_prompt = f"""
    Analyze this data analysis question: "{question}"
    Classify it as:
    'SIMPLE' if it only needs basic statistics or a single metric
    'MODERATE' if it needs some analysis and maybe visualization
    'COMPLEX' if it needs in-depth analysis with multiple aspects
    Return only one word: SIMPLE, MODERATE, or COMPLEX
    """
    response = _llm(classification_prompt)
    return response.strip()

def initial_analysis(_pandas_agent, df):
    """Perform initial EDA and store results in session state"""
    if not st.session_state.analysis_complete:
        st.write("**Data Overview**")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        st.write("\n**Categorical Columns Available:**")
        st.write(list(categorical_columns))
        
        st.write("**Data Quality Assessment**")
        columns_df = _pandas_agent.run("""For each column, provide:
        1. The data type
        2. Whether it's categorical or numerical
        3. A brief description of what the column represents
        """)
        st.write(columns_df)
        
        missing_values = _pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
        
        duplicates = _pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        
        st.session_state.analysis_complete = True
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I've completed the initial analysis. What would you like to know about your data? You can ask me anything about the patterns, relationships, or specific aspects of your dataset."
        })

@st.cache_data(show_spinner="Processing question...", ttl="10m")
def process_question(_pandas_agent, question, df):
    """Process any data-related question and determine appropriate visualization"""
    try:
        # First, let the agent analyze the question and form a plan
        analysis_prompt = f"""
        For this question: "{question}"
        
        1. What columns are relevant to this question?
        2. What type of analysis is needed?
        3. Would visualization help understand the answer?
        
        Return response as JSON:
        {{
            "relevant_columns": ["col1", "col2"],
            "analysis_type": "descriptive|comparative|relationship|trend",
            "viz_needed": true|false,
            "viz_suggestions": [
                {{
                    "type": "bar|box|scatter|line|pie",
                    "columns": ["col1", "col2"],
                    "purpose": "brief explanation"
                }}
            ]
        }}
        """
        
        try:
            # Get analysis plan
            plan = _pandas_agent.run(analysis_prompt)
            
            # Execute detailed query with context preservation
            execution_prompt = f"""
            For this question: "{question}"
            When analyzing:
            1. Consider relationships between relevant columns
            2. If finding specific values, include their related data
            3. Process complete rows of data when needed
            4. Include supporting evidence in the response
            
            Execute the analysis and return complete results.
            """
            
            response = _pandas_agent.run(execution_prompt)
            response_text = f"**Analysis:**\n{response}"
                
            # Create visualizations if needed
            try:
                if "viz_needed" in plan.lower() and "true" in plan.lower():
                    viz_text = []
                    for viz in eval(plan).get("viz_suggestions", []):
                        try:
                            fig = None
                            if viz["type"] == "bar":
                                cols = viz["columns"]
                                if len(cols) == 2:
                                    agg_data = df.groupby(cols[0])[cols[1]].mean().reset_index()
                                    fig = px.bar(
                                        agg_data,
                                        x=cols[0],
                                        y=cols[1],
                                        title=f'Average {cols[1]} by {cols[0]}'
                                    )
                                elif len(cols) == 1:
                                    value_counts = df[cols[0]].value_counts().reset_index()
                                    fig = px.bar(
                                        value_counts,
                                        x="index",
                                        y=cols[0],
                                        title=f'Distribution of {cols[0]}'
                                    )
                                    
                            elif viz["type"] == "box":
                                cols = viz["columns"]
                                fig = px.box(
                                    df,
                                    x=cols[0],
                                    y=cols[1] if len(cols) > 1 else None,
                                    title=f'Distribution of {cols[-1]}'
                                )
                                
                            elif viz["type"] == "scatter":
                                cols = viz["columns"]
                                fig = px.scatter(
                                    df,
                                    x=cols[0],
                                    y=cols[1],
                                    color=cols[2] if len(cols) > 2 else None,
                                    title=f'Relationship between {cols[0]} and {cols[1]}'
                                )
                            
                            if fig:
                                st.plotly_chart(fig)
                                viz_text.append(f"**{viz.get('purpose', 'Visualization')}**")
                                
                        except Exception as viz_error:
                            continue
            
                # Get additional insights
                insights_prompt = f"""
                Based on the analysis results, provide:
                1. Key patterns or insights
                2. Relevant context for the findings
                3. Any notable relationships in the data
                Keep insights focused on the original question.
                """
                
                insights = _pandas_agent.run(insights_prompt)
                response_text += f"\n\n**Additional Insights:**\n{insights}"
                
            except Exception as e:
                pass
            
            return response_text
            
        except Exception as e:
            # Direct question handling if JSON parsing fails
            response = _pandas_agent.run(f"""
            Analyze: {question}
            Important:
            1. Include all relevant column relationships
            2. Show supporting data
            3. Verify results before responding
            """)
            return f"**Analysis:**\n{response}"
            
    except Exception as e:
        error_msg = f"An error occurred while processing your question: {str(e)}"
        try:
            # Simple fallback analysis
            response = _pandas_agent.run(f"Analyze {question} in the simplest way possible.")
            return f"**Basic Analysis:**\n{response}"
        except Exception as fallback_error:
            return "Could not complete the analysis. Please try rephrasing your question."

def clicked(button):
    """Update session state when button is clicked"""
    st.session_state.clicked[button] = True