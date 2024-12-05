import streamlit as st
import pandas as pd
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import numpy as np
import os
from dotenv import load_dotenv, find_dotenv

# OpenAIKey
os.environ['OPENAI_API_KEY']=st.secrets["openai_api_key"]
load_dotenv(find_dotenv())


@st.cache_resource
def get_llm():
    """Initialize and return the OpenAI LLM instance"""
    return OpenAI(temperature=0)

# Cache the pandas agent creation
@st.cache_resource
def get_pandas_agent(_llm, df):
    """Create and return the pandas dataframe agent"""
    return create_pandas_dataframe_agent(_llm, df, verbose=True, allow_dangerous_code=True)

def create_visualization(df, column_name, viz_type, title=None):
    """Create different types of visualizations using Matplotlib based on the specified type"""
    try:
        # Create figure and axis objects
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "bar":
            if df[column_name].dtype in ['object', 'category']:
                value_counts = df[column_name].value_counts()
                value_counts.plot(kind='bar', ax=ax)
            else:
                df[column_name].plot(kind='bar', ax=ax)
                
        elif viz_type == "line":
            # Handle both series and dataframe inputs
            if isinstance(df, pd.DataFrame):
                if pd.api.types.is_numeric_dtype(df[column_name]):
                    # Create line plot with index as x-axis
                    ax.plot(df.index, df[column_name])
                    ax.set_ylabel(column_name)
                else:
                    raise ValueError(f"Column {column_name} must be numeric for line plots")
            else:
                ax.plot(df.index, df)
            
        elif viz_type == "scatter":
            ax.scatter(df.index, df[column_name])
            
        elif viz_type == "histogram":
            # Ensure the data is numeric and handle NaN values
            if pd.api.types.is_numeric_dtype(df[column_name]):
                data = df[column_name].dropna()
                
                # Calculate optimal number of bins using Sturges' rule
                n_bins = int(np.log2(len(data)) + 1)
                
                # Create histogram with density=False to show counts
                ax.hist(data, bins=n_bins, edgecolor='black', alpha=0.7)
                
                # Add mean and median lines
                mean_val = data.mean()
                median_val = data.median()
                ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
                
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                raise ValueError(f"Column {column_name} must be numeric for histograms")
            
        elif viz_type == "box":
            df[column_name].plot(kind='box', ax=ax)
            
        elif viz_type == "pie":
            value_counts = df[column_name].value_counts()
            ax.pie(value_counts.values, labels=value_counts.index, autopct='%1.1f%%')
            ax.axis('equal')
            
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        # Set title if provided
        if title:
            ax.set_title(title)
        
        # Customize the appearance
        ax.set_xlabel(column_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Display in Streamlit
        st.pyplot(fig)
        plt.close()
        return True
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

def store_visualization(column, viz_type, title, stats=None):
    """Store visualization metadata in session state"""
    viz_data = {
        'column': column,
        'type': viz_type,
        'title': title,
        'stats': stats,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.visualization_history.append(viz_data)

@st.cache_data(show_spinner="Processing question...", ttl="10m")
def process_question(_pandas_agent, question, df):
    try:
        # Keep existing context window logic
        context = [f"{msg['role']}: {msg['content']}" for msg in st.session_state.chat_history[-5:]]
        context_str = "\n".join(context)

        # Keep all existing visualization keywords and logic
        insights_keywords = ["insights", "analyze", "explain", "describe", "tell me about", "what do you see", "what can you tell"]
        explanation_keywords = ["explain", "describe", "tell me about", "what does", "analyze", "interpret"]
        visualization_keywords = ["show", "plot", "display", "visualize", "graph", "chart", "distribution", "histogram", "box", "vs", "versus", "compare", "correlation", "visualization"]
        
        # Add new analysis keywords
        analysis_keywords = ["what", "which", "how many", "list", "tell me", "find", "calculate", "compute", "analyze", 
                           "highest", "lowest", "top", "bottom", "average", "mean", "median", "most", "least"]
        
        needs_viz = any(keyword in question.lower() for keyword in visualization_keywords)
        needs_analysis = any(keyword in question.lower() for keyword in analysis_keywords)
        is_analysis_request = any(keyword in question.lower() for keyword in insights_keywords)
        is_viz_explanation = any(keyword in question.lower() for keyword in explanation_keywords) and "graph" in question.lower()

        response = None
        viz_data = None

        # Handle existing visualization logic first
        if is_analysis_request and st.session_state.visualization_history:
            # Keep existing visualization analysis logic
            last_viz = st.session_state.visualization_history[-1]
            columns_str = last_viz.get('column') if 'column' in last_viz else ', '.join(last_viz.get('columns', []))
            analysis_prompt = f"""
            Analyze this visualization of {columns_str} using actual data:
            1. For single column:
                - Distribution/frequency patterns
                - Key statistics (mean, median, mode)
                - Notable outliers
            2. For multiple columns:
                - Relationships/correlations
                - Trends across categories
                - Key differences between groups
            3. Business implications of the patterns
            Format: "Analysis: [your insights]"
            """
            response = _pandas_agent.run(analysis_prompt)
            if "Analysis:" in response:
                response = response.split("Analysis:")[1].strip()
            return response, None

        if is_viz_explanation and st.session_state.visualization_history:
            # Keep existing visualization explanation logic
            last_viz = st.session_state.visualization_history[-1]
            columns_str = last_viz.get('column') if 'column' in last_viz else ', '.join(last_viz.get('columns', []))
            analysis_prompt = f"""
            Analyze the {last_viz['type']} chart of {columns_str} that was just displayed.
            Provide insights about:
            1. The overall distribution/pattern
            2. Any notable trends or outliers
            3. Key statistics if relevant
            Keep the explanation clear and data-focused.
            """
            response = _pandas_agent.run(analysis_prompt)
            return response, None

        # Handle visualization if needed
        if needs_viz:
            # Keep all existing visualization logic
            viz_query = _pandas_agent.run(f"""
            For the question: "{question}"
            1. Identify which columns need to be visualized
            2. Choose the best visualization type (scatter/bar/line/box/pie)
            3. Return ONLY in this format:
            COLUMNS: column1, column2 (if comparing two columns)
            TYPE: <viz_type>
            """)
            
            try:
                lines = viz_query.strip().split('\n')
                columns = []
                viz_type = None
                
                for line in lines:
                    if line.startswith('COLUMNS:'):
                        columns = [c.strip() for c in line.replace('COLUMNS:', '').split(',')]
                    elif line.startswith('TYPE:'):
                        viz_type = line.replace('TYPE:', '').strip().lower()
                
                if columns and viz_type and all(col in df.columns for col in columns):
                    if len(columns) == 1:
                        viz_data = {
                            'column': columns[0],
                            'type': viz_type,
                            'title': f"{viz_type.capitalize()} of {columns[0]}"
                        }
                        create_visualization(df, columns[0], viz_type, viz_data['title'])
                    else:
                        viz_data = {
                            'columns': columns,
                            'type': viz_type,
                            'title': f"{viz_type.capitalize()} of {columns[0]} vs {columns[1]}"
                        }
                        create_multi_column_viz(df, columns, viz_type)
                    
                    if 'visualization_history' in st.session_state:
                        st.session_state.visualization_history.append(viz_data)
            except Exception as viz_error:
                st.error(f"Visualization error: {str(viz_error)}")

        # Add new analysis handling
        if needs_analysis:
            analysis_prompt = f"""
            Question: {question}
            
            Previous context:
            {context_str}
            
            Instructions:
            1. Provide a direct and specific answer to the question
            2. If the question asks for specific items (top N, bottom N, etc.), list them with their values
            3. If numerical calculations are needed, show the exact values
            4. If relevant, include key statistics or percentages
            5. Keep the response clear and concise
            
            Response format:
            - For lists: Use clear numbering (1., 2., etc.)
            - For calculations: Show the actual values
            - For comparisons: Include specific numbers
            """
            
            analysis_response = _pandas_agent.run(analysis_prompt)
            
            # If we already have a visualization response, combine them
            if response:
                response = f"{analysis_response}\n\n{response}"
            else:
                response = analysis_response

        # If we still don't have a response, use the original question
        if not response:
            response = _pandas_agent.run(analysis_prompt if not needs_viz else question)
        
        return response, viz_data
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}", None

# Cache initial analysis
@st.cache_data(show_spinner="Performing initial analysis...")
def initial_analysis(_pandas_agent, df):
    """Perform initial EDA and store results in session state"""
    if not st.session_state.analysis_complete:
        st.write("Data Overview")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        st.write("\n*Categorical Columns Available:*")
        st.write(list(categorical_columns))
        
        st.write("Data Quality Assessment")
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

def create_multi_column_viz(df, columns, viz_type):
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if viz_type == "scatter":
            ax.scatter(df[columns[0]], df[columns[1]], alpha=0.5)
        elif viz_type == "bar":
            # For categorical vs numerical comparisons
            grouped_data = df.groupby(columns[0])[columns[1]].mean().sort_values(ascending=False)
            grouped_data.plot(kind='bar', ax=ax)
        elif viz_type == "line":
            df.sort_values(columns[0]).plot(x=columns[0], y=columns[1], kind='line', ax=ax)
        elif viz_type == "box":
            df.boxplot(column=columns[1], by=columns[0], ax=ax)
            
        ax.set_xlabel(columns[0])
        ax.set_ylabel(columns[1])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        return True
    except Exception as e:
        st.error(f"Error in multi-column visualization: {str(e)}")
        return False
