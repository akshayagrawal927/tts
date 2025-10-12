import streamlit as st
"""
Pulse AI - Professional Text-to-SQL Assistant with Smart Charting and Fuzzy Search
Intelligent Data Analytics Platform
"""
from io import StringIO
from typing import Tuple
import os
import json
import uuid
import time
import random
import pyodbc
import logging
import re
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Dict, Any, List, Annotated, TypedDict, Literal
import base64
from io import BytesIO
from PIL import Image
from difflib import SequenceMatcher
from thefuzz import fuzz 
# Streamlit imports
import streamlit as st
import matplotlib.pyplot as plt

# LangChain and LangGraph components
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.synapse_connection import SynapseConnectionManager
from src.synapse_agent import ContextAwareSynapseAgent
from src.session_manager import SessionManager
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.config import AZURE_OPENAI_CONFIG, SYNAPSE_CONFIG
from styles.css import load_logo_base64

logger = logging.getLogger(__name__)


def get_user_id():
    """Generate or retrieve user ID for session isolation."""
    if 'user_id' not in st.session_state:
        # Generate unique user ID based on session info
        import hashlib
        session_info = f"{st.session_state.get('session_id', '')}{datetime.now().isoformat()}"
        st.session_state.user_id = hashlib.sha256(session_info.encode()).hexdigest()[:16]
        logger.info(f"Generated new user ID: {st.session_state.user_id}")
    
    return st.session_state.user_id

@st.cache_resource
def init_agent():
    """Initialize agent with caching to prevent reloading."""
    try:
        logger.info("Initializing Pulse AI agent")
        return ContextAwareSynapseAgent(AZURE_OPENAI_CONFIG)
    except Exception as e:
        logger.error(f"Failed to initialize Pulse AI: {e}")
        st.error(f"Failed to initialize Pulse AI: {e}")
        st.stop()

def init_session_state():
    """Initialize Streamlit session state with user isolation."""
    logger.info("Initializing session state with user isolation")

    # Generate unique user ID for session isolation
    if 'user_id' not in st.session_state:
        st.session_state.user_id = get_user_id()

    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()

    # Auto-load most recent session ONLY if no session is currently active
    if 'current_session_id' not in st.session_state or st.session_state.current_session_id is None:
        # Get all sessions for this user
        sessions = st.session_state.session_manager.get_all_sessions(st.session_state.user_id)
        
        if sessions:
            # Load the most recent session
            most_recent = sessions[0]  
            session_id = most_recent['session_id']
            thread_id = most_recent['thread_id']
            
            logger.info(f"Auto-loading most recent session: {session_id}")
            
            # Try to load from cache
            cached_session = st.session_state.session_manager.load_cached_session(
                session_id, st.session_state.user_id
            )
            
            if cached_session:
                st.session_state.chat_history = cached_session['chat_history']
                if 'agent' not in st.session_state:
                    st.session_state.agent = init_agent()
                st.session_state.agent.conversation_context = cached_session['agent_context']
                st.session_state.agent.query_history = cached_session['query_history']
            else:
                # Load from message history
                history = st.session_state.session_manager.get_session_history(
                    session_id, st.session_state.user_id
                )
                st.session_state.chat_history = history
            
            st.session_state.current_session_id = session_id
            st.session_state.current_thread_id = thread_id
            st.session_state.show_welcome = False if st.session_state.chat_history else True
            st.session_state.first_message_sent = len(st.session_state.chat_history) > 0
            
            # Cache in memory for faster access
            if 'loaded_sessions_cache' not in st.session_state:
                st.session_state.loaded_sessions_cache = {}
            
            st.session_state.loaded_sessions_cache[session_id] = {
                'chat_history': st.session_state.chat_history,
                'agent_context': st.session_state.agent.conversation_context if 'agent' in st.session_state else {},
                'query_history': st.session_state.agent.query_history if 'agent' in st.session_state else []
            }
        else:
            # No existing sessions, will create new one
            st.session_state.current_session_id = None
            st.session_state.current_thread_id = None
            st.session_state.chat_history = []
            st.session_state.show_welcome = True
            st.session_state.first_message_sent = False

    # Initialize other state variables if they don't exist
    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'agent' not in st.session_state:
        st.session_state.agent = init_agent()
        
        # Restore pending clarification if it exists
        if 'pending_clarification' in st.session_state:
            st.session_state.agent.pending_clarification = st.session_state['pending_clarification']
            logger.info("Restored pending clarification from session state")

    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True

    if 'current_followup_questions' not in st.session_state:
        st.session_state.current_followup_questions = []

    if 'first_message_sent' not in st.session_state:
        st.session_state.first_message_sent = False

    # Cache for loaded sessions to avoid reloading
    if 'loaded_sessions_cache' not in st.session_state:
        st.session_state.loaded_sessions_cache = {}

    if 'is_greeting_followup' not in st.session_state:
        st.session_state.is_greeting_followup = False

    # MODIFIED: Track timing for each message
    if 'message_timings' not in st.session_state:
        st.session_state.message_timings = {}


def display_header():
    """Display header without logo."""
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="app-title">
                <h1>Pulse AI</h1>
                <p>Data Analytics Assistant</p>
            </div>
        </div>
        <div class="bot-name">
             Assistant: Pulse AI
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message for new sessions."""
    agent = st.session_state.agent
    base_welcome_msg = agent.response_manager.get_welcome_message()
    dynamic_welcome_msg = agent._rephrase_message_with_llm(base_welcome_msg)
    
    st.markdown(f"""
    <div class="welcome-message">
        <div class="welcome-text">
            {dynamic_welcome_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_followup_questions(questions, is_greeting_response=False):
    """Display follow-up questions with context-aware labeling."""
    if questions and len(questions) > 0:
        if is_greeting_response:
            title = "Suggested questions to get started:"
        else:
            title = "Follow-up questions:"
        
        st.markdown(f"""
        <div class="followup-container">
            <div class="followup-title">{title}</div>
        </div>
        """, unsafe_allow_html=True)

        for i, question in enumerate(questions):
            if st.button(question, key=f"followup_{i}_{hash(question)}", use_container_width=True):
                # Clear current follow-up questions
                st.session_state.current_followup_questions = []

                # Process the follow-up question
                logger.info(f"User clicked follow-up question: {question}")
                process_user_input(question)

def parse_dataframe_from_string(raw_dataframe: str) -> pd.DataFrame:
    """Robust DataFrame parsing from various string formats."""
    
    from io import StringIO
    import ast
    import re
    
    # Clean the string
    cleaned = raw_dataframe.strip().lstrip('\ufeff').lstrip('\x00')
    
    if not cleaned:
        logger.debug("Empty DataFrame string")
        return pd.DataFrame()
    
    # Method 1: Check if it's a DataFrame string representation
    try:
        # If it looks like DataFrame.__str__ output, try to parse it
        if '\n' in cleaned and ('0  ' in cleaned or '1  ' in cleaned):
            logger.debug("Detected DataFrame string representation format")
            lines = cleaned.split('\n')
            
            # Find header line (usually first non-empty line)
            header_line = None
            data_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this looks like a header (no leading numbers)
                if not re.match(r'^\s*\d+\s+', line) and header_line is None:
                    # This might be the header
                    header_line = line
                elif re.match(r'^\s*\d+\s+', line):
                    # This looks like a data row
                    data_lines.append(line)
            
            if header_line and data_lines:
                # Parse header
                headers = re.split(r'\s{2,}', header_line.strip())
                
                # Parse data rows
                data_rows = []
                for line in data_lines:
                    # Remove leading index number
                    line_without_index = re.sub(r'^\s*\d+\s+', '', line)
                    # Split by multiple spaces
                    row_data = re.split(r'\s{2,}', line_without_index.strip())
                    
                    # Convert numeric strings to numbers
                    converted_row = []
                    for value in row_data:
                        try:
                            # Try int first, then float
                            if '.' in value:
                                converted_row.append(float(value))
                            else:
                                converted_row.append(int(value))
                        except ValueError:
                            converted_row.append(value)
                    
                    data_rows.append(converted_row)
                
                # Create DataFrame
                if data_rows and headers:
                    dataframe = pd.DataFrame(data_rows, columns=headers)
                    logger.debug(f"Method 1 success (DataFrame string): DataFrame with {len(dataframe)} rows")
                    return dataframe
    except Exception as e:
        logger.debug(f"Method 1 failed (DataFrame string parsing): {e}")
    
    # Method 2: Direct JSON parsing
    try:
        json_data = json.loads(cleaned)
        if json_data:
            dataframe = pd.DataFrame(json_data)
            logger.debug(f"Method 2 success: DataFrame with {len(dataframe)} rows")
            return dataframe
    except json.JSONDecodeError as e:
        logger.debug(f"Method 2 failed: JSON decode error - {e}")
    except Exception as e:
        logger.debug(f"Method 2 failed: Unexpected error - {e}")
    
    # Method 3: Fix common JSON issues and retry
    try:
        # Replace single quotes with double quotes
        if "'" in cleaned and '"' not in cleaned:
            fixed_json = cleaned.replace("'", '"')
        else:
            fixed_json = cleaned
            
        # Try to fix trailing commas
        fixed_json = re.sub(r',\s*}', '}', fixed_json)
        fixed_json = re.sub(r',\s*]', ']', fixed_json)
        
        json_data = json.loads(fixed_json)
        if json_data:
            dataframe = pd.DataFrame(json_data)
            logger.debug(f"Method 3 success: DataFrame with {len(dataframe)} rows")
            return dataframe
    except Exception as e:
        logger.debug(f"Method 3 failed: {e}")
    
    # Method 4: Pandas read_json with StringIO
    try:
        dataframe = pd.read_json(StringIO(cleaned), orient='records')
        logger.debug(f"Method 4 success: DataFrame with {len(dataframe)} rows")
        return dataframe
    except Exception as e:
        logger.debug(f"Method 4 failed: {e}")
    
    # Method 5: Try different pandas orientations
    for orient in ['index', 'columns', 'values', 'split']:
        try:
            dataframe = pd.read_json(StringIO(cleaned), orient=orient)
            logger.debug(f"Method 5 success with orient='{orient}': DataFrame with {len(dataframe)} rows")
            return dataframe
        except Exception:
            continue
    
    # Method 6: Try to extract data from malformed JSON manually
    try:
        # Sometimes the string might be Python dict format instead of JSON
        if cleaned.startswith('[') and cleaned.endswith(']'):
            # Try to evaluate as Python literal
            data = ast.literal_eval(cleaned)
            dataframe = pd.DataFrame(data)
            logger.debug(f"Method 6 success: DataFrame with {len(dataframe)} rows")
            return dataframe
    except Exception as e:
        logger.debug(f"Method 6 failed: {e}")
    
    # All methods failed
    logger.warning(f"All DataFrame parsing methods failed for: {cleaned[:100]}...")
    return pd.DataFrame()

def format_markdown_text(text: str) -> str:
    """
    Convert markdown-style formatting to HTML.
    Handles **bold**, __underline__, and other formatting.
    """
    import re
    
    # Remove smart quotes
    text = text.replace('"', '').replace('"', '').replace('"', '')
    
    # Convert **bold** to <strong>bold</strong>
    # Use regex to handle multiple occurrences
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    
    # Convert __underline__ to <u>underline</u>
    text = re.sub(r'__(.+?)__', r'<u>\1</u>', text)
    
    # Convert *italic* to <em>italic</em>
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    
    # Convert line breaks
    text = text.replace('\n', '<br>')
    
    return text

def stream_response(text: str, delay: float = 0.03):
    """
    Generator function that yields characters from text with a delay.
    Args:
        text: The text to stream
        delay: Delay between characters in seconds (0.03 = 30ms, human-like typing)
    """
    for char in text:
        yield char
        time.sleep(delay)

def format_timestamp(timestamp_str):
    """Format timestamp to HH:MM:SS format."""
    try:
        dt = datetime.fromisoformat(timestamp_str)
        return dt.strftime('%H:%M:%S')
    except:
        return ""
    
def calculate_ttft(user_timestamp, bot_timestamp):
    """Calculate time to first token between user message and bot response."""
    try:
        user_dt = datetime.fromisoformat(user_timestamp)
        bot_dt = datetime.fromisoformat(bot_timestamp)
        diff = (bot_dt - user_dt).total_seconds()
        return f"{diff:.2f}s"
    except:
        return ""


def display_chat_message(message_data, is_user=True, message_index=0):
    
    """Display chat message with timestamps."""
    
    # Handle both string and dict inputs
    if isinstance(message_data, str):
        message_data = {'content': message_data, 'timestamp': ''}
    
    timestamp = message_data.get('timestamp', '')
    formatted_time = format_timestamp(timestamp)
    
    if is_user:
        # ✅ Extract content properly
        content = message_data.get('content', message_data.get('text', ''))
        
        # 🔍 DEBUG: Log to check values
        logger.info(f"USER MESSAGE DEBUG - Content: {content[:50]}, Timestamp: {timestamp}, Formatted: {formatted_time}")
        
        st.markdown(f"""
        <div class="user-message">
            <div style="margin-bottom: 8px;">
                <strong>You:</strong>
            </div>
            <div style="margin-bottom: 8px;">{content}</div>
            <div style="text-align: right;">
                <span style="font-size: 0.7rem; color: #6b7280; font-weight: normal;">{formatted_time}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Calculate TTFT if this is a bot response
        ttft_display = ""
        if message_index > 0:
            prev_message = st.session_state.chat_history[message_index - 1]
            if prev_message.get('type') == 'user':
                user_timestamp = prev_message.get('timestamp', '')
                bot_timestamp = message_data.get('timestamp', '')
                ttft = calculate_ttft(user_timestamp, bot_timestamp)
                if ttft:
                    ttft_display = f' <span style="color: #9ca3af;"> {ttft}</span>'
        
        # PHASE 1: Stream insights first (with cursor animation)
        response_text = message_data.get('response', message_data.get('content', ''))
        message_placeholder = st.empty()
        is_latest_message = message_index == len(st.session_state.chat_history) - 1
        
        if is_latest_message and 'streamed' not in message_data:
            streamed_text = ""
            
            for char in stream_response(response_text, delay=0.02):
                streamed_text += char
                formatted_text = format_markdown_text(streamed_text)
                message_placeholder.markdown(f"""
                <div class="bot-message">
                    <div style="margin-bottom: 8px;">
                        <div class="bot-label">Pulse AI</div>
                    </div>
                    <div style="margin-top: 10px; margin-bottom: 8px; line-height: 1.6;">
                        {formatted_text}<span style="opacity: 0.5;">▌</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            formatted_final = format_markdown_text(response_text)
            message_placeholder.markdown(f"""
            <div class="bot-message">
                <div style="margin-bottom: 8px;">
                    <div class="bot-label">Pulse AI</div>
                </div>
                <div style="margin-top: 10px; margin-bottom: 8px; line-height: 1.6;">
                    {formatted_final}
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.7rem; color: #6b7280;">{formatted_time}{ttft_display}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            message_data['streamed'] = True
        else:
            formatted_text = format_markdown_text(response_text)
            st.markdown(f"""
            <div class="bot-message">
                <div style="margin-bottom: 8px;">
                    <div class="bot-label">Pulse AI</div>
                </div>
                <div style="margin-top: 10px; margin-bottom: 8px; line-height: 1.6;">
                    {formatted_text}
                </div>
                <div style="text-align: right;">
                    <span style="font-size: 0.7rem; color: #6b7280;">{formatted_time}{ttft_display}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # PHASE 2: Show SQL query immediately after insights
        if message_data.get('sql_query') and message_data.get('sql_query') not in ["Error occurred", "Error in SQL generation", "", "NO_CONTEXT_ERROR"]:
            with st.expander(" Generated T-SQL Query", expanded=False):
                st.code(message_data.get('sql_query'), language='sql')

        # PHASE 3: Show data table immediately after SQL
        dataframe = pd.DataFrame()
        raw_dataframe = message_data.get('dataframe')
        
        if raw_dataframe is not None:
            if isinstance(raw_dataframe, pd.DataFrame):
                dataframe = raw_dataframe
            elif isinstance(raw_dataframe, str) and raw_dataframe.strip():
                dataframe = parse_dataframe_from_string(raw_dataframe)
            elif isinstance(raw_dataframe, (list, dict)):
                try:
                    dataframe = pd.DataFrame(raw_dataframe)
                except Exception as e:
                    logger.warning(f"Could not convert parsed data to DataFrame: {e}")
                    dataframe = pd.DataFrame()

        if not dataframe.empty:
            with st.expander(" Query Results", expanded=True):
                if len(dataframe) > 20:
                    st.dataframe(dataframe.head(20), use_container_width=True, hide_index=True)
                    st.caption(f"Showing first 20 rows of {len(dataframe)} total results")
                else:
                    st.dataframe(dataframe, use_container_width=True, hide_index=True)

                # # Download button
                # csv = dataframe.to_csv(index=False)
                # st.download_button(
                #     label="📥 Download CSV",
                #     data=csv,
                #     file_name=f"synapse_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                #     mime="text/csv",
                #     key=f"download_{message_index}"
                # )

        chart_code = message_data.get('chart_code')
        if chart_code and chart_code.strip() and not chart_code.startswith("#"):
            with st.expander(" Generated Chart", expanded=True):
                try:
                    if not dataframe.empty:
                        plt.rcParams['figure.figsize'] = [3.6, 3.5]
                        plt.rcParams['figure.dpi'] = 100
                        exec_scope = {
                            'df': dataframe,
                            'pd': pd,
                            'plt': plt,
                            'np': np
                        }
                        exec(chart_code, exec_scope)
                        
                        fig = exec_scope.get('fig')
                        
                        if fig:
                            fig.tight_layout(pad=1.2)
                            col1, col2, col3 = st.columns([2, 8, 2])
                            with col2:
                                st.pyplot(fig, use_container_width=True)
                        else:
                            st.warning("Chart code was executed but did not produce a 'fig' object to display.")
                            st.code(chart_code, language='python')
                    else:
                        st.warning("No data available for chart generation.")
                except Exception as e:
                    st.error(f"An error occurred while displaying the chart: {e}")
                    st.code(chart_code, language='python')

        # ✅ PHASE 5: Show follow-ups with context-aware labeling
        if message_index == len(st.session_state.chat_history) - 1:
            follow_up_questions = message_data.get('follow_up_questions', [])
            if follow_up_questions:
                # Check if this is a greeting response
                user_question = ""
                if message_index > 0:
                    prev_message = st.session_state.chat_history[message_index - 1]
                    if prev_message.get('type') == 'user':
                        user_question = prev_message.get('content', '').lower()
                
                # Detect if previous message was a greeting
                greeting_keywords = ['hi', 'hello', 'hey', 'capabilities', 'what can you do', 
                                    'help me', 'how to start', 'introduce', 'about you']
                is_greeting_context = any(keyword in user_question for keyword in greeting_keywords)
                
                # Also check if response has no SQL/data (typical of greetings)
                has_data = not message_data.get('dataframe', pd.DataFrame()).empty
                has_sql = bool(message_data.get('sql_query', '').strip())
                
                if not has_data and not has_sql:
                    is_greeting_context = True
                
                st.session_state.current_followup_questions = follow_up_questions
                st.session_state.is_greeting_followup = is_greeting_context

def process_user_input(user_input):
    
    """Process user input and update chat history with user isolation."""
    logger.info(f"User {st.session_state.user_id} action: Submitted query - {user_input[:100]}...")

    # Hide welcome message after first input
    st.session_state.show_welcome = False

    # Check if this is the first message in the session
    is_first_message = not st.session_state.first_message_sent

    # Capture exact timestamp when user sends message
    user_timestamp = datetime.now().isoformat()
    
    # Add user message to chat
    user_message = {
        'type': 'user',
        'content': user_input,
        'timestamp': user_timestamp
    }
    st.session_state.chat_history.append(user_message)

    # Save user message with user ID
    st.session_state.session_manager.save_message(
        st.session_state.current_session_id, 
        st.session_state.user_id,
        'user', 
        user_input
    )

    # Update session name if this is the first message
    if is_first_message:
        st.session_state.session_manager.update_session_name(
            st.session_state.current_session_id, 
            st.session_state.user_id,
            user_input
        )
        st.session_state.first_message_sent = True
        logger.info(f"Session name updated for first message: {user_input[:50]}")

    with st.spinner(st.session_state.agent.response_manager.get_processing_message()):
        response_container = st.empty()
        sql_container = st.empty()
        data_container = st.empty()
        chart_container = st.empty()
        followup_container = st.empty()
        
        result = st.session_state.agent.process_query(
            user_input, 
            st.session_state.current_session_id,
            st.session_state.current_thread_id
        )

    # ✅ Capture timestamp when bot response is ready
    bot_timestamp = datetime.now().isoformat()
    
    # Calculate TTFT
    ttft = (datetime.fromisoformat(bot_timestamp) - datetime.fromisoformat(user_timestamp)).total_seconds()
    logger.info(f"TTFT: {ttft:.3f}s")

    # Store bot response with all data
    bot_message = {
        'type': 'bot',
        'content': result['response'],
        'response': result['response'],
        'sql_query': result.get('sql_query', ''),
        'dataframe': result.get('dataframe', pd.DataFrame()),
        'chart_code': result.get('chart_code', ''),
        'follow_up_questions': result.get('follow_up_questions', []),
        'timestamp': bot_timestamp  
    }
    
    st.session_state.chat_history.append(bot_message)

    # Save bot message with complete data
    st.session_state.session_manager.save_message(
        st.session_state.current_session_id,
        st.session_state.user_id,
        'bot',
        result['response'],
        result.get('sql_query'),
        result.get('dataframe'),
        result.get('chart_code'),
        result.get('follow_up_questions')
    )

    # Cache the complete session state for fast loading
    st.session_state.session_manager.cache_session_state(
        st.session_state.current_session_id,
        st.session_state.user_id,
        st.session_state.chat_history,
        st.session_state.agent.conversation_context,
        st.session_state.agent.query_history
    )
    
    st.rerun()


def sidebar_interface():
    """Sidebar with session management and logo with user isolation."""
    with st.sidebar:
        logo_path = "D:/datachat-ai/frontend/components/Coca-Cola_logo.svg.png" 
        logo_base64 = load_logo_base64(logo_path)

        if logo_base64:
            st.markdown(f"""
            <div class="sidebar-logo">
                <img src="{logo_base64}" alt="YASH Technologies">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sidebar-logo">
                <div class="sidebar-logo-text">YASH Technologies</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Chat Sessions")

        # New chat button
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("New Chat", width='stretch'):
                logger.info(f"User {st.session_state.user_id} action: Started new chat session")
                thread_id = str(uuid.uuid4())
                new_session_id = st.session_state.session_manager.create_session(
                    st.session_state.user_id, thread_id
                )
                st.session_state.current_session_id = new_session_id
                st.session_state.current_thread_id = thread_id
                st.session_state.chat_history = []
                st.session_state.show_welcome = True
                st.session_state.current_followup_questions = []
                st.session_state.first_message_sent = False
                # Reset agent context
                st.session_state.agent.conversation_context = {}
                st.session_state.agent.query_history = []
                # Clear cache for this session
                if new_session_id in st.session_state.loaded_sessions_cache:
                    del st.session_state.loaded_sessions_cache[new_session_id]
                st.rerun()

        st.divider()

        # Display sessions for current user only
        sessions = st.session_state.session_manager.get_all_sessions(st.session_state.user_id)

        if sessions:
            st.markdown("#### Recent Conversations")
            for session in sessions[:10]:
                session_name = session['name']
                session_id = session['session_id']
                thread_id = session['thread_id']

                col1, col2 = st.columns([4, 1])

                with col1:
                    is_active = session_id == st.session_state.current_session_id
                    button_label = f"{'✓ ' if is_active else ''}{session_name}"

                    if st.button(button_label, key=f"session_{session_id}", width='stretch'):
                        logger.info(f"User {st.session_state.user_id} action: Switched to session {session_id} - {session_name}")
                        
                        # Check if session is already loaded in cache
                        if session_id in st.session_state.loaded_sessions_cache:
                            logger.info(f"Loading session from memory cache: {session_id}")
                            cached_data = st.session_state.loaded_sessions_cache[session_id]
                            st.session_state.chat_history = cached_data['chat_history']
                            st.session_state.agent.conversation_context = cached_data['agent_context']
                            st.session_state.agent.query_history = cached_data['query_history']
                        else:
                            # Try to load from database cache first
                            cached_session = st.session_state.session_manager.load_cached_session(
                                session_id, st.session_state.user_id
                            )
                            
                            if cached_session:
                                logger.info(f"Loading session from database cache: {session_id}")
                                st.session_state.chat_history = cached_session['chat_history']
                                st.session_state.agent.conversation_context = cached_session['agent_context']
                                st.session_state.agent.query_history = cached_session['query_history']
                            else:
                                # Load from message history (slower fallback)
                                logger.info(f"Loading session from message history: {session_id}")
                                history = st.session_state.session_manager.get_session_history(
                                    session_id, st.session_state.user_id
                                )
                                st.session_state.chat_history = history
                                st.session_state.agent.conversation_context = {}
                                st.session_state.agent.query_history = []

                            # Cache in memory for faster future access
                            st.session_state.loaded_sessions_cache[session_id] = {
                                'chat_history': st.session_state.chat_history,
                                'agent_context': st.session_state.agent.conversation_context,
                                'query_history': st.session_state.agent.query_history
                            }

                        st.session_state.current_session_id = session_id
                        st.session_state.current_thread_id = thread_id
                        st.session_state.show_welcome = False if st.session_state.chat_history else True
                        st.session_state.current_followup_questions = []
                        st.session_state.first_message_sent = len(st.session_state.chat_history) > 0
                        st.rerun()

                with col2:
                    if st.button("🗑️", key=f"delete_{session_id}", help=f"Delete {session_name}"):
                        logger.info(f"User {st.session_state.user_id} action: Deleted session {session_id} - {session_name}")
                        st.session_state.session_manager.delete_session(session_id, st.session_state.user_id)
                        
                        # Remove from memory cache
                        if session_id in st.session_state.loaded_sessions_cache:
                            del st.session_state.loaded_sessions_cache[session_id]
                        
                        if session_id == st.session_state.current_session_id:
                            st.session_state.current_session_id = None
                            st.session_state.current_thread_id = None
                            st.session_state.chat_history = []
                            st.session_state.show_welcome = True
                            st.session_state.current_followup_questions = []
                            st.session_state.first_message_sent = False
                        st.rerun()
        else:
            st.info("No chat sessions yet. Start a new conversation!")

        st.divider()

        # Database connection status
        try:
            conn_manager = SynapseConnectionManager()
            conn_manager.get_connection()
        except Exception as e:
            st.error("🔴 Database Connection Error")
            st.caption(f"Error: {str(e)[:50]}...")
            logger.error(f"Database connection error in sidebar: {e}")

def main_chat_interface():
    
    """Main chat interface with user session isolation."""
    # Initialize session if needed
    if st.session_state.current_session_id is None:
        thread_id = str(uuid.uuid4())
        st.session_state.current_session_id = st.session_state.session_manager.create_session(
            st.session_state.user_id, thread_id
        )
        st.session_state.current_thread_id = thread_id
        logger.info(f"New session initialized for user {st.session_state.user_id}: {st.session_state.current_session_id}")
    
    # Load session data if chat history is empty but should exist
    if (not st.session_state.chat_history and 
        st.session_state.current_session_id and 
        st.session_state.current_session_id not in st.session_state.loaded_sessions_cache):
        
        cached_session = st.session_state.session_manager.load_cached_session(
            st.session_state.current_session_id, st.session_state.user_id
        )
        
        if cached_session:
            st.session_state.chat_history = cached_session['chat_history']
            st.session_state.agent.conversation_context = cached_session['agent_context']
            st.session_state.agent.query_history = cached_session['query_history']
            
            # Cache in memory
            st.session_state.loaded_sessions_cache[st.session_state.current_session_id] = cached_session

    # Display header
    display_header()

    # Show welcome message if it's a new chat
    if st.session_state.show_welcome and not st.session_state.chat_history:
        display_welcome_message()

    # Chat container
    chat_container = st.container()

    with chat_container:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if message['type'] == 'user':
                display_chat_message(message['content'], is_user=True, message_index=i)
            else:
                display_chat_message(message, is_user=False, message_index=i)

    if st.session_state.current_followup_questions:
        is_greeting = st.session_state.get('is_greeting_followup', False)
        display_followup_questions(
            st.session_state.current_followup_questions,
            is_greeting_response=is_greeting  
        )

    # Input area
    st.markdown("---")

    user_input = st.chat_input("💬 Ask Pulse AI about your data...", key="main_input")

    if user_input:
        # Clear follow-up questions when new input is entered
        st.session_state.current_followup_questions = []
        process_user_input(user_input)
