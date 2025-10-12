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

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages



# State definition for LangGraph with context and chart tracking
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_question: str
    corrected_question: str
    sql_query: str
    query_result: pd.DataFrame
    final_response: str
    chart_code: str
    follow_up_questions: List[str]
    session_id: str
    thread_id: str
    error_message: str
    conversation_context: Dict[str, Any]
    previous_queries: List[str]
    
    # NEW: Intent classification fields
    primary_intent: str
    contains_greeting: bool
    contains_data_request: bool
    intent_reasoning: str
    
    # Keep existing fields
    is_valid_query: bool
    is_greeting: bool
    needs_clarification: bool
    clarification_prompt: str
    best_chart_type: str
    user_intent: str
    visualization_suggestion: str
    
    # ✅ NEW: Parallel processing control fields
    sql_retry_count: int
    sql_validation_error: str
    response_ready: bool
    chart_ready: bool
    followup_ready: bool
    parallel_tasks_completed: int
