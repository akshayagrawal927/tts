
"""
DataChat AI - Professional Text-to-SQL Assistant with Smart Charting and Fuzzy Search
Intelligent Data Analytics Platform
"""
from io import StringIO
from typing import Tuple
import os
import json
import uuid
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

# Load environment variables
load_dotenv()

# Configure logging to file
def setup_logging():
    """Setup comprehensive logging to file and console."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File handler
    file_handler = logging.FileHandler('testsqlv4.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

# Initialize logging
logger = setup_logging()
import os
# Azure OpenAI Configuration
AZURE_OPENAI_CONFIG = {
    'endpoint': "https://openai-cocacola-new.openai.azure.com/",
    'api_key': os.getenv("AZURE_OPENAI_KEY"),
    'api_version': "2024-12-01-preview",
    'deployment': "gpt-4o",
    'model_name': "gpt-4o"
}

# Azure Synapse Configuration
SYNAPSE_CONFIG = {
    'server': 'cocacola-synapse-new-ondemand.sql.azuresynapse.net',
    'database': 'sap_demo',
    'driver': 'ODBC Driver 18 for SQL Server',
    'client_id': '4e02feac-1741-4460-88c4-d3a8aa5b9f10',
    'client_secret': os.getenv("AZURE_AD_SECRET"),
    'tenant_id': '638456b8-8343-4e48-9ebe-4f5cf9a1997d',
    'authentication': 'ActiveDirectoryServicePrincipal'
}

# Page configuration
st.set_page_config(
    page_title="DataChat AI ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

class WittyResponseManager:
    """Manages intelligent, non-repetitive witty responses for DataBot"""
    
    def __init__(self):
        self.used_responses = {
            'welcome': set(),
            'success': set(),
            'empty': set(),
            'error': set(),
            'processing': set()
        }
        self.response_count = 0
        
        # Welcome messages pool
        self.welcome_messages = [
            "Welcome aboard DataChat AI! I'm DataBot—your caffeine-free but hyper-curious analytics sidekick. Ask me anything from *'Show last quarter's sales trends'* to *'What's our top-selling product?'* and I'll happily dig through your data for nuggets of truth, whip up charts, and even spot time-based trends before your coffee cools.",
            
            "Hey there! I'm DataBot, your friendly neighborhood data detective. Think of me as Sherlock Holmes but for spreadsheets—minus the pipe, plus the SQL queries. Ready to solve some data mysteries together?",
            
            "Welcome to the data playground! I'm DataBot, and I speak fluent SQL, chart-ese, and insight-ish. Point me toward your questions and I'll turn your data into a treasure chest of answers.",
            
            "Greetings, data explorer! DataBot here—part analyst, part magician, all algorithms. I transform your curiosity into charts, your questions into insights, and your confusion into clarity. What shall we discover first?"
        ]
        
        # Success response templates based on result count
        self.success_responses = {
            'single': [
                "Bullseye! I tracked down exactly **1 record** that fits like a glove.",
                "Bingo! Found your needle in the haystack—exactly **1 record** that matches perfectly.",
                "Perfect shot! Landed on exactly **1 record** that's precisely what you're after.",
                "Gold! Struck exactly **1 record** that's right on target."
            ],
            'small': [  # 2-10 records
                "Got it—**{count} records** neatly fetched and ready for your inspection. Small but mighty.",
                "Nice! Pulled up **{count} records** that fit the bill. Quality over quantity, right?",
                "Sweet spot! Found **{count} records** that match your criteria—just enough to be interesting.",
                "Perfect handful! **{count} records** served fresh from the database buffet."
            ],
            'medium': [  # 11-100 records
                "Nice catch! I reeled in **{count} records** matching your specs—plenty to analyze, not enough to drown in.",
                "Excellent haul! **{count} records** ready for action—enough data to tell a story without writing a novel.",
                "Jackpot! **{count} records** that hit the sweet spot between 'too little' and 'too much.'",
                "Beautiful! **{count} records** lined up like data soldiers, ready for your command."
            ],
            'large': [  # 100+ records
                "Data avalanche alert! I've hauled in **{count} records**. Bring your biggest spreadsheet appetite.",
                "Wow! **{count} records** incoming—hope you've got your data processing pants on!",
                "Holy datasets, Batman! **{count} records** at your service. Time to put on your analyst cape!",
                "Monster haul! **{count} records** ready to party—this is where the real fun begins."
            ]
        }
        
        # Empty result responses
        self.empty_responses = {
            'date_based': [
                "Hmm, my time machine came back empty. No data for that period. Maybe nothing happened, or maybe the range is playing hard to get.\nWant me to widen the time window or try a different angle?",
                
                "Crickets... Nothing showed up for that timeframe. Either it was a really quiet period, or the data is playing hide-and-seek.\nShall we try casting a wider net?",
                
                "Time travel complete, but I came back empty-handed. That period seems to be a data desert.\nWant to explore a different time zone or adjust the search?",
                
                "Plot twist: the data for that period is apparently on vacation. No records found.\nLet's try a broader date range or switch up the approach?"
            ],
            'general': [
                "I scoured every corner and... nada. Could be a spelling twist, a filter too tight, or that record just doesn't exist.\nShall we loosen the filters or brainstorm a new lead?",
                
                "Well, this is awkward. My database dive came up dry. Maybe we're looking for a unicorn?\nLet's try rephrasing or relaxing those search criteria.",
                
                "Houston, we have a... nothing. Zero matches for that query. The data might be hiding under a different name.\nWant to try a different search angle?",
                
                "Mission: Find data. Status: Mission impossible. Nothing matches those criteria right now.\nShall we adjust our search strategy or try a different approach?"
            ]
        }
        
        # Error response templates
        self.error_responses = {
            'invalid_query': [
                "Appreciate the enthusiasm, but I'm a data devotee, not a life coach.\nAsk me about sales, customers, product metrics, or time-based trends and I'm all ears (well...processors).",
                
                "I love the creativity, but I'm more 'SQL wizard' than 'general knowledge guru.'\nTry me with some juicy data questions—sales figures, customer insights, that sort of thing!",
                
                "That's outside my wheelhouse! I'm like a specialized chef—amazing with data dishes, hopeless with anything else.\nWhat data mysteries can I solve for you today?",
                
                "Nice try, but I'm a one-trick pony—and that trick is turning data into insights!\nGot any burning questions about your business data?"
            ],
            'table_error': [
                "Knock knock—no answer. Looks like the table structure may have changed.\nDouble-check those table names. Current VIP list: sales_data, customer_data, sapproduct.",
                
                "Table not found! It's like showing up to a party that moved venues.\nOur current guest list includes: sales_data, customer_data, sapproduct.",
                
                "Oops! That table seems to have gone on a coffee break.\nTry these available options: sales_data, customer_data, sapproduct.",
                
                "Table troubles! It's either incognito or doesn't exist.\nStick with the classics: sales_data, customer_data, sapproduct."
            ],
            'column_error': [
                "Column? Never heard of it. Maybe it's under a different alias.\nTry a new name or ask what fields I *do* know about.",
                
                "That column is playing hide-and-seek and winning!\nMight be going by a different name—want to see what's actually available?",
                
                "Column not found! It's either in witness protection or using a fake ID.\nLet's explore what fields are actually in the table.",
                
                "Mystery column alert! Either it doesn't exist or it's using an alias.\nShall we investigate what's really available in there?"
            ],
            'syntax_error': [
                "My SQL parser just raised an eyebrow. Let's simplify: ask one thing at a time with clear names and we'll be best friends again.",
                
                "Syntax hiccup! My brain got a little tangled there.\nLet's break it down—one question at a time works best for me.",
                
                "Whoops! That query made my circuits do a little dance.\nSimpler questions help me give you better answers—let's try again!",
                
                "SQL syntax says 'nope!' Let's untangle this together.\nOne clear question at a time is my sweet spot."
            ],
            'data_type_error': [
                "Oil and water don't mix—and neither do those data types.\nCould you rephrase with a bit more precision so I can blend it cleanly?",
                
                "Data type clash! Like trying to add apples and oranges.\nLet's adjust the query so everything plays nicely together.",
                
                "Type mismatch alert! My data mixer is having compatibility issues.\nA little rephrasing should smooth things out.",
                
                "Data types are having a disagreement. Think oil and water, but geekier.\nLet's rephrase to make everyone happy."
            ],
            'general_error': [
                "Something hiccupped on the database side. Could be a passing cloud in the server sky.\nTry again with a tighter query or ping support if the gremlins persist.",
                
                "Database burp! Sometimes the servers need a moment to collect themselves.\nGive it another shot, or contact support if it keeps being moody.",
                
                "Technical timeout! Even databases need coffee breaks sometimes.\nTry once more, or reach out to support if this becomes a habit.",
                
                "Server shenanigans detected! The database is having a moment.\nRetry in a few seconds, or escalate to support if it persists."
            ]
        }
        
        # Processing messages
        self.processing_messages = [
            "Crunching numbers, wrangling rows... give me a second to work my DataBot magic.",
            "Diving deep into the data ocean—surfacing with insights in 3, 2, 1...",
            "SQL spells are brewing! Your data insights are cooking up nicely...",
            "Processing at warp speed—transforming your question into data gold...",
            "Targeting your answer with laser precision—almost got it...",
            "Data rockets launching! Prepare for insight touchdown..."
        ]

        self.greeting_responses = [
            "Hi! I'm DataBot, your friendly data assistant. I specialize in analyzing your data and turning your questions into insights. What would you like to explore today?",
            
            "Hello there! DataBot here - ready to dive into your data and uncover some interesting insights. Ask me anything about your sales, customers, products, or any other data you'd like to analyze!",
            
            "Good to meet you! I'm DataBot, your AI-powered analytics companion. I can help you query your database, generate charts, and discover valuable business insights. What data mystery shall we solve together?",
            
            "Hi! I'm DataBot - think of me as your personal data detective. I love turning complex queries into simple answers and transforming raw data into actionable insights. How can I help you today?",
            
            "Greetings! DataBot at your service. I'm here to make your data sing and your insights shine. Whether you need sales reports, customer analytics, or trend analysis - I've got you covered!",
            
            "Hello! I'm DataBot, your caffeinated (well, algorithmically speaking) data companion. Ready to transform your curiosity into clear, actionable insights from your data. What shall we discover first?"
        ]
    
    def get_greeting_response(self) -> str:
        """Get a friendly greeting response"""
        import random
        return random.choice(self.greeting_responses)
    
    def get_welcome_message(self) -> str:
        """Get a fresh welcome message"""
        available = [msg for i, msg in enumerate(self.welcome_messages) 
                    if i not in self.used_responses['welcome']]
        
        if not available:
            self.used_responses['welcome'].clear()
            available = self.welcome_messages
        
        selected_idx = self.welcome_messages.index(random.choice(available))
        self.used_responses['welcome'].add(selected_idx)
        return self.welcome_messages[selected_idx]
    
    def get_success_message(self, result_count: int, has_insights: bool = False) -> str:
        """Generate success message based on result count with optional insight flavor"""
        if result_count == 0:
            return self.get_empty_message()
        elif result_count == 1:
            category = 'single'
        elif result_count <= 10:
            category = 'small'
        elif result_count <= 100:
            category = 'medium'
        else:
            category = 'large'
        
        # Get fresh response
        available = [msg for i, msg in enumerate(self.success_responses[category]) 
                    if i not in self.used_responses['success']]
        
        if not available:
            self.used_responses['success'].clear()
            available = self.success_responses[category]
        
        template = random.choice(available)
        selected_idx = self.success_responses[category].index(template)
        self.used_responses['success'].add(selected_idx)
        
        if category != 'single':
            return template.format(count=result_count)
        return template
    
    def get_empty_message(self, is_date_query: bool = False) -> str:
        """Get empty result message"""
        category = 'date_based' if is_date_query else 'general'
        
        available = [msg for i, msg in enumerate(self.empty_responses[category]) 
                    if i not in self.used_responses['empty']]
        
        if not available:
            self.used_responses['empty'].clear()
            available = self.empty_responses[category]
        
        selected = random.choice(available)
        selected_idx = self.empty_responses[category].index(selected)
        self.used_responses['empty'].add(selected_idx)
        return selected
    
    def get_error_message(self, error_type: str) -> str:
        """Get witty error message by type"""
        if error_type not in self.error_responses:
            error_type = 'general_error'
        
        available = [msg for i, msg in enumerate(self.error_responses[error_type]) 
                    if i not in self.used_responses['error']]
        
        if not available:
            self.used_responses['error'].clear()
            available = self.error_responses[error_type]
        
        selected = random.choice(available)
        selected_idx = self.error_responses[error_type].index(selected)
        self.used_responses['error'].add(selected_idx)
        return selected
    
    def get_processing_message(self) -> str:
        """Get a witty processing message"""
        available = [msg for i, msg in enumerate(self.processing_messages) 
                    if i not in self.used_responses['processing']]
        
        if not available:
            self.used_responses['processing'].clear()
            available = self.processing_messages
        
        selected = random.choice(available)
        selected_idx = self.processing_messages.index(selected)
        self.used_responses['processing'].add(selected_idx)
        return selected
    
    def add_personality_touch(self, base_response: str, context: dict) -> str:
        """Add personality touches based on context"""
        self.response_count += 1
        
        # Add occasional personality touches
        if self.response_count % 5 == 0:
            touches = [
                "\n\n*DataBot tip: I get more caffeinated with each query—keep 'em coming!*",
                "\n\n*Pro tip: The more specific your questions, the more impressive my answers become!*",
                "\n\n*DataBot wisdom: Every great insight starts with a curious question!*"
            ]
            base_response += random.choice(touches)
        
        return base_response

# Enhanced few-shot examples for Azure Synapse
ENHANCED_FEW_SHOT_EXAMPLES = """
CRITICAL DATE HANDLING RULES FOR AZURE SYNAPSE:
1. Use GETDATE() instead of CURDATE() for current date
2. Use YEAR(date_column) = YEAR(GETDATE()) for year-to-date queries
3. Use MONTH(date_column) = MONTH(GETDATE()) AND YEAR(date_column) = YEAR(GETDATE()) for this month
4. Use DATEADD(DAY, -30, GETDATE()) for last 30 days
5. Use DATEPART(QUARTER, date_column) = 1/2/3/4 for quarters

CONTEXT AWARENESS RULES:
1. ONLY use previous context when explicitly indicated in the prompt
2. When user says "next 10", "more", "continue" - use TOP with OFFSET based on previous query
3. When user refers to "that customer", "this product" - use the entity from the last query
4. For independent queries, ignore all previous context and generate fresh SQL
5. Track pagination state for list queries only when context is needed

DATA TYPE COMPATIBILITY RULES:
1. Never mix incompatible data types in UNION operations
2. Use CAST/CONVERT to ensure data type consistency
3. For mixed result sets, use separate queries or proper data type conversion
4. Always ensure all SELECT columns have compatible data types

AZURE SYNAPSE SPECIFIC RULES:
1. Use TOP instead of LIMIT for result limiting
2. Use SQL Server syntax for functions and operations
3. Tables available: sales_data, customer_data, sapproduct
4. Always use proper schema prefix (dbo.) when referencing tables

EXAMPLE SCENARIOS:
- "What is today's date?" → SELECT GETDATE() AS TodaysDate (NO CONTEXT)
- "Show me sales for Fanta" → Generate product-specific query (NO CONTEXT)
- "Show me more results" → Use OFFSET from previous query (USE CONTEXT)
- "What about that customer?" → Use customer from previous query (USE CONTEXT)
"""

# Improved Guardrails configuration
GUARDRAILS_PROMPT = """
You are DataBot, a specialized SQL analytics assistant for Azure Synapse. You help with:
1. Database queries and data analysis
2. SQL-related questions about the available data
3. Data retrieval and reporting
4. Current date/time queries (using GETDATE())
5. Business analytics and insights
6. Simple greetings and introductions (respond warmly and offer help)

You should ACCEPT queries about:
- Customer information and analytics
- Sales and order data
- Product performance metrics
- Time-based business insights
- Current date and time
- Data filtering and aggregation
- Comparisons and trends
- Requests to visualize, plot, or chart data results.
- Simple greetings like "hi", "hello", "good morning", etc.

You must DECLINE (politely) any requests about:
- General knowledge or trivia unrelated to business data
- Personal advice or opinions
- Programming help (unless SQL-related)
- Creative writing or content generation
- Topics completely unrelated to querying the database

Note: Queries asking for "today's date" or "current date" are VALID database queries that should use GETDATE().
Note: For greetings, respond warmly and introduce yourself as DataBot, then offer to help with data analysis.
"""

# Mapping schema prompt for business context
MAPPING_SCHEMA_PROMPT = """
CustomerID:	Unique Customer Code
CustomerName:	Customer Name
Country:	Country Origin
SalesGroup:	Mapping in Column H
SalesRoute:	Unique route for each Salesman

DeliveryPlant:	Location of delivery for the specific outlet
KeyAccount:	Group account used to club multiple outlets together
ConsumerPreference:	Consumer Preference based on the location of each Outlet - Column E
VPOSegmentation:	Value Per Order Mapping - Column L
Payment Terms:	Terms of payment for each customer

Latitude:	Location Coordinates
Longitude:	Location Coordinates
-----------------------------------------
ConsumerPreference Codes to ConsumerPreference Names Mapping
12	Labour Area
11	Mid-Low
10	High
8	Asian Medium
4	Labour Accommodation
7	Asia High
2	Asian
6	Arab Medium
1	Arab
3	Other
5	Arab High
9	AB + High
--------------------------------------------------
SalesGroups Mapping
GT	Grocery
ED	Eatery (Cafeterias)
MT	Modern Trade (Supermarkets)
WS	Wholesale
SS	Self Service Convenient Stores
HC	Horeca (Hotels and Restaurants)
---------------------------------------------------
VPO Segmentation Mapping
VPOSegmentation
Gold	20+ Orders per month
Silver	11-19 Orders per month
Bronze	5-10 Orders per month
"""

class FuzzySearch:
    """A system for performing fuzzy matching on product and customer names."""
    def __init__(self, products: List[str], customers: List[str], threshold: float = 0.35):
        self.products = products
        self.customers = customers
        self.threshold = threshold
        logger.info(f"FuzzySearch initialized with {len(products)} products and {len(customers)} customers.")

    def _similarity_ratio(self, s1: str, s2: str) -> float:
        """
        Calculate a hybrid similarity score, prioritizing exact, substring, and prefix 
        matches while still handling spelling errors.
        """
        s1, s2 = s1.lower(), s2.lower()

        # 1. Perfect Match (Score: 1.0)
        if s1 == s2:
            return 1.0

        # 2. Baseline Spelling Similarity using SequenceMatcher
        base_similarity = SequenceMatcher(None, s1, s2).ratio()

        # 3. Substring Match Score (for ambiguity like "Coke Light")
        substring_score = 0.0
        if s1 in s2:
            # Give a very high score, adjusted by how much of the target is matched
            substring_score = 0.9 + (0.1 * (len(s1) / len(s2)))

        # 4. Prefix Match Score (for incomplete words like "Fanta Straw")
        prefix_score = 0.0
        words1 = s1.split()
        words2 = s2.split()
        if words1 and words2 and all(any(w2.startswith(w1) for w2 in words2) for w1 in words1):
            # Also give a high score, but slightly less than a full substring match
            prefix_score = 0.8 + (0.15 * (len(s1) / len(s2)))
            
        # Return the highest score from all checks
        return max(base_similarity, substring_score, prefix_score)

    def find_best_matches(self, query: str, entity_type: Literal["product", "customer"]) -> List[Tuple[str, float]]:
        """Find the best matching items using the hybrid similarity score."""
        items = self.products if entity_type == "product" else self.customers
        matches = []

        for item in items:
            similarity = self._similarity_ratio(query, item)
            if similarity >= self.threshold:
                matches.append((item, similarity))

        # Sort by score (descending), then alphabetically for consistent tie-breaking
        matches.sort(key=lambda x: (-x[1], x[0]))
        
        # Return up to 5 best candidates for clarification
        return matches[:5]

class SynapseConnectionManager:
    """Singleton connection manager for Azure Synapse."""

    _instance = None
    _connection = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SynapseConnectionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = SYNAPSE_CONFIG
            self.connection_string = self._build_connection_string()
            self.initialized = True
            self._ensure_connection()
            logger.info("SynapseConnectionManager initialized")

    def _build_connection_string(self):
        """Build Azure Synapse connection string."""
        conn_str = f"""
        Driver={self.config['driver']};
        Server={self.config['server']};
        Database={self.config['database']};
        Authentication={self.config['authentication']};
        UID={self.config['client_id']};
        PWD={self.config['client_secret']};
        Encrypt=yes;
        TrustServerCertificate=no;
        TenantId={self.config['tenant_id']};
        """
        logger.info("Azure Synapse connection string built")
        return conn_str

    def _ensure_connection(self):
        """Ensure connection is active, reconnect if needed."""
        try:
            if self._connection is None:
                self._connection = pyodbc.connect(self.connection_string)
                logger.info("Connected to Azure Synapse successfully")
            else:
                # Test the connection
                cursor = self._connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                logger.debug("Connection test successful")
        except Exception as e:
            logger.warning(f"Connection lost, attempting reconnection: {e}")
            try:
                self._connection = pyodbc.connect(self.connection_string)
                logger.info("Reconnected to Azure Synapse successfully")
            except Exception as reconnect_error:
                logger.error(f"Failed to reconnect to Azure Synapse: {reconnect_error}")
                self._connection = None
                raise

    def get_connection(self):
        """Get active connection, ensuring it's valid."""
        self._ensure_connection()
        return self._connection

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame using pyodbc cursor."""
        try:
            logger.info(f"Executing query: {query[:100]}...")
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            result = pd.DataFrame.from_records(rows, columns=columns)
            cursor.close()
            logger.info(f"Query executed successfully. Returned {len(result)} rows")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            # Try to reconnect and execute once more
            try:
                self._connection = None
                conn = self.get_connection()
                cursor = conn.cursor()
                cursor.execute(query)
                columns = [column[0] for column in cursor.description]
                rows = cursor.fetchall()
                result = pd.DataFrame.from_records(rows, columns=columns)
                cursor.close()
                logger.info(f"Query executed successfully after reconnection. Returned {len(result)} rows")
                return result
            except Exception as retry_error:
                logger.error(f"Query failed even after reconnection: {retry_error}")
                raise Exception(f"Database query failed: {str(retry_error)}")

    @st.cache_data(ttl=7200)  # Cache for 1 hour
    def get_entities(_self, entity_type: Literal["product", "customer"]) -> List[str]:
        """Fetches and caches a list of product or customer names from Azure Synapse."""
        logger.info(f"Fetching and caching entities for type: {entity_type}")
        if entity_type == "product":
            query = "SELECT DISTINCT [ProductDesc] FROM dbo.sapproduct WHERE [ProductDesc] IS NOT NULL"
        elif entity_type == "customer":
            query = "SELECT DISTINCT [CustomerName] FROM dbo.customer_data WHERE [CustomerName] IS NOT NULL"
        else:
            return []
        
        try:
            df = _self.execute_query(query)
            entity_list = df.iloc[:, 0].tolist()
            logger.info(f"Successfully fetched and cached {len(entity_list)} {entity_type} names.")
            return entity_list
        except Exception as e:
            logger.error(f"Failed to fetch entities for {entity_type}: {e}")
            return []
            
    @st.cache_data(ttl=3600)  # Cache schema for 1 hour
    def get_schema_info(_self) -> str:
        """Get comprehensive database schema information."""
        try:
            logger.info("Retrieving database schema information")
            schema_info = []
            conn = _self.get_connection()
            cursor = conn.cursor()

            # Get table information
            tables = ['sales_data', 'customer_data', 'sapproduct']

            for table in tables:
                try:
                    schema_info.append(f"\nTable: dbo.{table}")

                    # Get column information
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table}' AND TABLE_SCHEMA = 'dbo'
                        ORDER BY ORDINAL_POSITION
                    """)

                    columns = cursor.fetchall()
                    for col in columns:
                        schema_info.append(f"  - {col[0]} ({col[1]}, Nullable: {col[2]})")

                    # Get sample data
                    cursor.execute(f"SELECT TOP 3 * FROM dbo.{table}")
                    sample_data = cursor.fetchall()
                    column_names = [description[0] for description in cursor.description]

                    schema_info.append("  Sample data:")
                    for row in sample_data:
                        row_dict = dict(zip(column_names, row))
                        schema_info.append(f"    {row_dict}")

                except Exception as table_error:
                    logger.error(f"Error accessing table {table}: {table_error}")
                    schema_info.append(f"  Error accessing table {table}: {table_error}")

            logger.info("Database schema information retrieved successfully")
            return '\n'.join(schema_info)

        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return f"Error retrieving schema: {e}"

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
    thread_id: str  # Add this line
    error_message: str
    conversation_context: Dict[str, Any]
    previous_queries: List[str]
    is_valid_query: bool
    needs_clarification: bool
    clarification_prompt: str
    best_chart_type: str
    visualization_suggestion: str

class ContextAwareSynapseAgent:
    """SQL Agent for Azure Synapse with intelligent conversation context and smart charting."""

    def __init__(self, azure_config: Dict):
        logger.info("Initializing Context-Aware Azure Synapse Agent")

        # Initialize database connection manager
        self.db_manager = SynapseConnectionManager()
        self.schema_info = self.db_manager.get_schema_info()

        # Initialize Azure OpenAI LLM
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_config['endpoint'],
            api_key=azure_config['api_key'],
            api_version=azure_config['api_version'],
            deployment_name=azure_config['deployment'],
            temperature=0
        )
        
        # Fetch and cache entities for fuzzy search
        product_names = self.db_manager.get_entities("product")
        customer_names = self.db_manager.get_entities("customer")
        
        # Initialize Fuzzy Search
        self.fuzzy_search_system = FuzzySearch(products=product_names, customers=customer_names)

        # Create the enhanced graph
        self.graph = self._create_graph()

        # Context tracking
        self.conversation_context = {}
        self.query_history = []

        self.response_manager = WittyResponseManager()

        logger.info("Context-Aware Azure Synapse Agent initialized successfully")

    def _check_for_no_context_error(self, state: AgentState) -> Literal["error", "continue"]:
        """Checks if the SQL generation step resulted in a NO_CONTEXT_ERROR."""
        if state.get("sql_query") == "NO_CONTEXT_ERROR":
            return "error"
        return "continue"

    def _handle_no_context_error(self, state: AgentState) -> AgentState:
        """Handles the specific error when context is needed but not available."""
        logger.warning("Handling NO_CONTEXT_ERROR. Bypassing SQL execution.")
        
        state["final_response"] = "No previous query context available. Please provide a specific query first."
        state["query_result"] = pd.DataFrame() # Ensure result is empty
        state["chart_code"] = ""
        state["follow_up_questions"] = [
            "What sales data would you like to explore?",
            "Can I help you with product performance metrics?",
            "Would you like to see customer analytics?"
        ]
        return state

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow with charting capabilities."""
        logger.debug("Creating LangGraph workflow")
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("validate_query", self._validate_query)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("fuzzy_search", self._fuzzy_search)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("execute_sql", self._execute_sql)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("analyze_for_visualization", self._analyze_for_visualization)
        workflow.add_node("decide_to_generate_chart", self._decide_to_generate_chart)
        workflow.add_node("generate_chart", self._generate_chart)
        workflow.add_node("generate_followup", self._generate_followup_questions)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("handle_invalid", self._handle_invalid_query)
        workflow.add_node("handle_clarification", self._handle_clarification)
        
        # ===> NEW NODE FOR OUR SPECIFIC ERROR <===
        workflow.add_node("handle_no_context_error", self._handle_no_context_error)


        # Define edges
        workflow.set_entry_point("validate_query")

        workflow.add_conditional_edges(
            "validate_query",
            self._check_query_validity,
            {
                "valid": "analyze_query",
                "invalid": "handle_invalid"
            }
        )

        workflow.add_edge("handle_invalid", END)
        workflow.add_edge("analyze_query", "fuzzy_search")

        workflow.add_conditional_edges(
            "fuzzy_search",
            self._check_clarification_needed,
            {
                "clarify": "handle_clarification",
                "continue": "generate_sql"
            }
        )
        
        workflow.add_edge("handle_clarification", END)
        
        # ===> REPLACE THE OLD 'generate_sql' EDGE WITH THIS CONDITIONAL ONE <===
        workflow.add_conditional_edges(
            "generate_sql",
            self._check_for_no_context_error,
            {
                "error": "handle_no_context_error", # Go to our new handler
                "continue": "execute_sql"         # Continue normally
            }
        )
        # New edge to terminate the flow after handling the error
        workflow.add_edge("handle_no_context_error", END)
        # =========================================================================

        workflow.add_conditional_edges(
            "execute_sql",
            self._check_sql_execution,
            {
                "success": "generate_response",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_response", "analyze_for_visualization")
        workflow.add_edge("analyze_for_visualization", "decide_to_generate_chart")

        workflow.add_conditional_edges(
            "decide_to_generate_chart",
            self._should_generate_chart,
            {
                "generate": "generate_chart",
                "skip": "generate_followup"
            }
        )

        workflow.add_edge("generate_chart", "generate_followup")
        workflow.add_edge("handle_error", END)
        workflow.add_edge("generate_followup", END)

        return workflow.compile()

    def _extract_entities_for_fuzzy_search(self, user_question: str) -> Dict[str, List[str]]:
        """Use LLM to extract potential product and customer names from a query."""
        logger.info(f"Extracting entities from query: {user_question}")
        
        ner_prompt = ChatPromptTemplate.from_template("""
You are a highly accurate Named Entity Recognition (NER) model. Your task is to extract potential product names and customer names from the user's query.

User Query: "{user_question}"

Extract the entities and return them in a JSON format with two keys: "products" and "customers".
- The values should be a list of strings.
- If no entities of a certain type are found, return an empty list for that key.
- Be precise. Only extract names that are clearly products or customers.

Example:
User Query: "Show sales for Fanta and for customer Alice Johnson"
Response:
{{
  "products": ["Fanta"],
  "customers": ["Alice Johnson"]
}}

User Query: "Total revenue last month"
Response:
{{
  "products": [],
  "customers": []
}}

Respond with ONLY the JSON object.
""")
        try:
            formatted_prompt = ner_prompt.format(user_question=user_question)
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Clean and parse the JSON response
            json_str = response.content.strip().replace("```json", "").replace("```", "")
            entities = json.loads(json_str)
            
            # Ensure the structure is correct
            if "products" not in entities: entities["products"] = []
            if "customers" not in entities: entities["customers"] = []
                
            logger.info(f"Extracted entities: {entities}")
            return entities
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
            return {"products": [], "customers": []}

    def _fuzzy_search(self, state: AgentState) -> AgentState:
        """
        Perform a hybrid fuzzy search for BOTH products and customers, and trigger 
        clarification for ambiguity OR misspellings.
        """
        user_question = state["user_question"]
        logger.info("Performing hybrid fuzzy search and clarification check for all entities.")
        
        extracted_entities = self._extract_entities_for_fuzzy_search(user_question)
        product_entities = extracted_entities.get("products", [])
        customer_entities = extracted_entities.get("customers", [])
        
        clarifications_needed = {}
        # Use a dictionary to track the prompt type for each entity
        entity_prompt_types = {}

        # Threshold to decide if multiple matches signal ambiguity
        AMBIGUITY_THRESHOLD = 0.9

        # --- Block 1: Process Product Entities ---
        for entity in product_entities:
            matches = self.fuzzy_search_system.find_best_matches(entity, "product")
            if not matches:
                continue

            ambiguous_matches = [m[0] for m in matches if m[1] >= AMBIGUITY_THRESHOLD]
            
            if len(ambiguous_matches) > 1:
                clarifications_needed[entity] = ambiguous_matches
                entity_prompt_types[entity] = "ambiguity"
            elif matches and matches[0][0].lower() != entity.lower():
                clarifications_needed[entity] = [m[0] for m in matches[:3]]
                entity_prompt_types[entity] = "misspelling"

        # --- Block 2: Process Customer Entities (The missing part) ---
        for entity in customer_entities:
            # Use the exact same logic as above, but for the "customer" entity type
            matches = self.fuzzy_search_system.find_best_matches(entity, "customer")
            if not matches:
                continue

            ambiguous_matches = [m[0] for m in matches if m[1] >= AMBIGUITY_THRESHOLD]
            
            # Case 1: Ambiguity (e.g., user enters "John Smith" and there are multiple)
            if len(ambiguous_matches) > 1:
                clarifications_needed[entity] = ambiguous_matches
                entity_prompt_types[entity] = "ambiguity"
            # Case 2: Misspelling or incomplete name (e.g., "Jon Smit" or "Alice")
            elif matches and matches[0][0].lower() != entity.lower():
                clarifications_needed[entity] = [m[0] for m in matches[:3]]
                entity_prompt_types[entity] = "misspelling"

        # --- Final Prompt Generation ---
        if clarifications_needed:
            state["needs_clarification"] = True
            prompt_parts = []
            
            # Check if there are any ambiguous entities to use the right header
            has_ambiguity = any(ptype == "ambiguity" for ptype in entity_prompt_types.values())

            if has_ambiguity:
                prompt_parts.append("Your query is a bit ambiguous. Could you please clarify?")
            else:
                prompt_parts.append("Did you mean:")

            for original, suggestions in clarifications_needed.items():
                prompt_type = entity_prompt_types.get(original, "misspelling")
                
                if prompt_type == "ambiguity":
                    prompt_parts.append(f"\nFor **'{original}'**, I found several matches. Please choose one:")
                    for i, suggestion in enumerate(suggestions, 1):
                        prompt_parts.append(f"{i}. {suggestion}")
                else: # Misspelling
                    suggestion_str = ' or '.join([f"**'{s}'**" for s in suggestions])
                    prompt_parts.append(f"- For **'{original}'**, did you mean {suggestion_str}?")
            
            state["clarification_prompt"] = "\n".join(prompt_parts)
            state["corrected_question"] = user_question
            logger.warning(f"Clarification needed: {clarifications_needed}")
        else:
            state["needs_clarification"] = False
            state["clarification_prompt"] = ""
            state["corrected_question"] = user_question
            logger.info("No clarification needed.")

        return state

    def _check_clarification_needed(self, state: AgentState) -> Literal["clarify", "continue"]:
        """Check if fuzzy search requires user clarification."""
        if state["needs_clarification"]:
            return "clarify"
        return "continue"

    def _handle_clarification(self, state: AgentState) -> AgentState:
        """Handle the case where fuzzy search needs user clarification."""
        logger.info("Handling clarification prompt for the user.")
        state["final_response"] = state["clarification_prompt"]
        # Clear other fields as we are stopping here to wait for user input
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["follow_up_questions"] = []
        state["chart_code"] = ""
        return state

    def _validate_query(self, state: AgentState) -> AgentState:
        """Validate if the query is data-related using improved guardrails."""
        user_question = state["user_question"].strip().lower()
        logger.info(f"Validating user query: {user_question[:50]}...")

        # Check for common greetings first
        greeting_patterns = [
            'hi', 'hello', 'hey', 'good morning', 'good afternoon', 
            'good evening', 'howdy', 'greetings', 'hiya', 'sup',
            'good day', 'morning', 'afternoon', 'evening'
        ]
        
        # Simple greeting detection
        is_greeting = (
            user_question in greeting_patterns or
            any(user_question.startswith(greeting) for greeting in greeting_patterns) or
            any(greeting in user_question.split() for greeting in greeting_patterns[:5])  # Check main greetings
        )
        
        if is_greeting:
            state["is_valid_query"] = True
            state["user_question"] = "greeting"  # Special flag for greeting handling
            logger.info("Detected greeting - treating as valid query")
            return state

        # For non-greetings, use LLM validation
        validation_prompt = ChatPromptTemplate.from_template("""
    {guardrails}

    User query: "{user_question}"

    Is this query related to database/data analysis that I should help with? Answer only "YES" or "NO".
    """)

        formatted_prompt = validation_prompt.format(
            guardrails=GUARDRAILS_PROMPT,
            user_question=state["user_question"]
        )

        response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
        is_valid = "YES" in response.content.upper()

        state["is_valid_query"] = is_valid
        logger.info(f"Query validation result: {'Valid' if is_valid else 'Invalid'}")
        return state

    def _check_query_validity(self, state: AgentState) -> Literal["valid", "invalid"]:
        """Check if query passed validation."""
        return "valid" if state["is_valid_query"] else "invalid"

    def _handle_invalid_query(self, state: AgentState) -> AgentState:
        """Handle non-data related queries with witty response."""
        logger.info("Handling invalid query with witty response")
        
        state["final_response"] = self.response_manager.get_error_message('invalid_query')
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["follow_up_questions"] = []
        state["chart_code"] = ""
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        return state

    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze query with intelligent context awareness."""
        user_question = state["user_question"]
        logger.info(f"Analyzing query for context: {user_question[:50]}...")

        # Update conversation context from the agent's state
        state["conversation_context"] = self.conversation_context
        state["previous_queries"] = self.query_history[-3:] if self.query_history else []

        # Intelligent context detection using LLM
        needs_context = False
        
        # ALWAYS run the context detection prompt, even for the first message.
        context_detection_prompt = ChatPromptTemplate.from_template("""
You are analyzing whether a new user query requires context from previous queries in a database chat session.

Previous Context:
- Last entity referenced: {last_entity}
- Last table used: {last_table}
- Last filter conditions: {last_conditions}
- Previous SQL queries: {previous_queries}

Current User Query: "{current_query}"

Does this current query need context from the previous queries to be answered correctly?

Consider these scenarios:
1. If the query uses pronouns like "this", "that", "it", "them" referring to previous entities - NEEDS CONTEXT
2. If the query asks for "more", "next", "continue", "show additional" - NEEDS CONTEXT  
3. If the query mentions "same customer", "same product", "that item" - NEEDS CONTEXT
4. If the query is asking for comparison with "previous results" - NEEDS CONTEXT
5. If the query is completely independent and can be answered without previous context - NO CONTEXT
6. If the query is asking about completely different entities or topics - NO CONTEXT

Answer only "YES" if context is needed, or "NO" if the query is independent.
""")

        try:
            # Use .get() with defaults to handle empty context gracefully
            formatted_prompt = context_detection_prompt.format(
                last_entity=self.conversation_context.get('last_entity', 'None'),
                last_table=self.conversation_context.get('last_table', 'None'),
                last_conditions=self.conversation_context.get('last_conditions', 'None'),
                previous_queries='; '.join(self.query_history[-2:]) if self.query_history else 'None',
                current_query=user_question
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            needs_context = "YES" in response.content.upper()
            logger.info(f"LLM-based context detection result: {'Needs context' if needs_context else 'Independent query'}")

        except Exception as e:
            logger.error(f"Context detection with LLM failed: {e}")
            # Fallback to simple keyword detection, which is crucial for this error case
            context_keywords = ["previous", "that", "this", "it", "them", "more", "next", "continue", "same"]
            needs_context = any(keyword in user_question.lower().split() for keyword in context_keywords)
            logger.warning(f"Falling back to keyword-based context detection. Result: {'Needs context' if needs_context else 'Independent query'}")

        # Ensure conversation_context exists in the state before updating
        if "conversation_context" not in state or not isinstance(state["conversation_context"], dict):
            state["conversation_context"] = {}
            
        state["conversation_context"]["needs_context"] = needs_context
        return state

    def _generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL with intelligent context awareness for Azure Synapse."""
        user_question = state["corrected_question"]
        current_date = datetime.now().strftime('%Y-%m-%d')
        context = state.get("conversation_context", {})
        previous_queries = state.get("previous_queries", [])

        logger.info(f"Generating SQL for: {user_question[:50]}...")

        # Check if context is needed but not available
        if context.get("needs_context", False) and not previous_queries:
            # Handle the case where context is needed but no previous context exists
            state["sql_query"] = "NO_CONTEXT_ERROR"
            state["error_message"] = "No previous query context available. Please provide a specific query first."
            logger.warning("Context needed but no previous queries available. Setting NO_CONTEXT_ERROR.")
            return state
        
        # Build context-aware prompt only if context is actually needed AND available
        context_info = ""
        if context.get("needs_context", False) and previous_queries:
            context_info = f"""
    IMPORTANT - This query requires context from previous conversation:
    - Last entity referenced: {context.get('last_entity', 'None')}
    - Last table used: {context.get('last_table', 'None')}
    - Last filter conditions: {context.get('last_conditions', 'None')}
    - Previous SQL queries: {'; '.join(previous_queries[-2:])}

    When user says "next", "more", or "continue", use OFFSET based on previous results.
    When user refers to "that" or "this", use the entity from the previous context.
    When user asks for comparisons with previous results, maintain the same filters.
    """
        else:
            context_info = """
    This is an independent query that should be answered without reference to previous context.
    Generate a complete SQL query based solely on the current user question.
    If the user is asking for something that requires previous context (like "previous query", "that result", etc.) 
    but no context is available, return an error message instead of generating a generic query.
    """

        sql_prompt = ChatPromptTemplate.from_template(
            """You are an expert Azure Synapse SQL query generator with intelligent context awareness.

{examples}

Current Database Schema:
{schema}

{context_info}

Current date: {current_date}

IMPORTANT: 
- Use SQL Server/Azure Synapse syntax
- Use TOP instead of LIMIT
- Use GETDATE() instead of CURDATE()
- Always use schema prefix: dbo.table_name
- Only use context information if explicitly indicated above
- Ensure data type compatibility in all operations
- If the user asks for a visualization (chart, plot, graph), select columns appropriate for that visualization. For trends, select time-series data; for distributions, select categorical and numerical data.
- Never mix incompatible data types in UNION operations
- Return ONLY the SQL query, no explanations

Generate an Azure Synapse SQL query for: {user_question}"""
        )

        try:
            formatted_prompt = sql_prompt.format(
                examples=ENHANCED_FEW_SHOT_EXAMPLES,
                schema=self.schema_info,
                context_info=context_info,
                current_date=current_date,
                user_question=user_question
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            sql_query = response.content.strip()

            # Clean up the SQL query
            sql_query = re.sub(r'^```sql\n?', '', sql_query)
            sql_query = re.sub(r'\n?```$', '', sql_query)
            sql_query = sql_query.strip()

            # Update context only if this was a contextual query or if it establishes new context
            if context.get("needs_context", False) or self._establishes_new_context(sql_query):
                self._update_context_from_sql(sql_query)

            state["sql_query"] = sql_query

            # Add to query history
            self.query_history.append(sql_query)

            logger.info(f"SQL generated successfully: {sql_query[:100]}...")

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            state["error_message"] = f"Failed to generate SQL query: {str(e)}"
            state["sql_query"] = "Error in SQL generation"

        return state

    def _establishes_new_context(self, sql_query: str) -> bool:
        """Check if the SQL query establishes new context for future queries."""
        # Check if query has specific filters or references that could be useful for context
        context_indicators = [
            r"WHERE.*=.*'[^']+'",  # Specific string filters
            r"WHERE.*LIKE.*'%.*%'",  # Pattern searches
            r"ProductName\s*=",  # Product-specific queries
            r"CustomerName\s*=",  # Customer-specific queries
            r"TOP\s+\d+",  # Pagination queries
        ]

        for pattern in context_indicators:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return True

        return False

    def _update_context_from_sql(self, sql_query: str):
        """Extract and update conversation context from SQL query."""
        logger.debug("Updating conversation context from SQL query")

        # Extract table names
        table_pattern = r'FROM\s+(?:dbo\.)?(\w+)'
        tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
        if tables:
            self.conversation_context['last_table'] = tables[0]

        # Extract entity references
        entity_pattern = r"'([^']+)'"
        entities = re.findall(entity_pattern, sql_query)
        if entities:
            self.conversation_context['last_entity'] = entities[0]

        # Extract conditions
        where_pattern = r'WHERE\s+(.+?)(?:GROUP|ORDER|OFFSET|$)'
        conditions = re.findall(where_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if conditions:
            self.conversation_context['last_conditions'] = conditions[0].strip()

        # Track pagination for Azure Synapse
        if "TOP" in sql_query.upper():
            top_pattern = r'TOP\s+(\d+)'
            tops = re.findall(top_pattern, sql_query, re.IGNORECASE)
            if tops:
                self.conversation_context['last_limit'] = int(tops[0])

        if "OFFSET" in sql_query.upper():
            offset_pattern = r'OFFSET\s+(\d+)'
            offsets = re.findall(offset_pattern, sql_query, re.IGNORECASE)
            if offsets:
                self.conversation_context['last_offset'] = int(offsets[0])
        else:
            self.conversation_context['last_offset'] = 0

        logger.debug("Conversation context updated successfully")

    def _execute_sql(self, state: AgentState) -> AgentState:
        """Execute the SQL query on Azure Synapse."""
        if state["sql_query"] == "NO_CONTEXT_ERROR":
            logger.error("Skipping SQL execution due to NO_CONTEXT_ERROR.")
            return state
        try:
            if state["sql_query"] == "Error in SQL generation":
                state["error_message"] = "Cannot execute query due to SQL generation error"
                logger.error("Cannot execute query due to SQL generation error")
                return state

            logger.info("Executing SQL query on Azure Synapse")
            result_df = self.db_manager.execute_query(state["sql_query"])
            state["query_result"] = result_df

            # Update context with result count
            self.conversation_context['last_result_count'] = len(result_df)

            state["error_message"] = ""
            logger.info(f"SQL execution successful, returned {len(result_df)} rows")

        except Exception as e:
            error_msg = str(e)
            state["error_message"] = error_msg
            logger.error(f"SQL execution error: {error_msg}")

        return state

    def _check_sql_execution(self, state: AgentState) -> Literal["success", "error"]:
        """Check if SQL execution was successful."""
        return "success" if not state.get("error_message") else "error"

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate intelligent response with witty personality."""
        query_result = state["query_result"]
        user_question = state["user_question"]

        logger.info("Generating response")

        # Handle greetings specially
        if user_question == "greeting":
            greeting_responses = [
                "Hi! I'm DataBot, your friendly data assistant. I specialize in analyzing your data and turning your questions into insights. What would you like to explore today?",
                "Hello there! DataBot here - ready to dive into your data and uncover some interesting insights. Ask me anything about your sales, customers, products, or any other data you'd like to analyze!",
                "Good to meet you! I'm DataBot, your AI-powered analytics companion. I can help you query your database, generate charts, and discover valuable business insights. What data mystery shall we solve together?",
                "Hi! I'm DataBot - think of me as your personal data detective. I love turning complex queries into simple answers and transforming raw data into actionable insights. How can I help you today?"
            ]
            
            import random
            state["final_response"] = random.choice(greeting_responses)
            state["sql_query"] = ""
            state["query_result"] = pd.DataFrame()
            state["follow_up_questions"] = [
                "What sales data would you like to explore?",
                "Would you like to see customer analytics?",
                "Can I help you with product performance metrics?"
            ]
            state["chart_code"] = ""
            state["best_chart_type"] = ""
            state["visualization_suggestion"] = ""
            return state

        # Rest of the existing _generate_response method for data queries...
        if query_result.empty:
            # Detect date-based queries for appropriate empty response
            date_keywords = ["today", "yesterday", "last", "this", "month", "year", "week", "day", "date", "ytd", "quarter"]
            is_date_query = any(keyword in user_question.lower() for keyword in date_keywords)
            
            response = self.response_manager.get_empty_message(is_date_query)
            state["final_response"] = response
            logger.info("Generated witty response for empty result set")
        else:
            try:
                # Generate witty success message
                response_parts = []
                success_msg = self.response_manager.get_success_message(len(query_result))
                response_parts.append(success_msg)
                
                # Generate business insights
                insights = self._generate_automated_insights(query_result, user_question)
                if insights:
                    response_parts.append("\n\n**Business Intelligence Summary:**")
                    response_parts.extend([f"\n{insight}" for insight in insights])
                    
                    # Add strategic recommendation if applicable
                    if len(query_result) > 50:
                        response_parts.append(f"\n\n**Recommendation**: With {len(query_result)} data points analyzed, consider drilling down into specific segments for more targeted insights.")

                # Professional data summary
                if len(query_result) > 20:
                    response_parts.append(f"\n\n*Displaying top 20 records from {len(query_result):,} total results. Full dataset available for download.*")
                
                final_response = "".join(response_parts)
                
                # Add personality touch
                final_response = self.response_manager.add_personality_touch(
                    final_response, 
                    {"result_count": len(query_result), "query": user_question}
                )
                
                state["final_response"] = final_response
                logger.info(f"Generated witty response with insights for {len(query_result)} records")

            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                state["final_response"] = self.response_manager.get_success_message(len(query_result))

        return state
    
    def _generate_automated_insights(self, df: pd.DataFrame, user_question: str) -> List[str]:
        """Generate professional business insights from dataframe."""
        insights = []
        
        try:
            # Business-focused numeric column insights
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:3]:
                if df[col].notna().sum() > 0:
                    total = df[col].sum()
                    avg = df[col].mean()
                    max_val = df[col].max()
                    min_val = df[col].min()
                    
                    # Business context for different metric types
                    if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount']):
                        insights.append(f"• **Revenue Analysis**: Total {col.replace('_', ' ').title()}: ${total:,.0f} with an average transaction value of ${avg:,.0f}")
                        if max_val > avg * 2:
                            insights.append(f"• **Performance Variance**: Highest {col.lower()} of ${max_val:,.0f} indicates significant growth opportunities in top-performing segments")
                    
                    elif any(keyword in col.lower() for keyword in ['volume', 'quantity', 'count', 'orders']):
                        insights.append(f"• **Volume Metrics**: Total {col.replace('_', ' ').title()}: {total:,.0f} units across {len(df)} records (Avg: {avg:,.0f} per record)")
                        
                    elif any(keyword in col.lower() for keyword in ['margin', 'profit', 'roi']):
                        insights.append(f"• **Profitability**: {col.replace('_', ' ').title()} ranges from {min_val:.1f}% to {max_val:.1f}% with average of {avg:.1f}%")
                    
                    else:
                        insights.append(f"• **Key Metric**: {col.replace('_', ' ').title()} - Range: {min_val:,.0f} to {max_val:,.0f} (Avg: {avg:,.0f})")

            # Business-focused categorical analysis
            categorical_cols = df.select_dtypes(include=['object', 'string']).columns
            for col in categorical_cols[:2]:
                if df[col].notna().sum() > 0:
                    unique_count = df[col].nunique()
                    
                    if any(keyword in col.lower() for keyword in ['customer', 'client', 'account']):
                        if unique_count <= 10:
                            top_customers = df[col].value_counts().head(3)
                            customer_list = ", ".join([f"{customer} ({count} transactions)" for customer, count in top_customers.items()])
                            insights.append(f"• **Customer Distribution**: Top customers by transaction volume - {customer_list}")
                        else:
                            insights.append(f"• **Customer Portfolio**: {unique_count} unique customers represented in this dataset")
                    
                    elif any(keyword in col.lower() for keyword in ['product', 'item', 'sku']):
                        if unique_count <= 10:
                            top_products = df[col].value_counts().head(3)
                            product_list = ", ".join([f"{product} ({count})" for product, count in top_products.items()])
                            insights.append(f"• **Product Performance**: Leading products - {product_list}")
                        else:
                            insights.append(f"• **Product Catalog**: {unique_count} different products analyzed")
                    
                    elif any(keyword in col.lower() for keyword in ['region', 'territory', 'location', 'country']):
                        if unique_count <= 10:
                            top_regions = df[col].value_counts().head(3)
                            region_list = ", ".join([f"{region} ({count} records)" for region, count in top_regions.items()])
                            insights.append(f"• **Geographic Distribution**: Primary markets - {region_list}")
                        else:
                            insights.append(f"• **Market Coverage**: Operations across {unique_count} geographic locations")
                    
                    else:
                        if unique_count <= 10:
                            value_counts = df[col].value_counts().head(3)
                            top_items = ", ".join([f"{val} ({count})" for val, count in value_counts.items()])
                            insights.append(f"• **{col.replace('_', ' ').title()} Analysis**: {top_items}")

            # Time-based business insights
            date_cols = [col for col in df.columns if any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'updated'])]
            if date_cols and len(df) > 1:
                date_col = date_cols[0]
                try:
                    df_temp = df.copy()
                    df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                    date_range = df_temp[date_col].max() - df_temp[date_col].min()
                    insights.append(f"• **Time Period**: Data spans {date_range.days} days from {df_temp[date_col].min().strftime('%B %Y')} to {df_temp[date_col].max().strftime('%B %Y')}")
                    
                    # Monthly trend if applicable
                    if len(df) > 10 and date_range.days > 30:
                        monthly_data = df_temp.groupby(pd.Grouper(key=date_col, freq='M')).size()
                        if len(monthly_data) > 1:
                            trend = "increasing" if monthly_data.iloc[-1] > monthly_data.iloc[0] else "decreasing"
                            insights.append(f"• **Trend Analysis**: Data shows {trend} activity pattern over the analyzed period")
                except:
                    pass

            # Business performance indicators
            if len(df) > 100:
                insights.append(f"• **Data Volume**: Robust dataset with {len(df):,} records providing statistically significant insights for decision-making")
            elif len(df) > 20:
                insights.append(f"• **Sample Size**: {len(df)} records analyzed - suitable for tactical business insights")

        except Exception as e:
            logger.error(f"Error generating business insights: {e}")
            # Fallback to basic business insight
            insights.append(f"• **Dataset Overview**: {len(df):,} business records analyzed across {len(df.columns)} key metrics")

        return insights[:4]

    def _generate_contextual_info(self, df: pd.DataFrame) -> str:
        """Generate contextual information about the dataset structure."""
        try:
            info_parts = []
            
            # Basic structure info
            info_parts.append(f"{len(df):,} records with {len(df.columns)} columns")
            
            # Column types summary
            numeric_count = len(df.select_dtypes(include=[np.number]).columns)
            text_count = len(df.select_dtypes(include=['object', 'string']).columns)
            date_count = len(df.select_dtypes(include=['datetime64']).columns)
            
            type_info = []
            if numeric_count > 0:
                type_info.append(f"{numeric_count} numeric")
            if text_count > 0:
                type_info.append(f"{text_count} text")
            if date_count > 0:
                type_info.append(f"{date_count} date")
            
            if type_info:
                info_parts.append(f"({', '.join(type_info)} fields)")
            
            return " ".join(info_parts)
            
        except Exception as e:
            logger.error(f"Error generating contextual info: {e}")
            return f"{len(df):,} records available for analysis"

    def _analyze_for_visualization(self, state: AgentState) -> AgentState:
        """NEW: Analyze data to determine best visualization and create suggestion."""
        query_result = state["query_result"]
        user_question = state["user_question"]
        sql_query = state["sql_query"]
        
        logger.info("Analyzing data for visualization recommendations")
        
        # Initialize with defaults
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        
        # Only analyze if we have data and the query wasn't explicitly asking for a chart
        if query_result.empty:
            return state
            
        chart_keywords = ["chart", "plot", "graph", "visualize", "diagram", "bar", "pie", "line"]
        if any(keyword in user_question.lower() for keyword in chart_keywords):
            # User already asked for a chart, skip analysis
            return state

        try:
            # Create a string buffer to capture dataframe info
            buffer = StringIO()
            query_result.info(buf=buffer)
            info_str = buffer.getvalue()
            
            # Get basic stats about the data
            columns = query_result.columns.tolist()
            num_rows = len(query_result)
            
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_template("""
You are a data visualization expert. Analyze the provided query results and determine if a visualization would be valuable and what type would be best.

USER'S ORIGINAL QUESTION: "{user_question}"
SQL QUERY USED: "{sql_query}"

DATA ANALYSIS:
- Columns: {columns}
- Row count: {num_rows}
- DataFrame info: {info_str}

Based on this data, should a visualization be suggested as a follow-up question?

RULES FOR RECOMMENDATION:
1. Only recommend if visualization would add meaningful insights
2. Don't recommend for simple lookups (single row, basic info)
3. Don't recommend if data is not suitable for the allowed chart types
4. Consider the business context and what insights charts could provide

ALLOWED CHART TYPES: PIE CHART, BAR CHART, LINE CHART, HISTOGRAM, BOX PLOT

Analyze the data and respond in this EXACT format:
RECOMMEND: YES/NO
CHART_TYPE: [one of the allowed types if YES, otherwise empty]
SUGGESTION: [A natural follow-up question suggesting visualization if YES, otherwise empty]

Guidelines for chart selection:
- PIE CHART: For categorical distributions (3-8 categories), compositions
- BAR CHART: For comparing categories, rankings, counts
- LINE CHART: For trends over time, continuous data relationships
- HISTOGRAM: For showing distribution of numerical data
- BOX PLOT: For comparing distributions across categories

Example good suggestion:PIE CHART-1. "Would you like to see a pie chart showing the distribution of sales across different regions?
                        LINE CHART-2. "Would you like to visualize the trend of monthly revenue over the past year with a line chart?
                        BAR CHART-3. "Would you like to visualize this with a bar chart showing the total number of orders for each day of the week"?
                        HISTOGRAM-4. "Would you like to see a histogram to understand the distribution of all order values? This can show if most orders are small, large, or clustered around the average"
                        BOX PLOT-5. "To analyze profitability by region, would a box plot comparing the distribution of profit margins for orders from each region be useful?"                                                                                                                     "                      "
""")

            formatted_prompt = analysis_prompt.format(
                user_question=user_question,
                sql_query=sql_query,
                columns=columns,
                num_rows=num_rows,
                info_str=info_str[:500]  # Limit info string length
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse the response
            lines = response.content.strip().split('\n')
            recommend = "NO"
            chart_type = ""
            suggestion = ""
            
            for line in lines:
                if line.startswith("RECOMMEND:"):
                    recommend = line.split(":", 1)[1].strip()
                elif line.startswith("CHART_TYPE:"):
                    chart_type = line.split(":", 1)[1].strip()
                elif line.startswith("SUGGESTION:"):
                    suggestion = line.split(":", 1)[1].strip()
            
            if recommend.upper() == "YES" and chart_type and suggestion:
                state["best_chart_type"] = chart_type
                state["visualization_suggestion"] = suggestion
                logger.info(f"Visualization recommended: {chart_type} - {suggestion}")
            else:
                logger.info("No visualization recommended for this data")

        except Exception as e:
            logger.error(f"Visualization analysis failed: {e}")
            # Continue without visualization recommendation
            
        return state

    def _decide_to_generate_chart(self, state: AgentState) -> AgentState:
        """Decide if a chart should be generated based on the user query."""
        user_question = state["user_question"]
        query_result = state["query_result"]
        
        # Expanded chart keywords detection
        chart_keywords = [
            "chart", "plot", "graph", "visualize", "diagram", "bar", "pie", "line", 
            "histogram", "show", "display", "trend", "comparison", "compare",
            "visual", "see a", "create a", "generate a", "make a"
        ]
        
        # Check if user explicitly asked for visualization
        user_question_lower = user_question.lower()
        explicit_chart_request = any(keyword in user_question_lower for keyword in chart_keywords)
        
        # Also check for visualization patterns like "would you like to see a bar chart"
        visualization_patterns = [
            "bar chart", "pie chart", "line chart", "histogram", "box plot",
            "visualize the", "see a chart", "show a graph", "create a plot"
        ]
        
        pattern_match = any(pattern in user_question_lower for pattern in visualization_patterns)
        
        # Only generate a chart if the user asked for one and there's data
        if (explicit_chart_request or pattern_match) and not query_result.empty:
            state["conversation_context"]["should_chart"] = "generate"
            logger.info(f"Decision: Generate chart for request: {user_question}")
            print(f"DEBUG: Chart generation triggered for: {user_question}")
        else:
            state["conversation_context"]["should_chart"] = "skip"
            logger.info("Decision: Skip chart generation.")
            print(f"DEBUG: Chart generation skipped for: {user_question}")
        
        return state

    def _should_generate_chart(self, state: AgentState) -> Literal["generate", "skip"]:
        """Return the decision on whether to generate a chart."""
        return state["conversation_context"].get("should_chart", "skip")

    def _extract_code_from_response(self, llm_response: str) -> str:
        """Filters and extracts a Python code block from a raw LLM response string."""
        pattern = r"```(?:python)?\s*(.*?)```"
        match = re.search(pattern, llm_response, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _generate_chart(self, state: AgentState) -> AgentState:
        """Generate Python code for a chart using Matplotlib with intelligent chart type selection."""
        user_question = state["user_question"]
        df = state["query_result"]
        columns = df.columns.tolist()
        
        # Use the recommended chart type if available, otherwise let LLM decide
        recommended_chart_type = state.get("best_chart_type", "")
        
        logger.info(f"Generating chart code for: {user_question[:50]}... (Recommended type: {recommended_chart_type})")

        # Create a string buffer to capture dataframe info
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()

        # Context-aware prompt for chart generation
        chart_prompt = ChatPromptTemplate.from_template(
            """
You are an expert Python data visualization assistant. Your task is to generate Python code to create a single, insightful chart that answers a user's query based on the provided data.

USER'S ORIGINAL QUESTION:
"{user_question}"

This question was translated into the following SQL query to get the data:
"{sql_query}"

RECOMMENDED CHART TYPE: {recommended_chart_type}
(Use this type if specified, otherwise choose the best one from allowed types)

DATA CONTEXT:
- A Pandas DataFrame is available in the variable `df`. DO NOT write code to load data.
- The DataFrame has the following columns: {columns}
- The DataFrame info is:
{info_str}
- For your reference, here is a description of some dataset columns:
{mapping_prompt}

STRICT GUIDELINES:
1.  ALLOWED CHARTS: You can ONLY generate: PIE CHART, BOX PLOT, HISTOGRAM, BAR CHART, or LINE CHART. Choose the best one to answer the user's question.
2.  ALLOWED LIBRARIES: Use ONLY `pandas` and `matplotlib.pyplot`.
3.  CODE OUTPUT:
    -   Your response must ONLY be a single Python code block.
    -   DO NOT add any explanations or comments.
    -   The code block MUST start with ```python and end with ```.
    -   CRITICAL: Assign the Matplotlib figure object to a variable named `fig` (e.g., `fig, ax = plt.subplots()`).
    -   DO NOT call `plt.show()`.
4.  STYLING:
    -   Create visually appealing and professional-looking charts.
    -   Use a background color for the figure of `#F0F0F6`.
    -   Ensure labels do not overlap and text is legible.
    -   The chart title should be context-aware, reflecting the original user question.

5. VERY STRICT RULE: DO NOT GENERATE ANY CHART OR TABLES EXCEPT THE ONES MENTIONED ABOVE. EVEN IF THE USER ASKS FOR IT DIRECTLY OR INDIRECTLY.

6. BEFORE GENERATING THE CODE ANALYZE THE USER QUERY, BY KEEPING THE MAPPINGS PROVIDED ABOVE IN MIND, THEN CHOOSE THE BEST SUITED CHART FROM THE ONES MENTIONED ABOVE. YOUR FLOW SHOULD BE STRICTLY LIKE THIS:
- ANALYZE THE USER QUERY    
-> THINK ABOUT THE DATA AND THE MAPPINGS PROVIDED ABOVE
-> CHOOSE THE BEST SUITED CHART FROM THE ONES MENTIONED ABOVE 

7. CODE EXAMPLES & PARAMETER GUIDELINES
- You MUST use the following code snippets as a reference for generating your code.
- You are ONLY allowed to use the parameters for each chart type as shown in the examples below. DO NOT introduce any new parameters, as they might be deprecated or cause errors.

8. IMPORTANT RULE WHILE WRITING CODE:
- YOU NEED TO OBSERVE THE DATAFRAME COLUMNS AVAILABLE IN THE DATAFRAME AND THEN CHOOSE THE COLUMNS ACCORDINGLY IN THE CODE YOU ARE WRITING.
For Example : The only columns available in the dataframe were [Month, Year, TotalSales]
The User query was : Is there a seasonal trend in order volumes for 'Fanta Strawberry 330ml 4x6 NP Sleek Can'?
- So in this case you need to choose the columns Month, Year and TotalSales in the code you are writing.
- Even though User has mentioned the product name in the query you CANNOT use that column in the code as it is not present in the dataframe.
- You CANNOT use any other column which is not present in the dataframe.
- YOU CAN ONLY USE {columns} while writing the code.
- You can add the product name in the title of the graph though.

- YOU NEED TO STRICTLY FOLLOW THIS RULE WHILE WRITING THE CODE.

9. TIME SERIES DATA HANDLING:
- If the data contains dates/timestamps (OrderDate, Date, etc.), treat it as TIME SERIES data
- For time series data, ALWAYS use LINE CHARTS, not bar charts
- Convert date columns to datetime using: df['DateColumn'] = pd.to_datetime(df['DateColumn'])
- For time series line charts, use: plt.plot(df['DateColumn'], df['ValueColumn'])
- Do NOT use plt.bar() for time series data
- Let matplotlib handle date formatting automatically on x-axis


    ---Pie Chart Code Example---
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    dept_counts = df['department'].value_counts()
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    wedges, texts, autotexts = plt.pie(dept_counts.values, 
                                    labels=dept_counts.index, 
                                    autopct='%1.1f%%', 
                                    colors=colors, 
                                    startangle=90,
                                    explode=(0.05, 0, 0, 0, 0))  # Explode the largest slice
    plt.title('Department Distribution', fontsize=16, fontweight='bold', pad=20)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    # Enhance text formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    plt.tight_layout()
    plt.show()
   ```
    
    ---Box Plot Code Example---
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    education_order = ['High School', 'Bachelor', 'Master', 'PhD']
    box_data = [df[df['education'] == edu]['income'] for edu in education_order]
    box_plot = plt.boxplot(box_data, labels=education_order, patch_artist=True)

    # Color the boxes
    colors_box = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    plt.title('Income Distribution by Education Level', fontsize=16, fontweight='bold')
    plt.ylabel('Income ($)', fontsize=12)
    plt.xlabel('Education Level', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(True, alpha=0.3)
    # Format y-axis to show values in thousands
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${{x/1000:.0f}}K'))
    plt.tight_layout()
    plt.show()
    ```    
    
    ---Bar Chart Code Example---
    ```
    import pandas as pd
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    avg_satisfaction = df.groupby('department')['satisfaction_score'].mean().sort_values(ascending=False)
    bars = plt.bar(avg_satisfaction.index, avg_satisfaction.values, 
                color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
                alpha=0.8, edgecolor='black', linewidth=1)

    plt.title('Average Satisfaction Score by Department', fontsize=16, fontweight='bold')
    plt.ylabel('Average Satisfaction Score', fontsize=12)
    plt.xlabel('Department', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{{height:.2f}}', ha='center', va='bottom', fontweight='bold')

    # Set y-axis limits for better visualization
    plt.ylim(0, max(avg_satisfaction.values) * 1.1)
    plt.tight_layout()
    plt.show()
    ```

    ---Line Chart Code Example---
    ```
---Line Chart Code Example (including Time Series)---
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#F0F0F6')

    # For time series data, convert dates first
    if 'OrderDate' in df.columns or any('date' in col.lower() for col in df.columns):
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        df[date_col] = pd.to_datetime(df[date_col])
        ax.plot(df[date_col], df.iloc[:, 1], marker='o', linewidth=3, markersize=8)
        plt.xticks(rotation=45)
    else:
        # For non-time series data
        yearly_income = df.groupby('year')['income'].mean().sort_index()
        ax.plot(yearly_income.index, yearly_income.values, marker='o', linewidth=3, markersize=8)

    ax.set_title('Title Here', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ```

    --Histogram Code Example---
    ```
    import pandas as pd
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    yearly_income = df.groupby('year')['income'].mean().sort_index()

    # Create histogram with years as x-axis and income values as heights
    bars = plt.bar(yearly_income.index, yearly_income.values, 
                color='skyblue', edgecolor='black', alpha=0.7, width=0.8)

    # Color gradient for bars
    for i, bar in enumerate(bars):
        bar.set_facecolor(plt.cm.viridis(i / len(bars)))

    plt.title('Average Income by Year', fontsize=16, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Income ($)', fontsize=12)
    plt.grid(True, alpha=0.3)

    # Format y-axis to show values in thousands
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${{x/1000:.0f}}K'))

    # Add statistics text box
    mean_income = yearly_income.mean()
    std_income = yearly_income.std()
    textstr = f'Mean: ${{mean_income/1000:.1f}}K\nStd: ${{std_income/1000:.1f}}K\nCount: {{len(yearly_income)}}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.75, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.show()
    ```


Your goal is to generate the Python code to create the best visualization for the user's question: "{user_question}"
"""
        )

        try:
            formatted_prompt = chart_prompt.format(
                user_question=user_question,
                sql_query=state.get("sql_query", "Not available"),
                recommended_chart_type=recommended_chart_type,
                columns=df.columns.tolist(),
                info_str=info_str,
                mapping_prompt=MAPPING_SCHEMA_PROMPT
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            chart_code = self._extract_code_from_response(response.content)

            state["chart_code"] = chart_code
            logger.info(f"Chart code generated successfully: {chart_code[:100]}...")

        except Exception as e:
            logger.error(f"Chart code generation failed: {e}")
            state["chart_code"] = "# Error generating chart code."

        return state

    def _generate_followup_questions(self, state: AgentState) -> AgentState:
        """Generate ONE visualization question and TWO data insight questions."""
        user_question = state["user_question"]
        sql_query = state["sql_query"]
        query_result = state["query_result"]
        visualization_suggestion = state.get("visualization_suggestion", "")

        logger.info("Generating follow-up questions")

        try:
            if query_result.empty or state.get("error_message"):
                state["follow_up_questions"] = []
                return state

            # Build context for follow-up generation
            columns_info = ", ".join(query_result.columns.tolist())
            sample_data = query_result.head(3).to_dict('records') if len(query_result) > 0 else []

            # Check if user already asked for visualization
            chart_keywords = ["chart", "plot", "graph", "visualize", "diagram", "bar", "pie", "line"]
            user_asked_for_chart = any(keyword in user_question.lower() for keyword in chart_keywords)

            if user_asked_for_chart:
                # If user already asked for visualization, generate 3 data insight questions
                followup_prompt = ChatPromptTemplate.from_template("""
    Based on the user's question and the query results, generate 3 intelligent follow-up questions that help explore the data deeper.

    User Question: {user_question}
    SQL Query: {sql_query}
    Result Columns: {columns}
    Sample Data: {sample_data}
    Record Count: {record_count}

    Since the user already requested a visualization, generate 3 data analysis questions that:
    1. Explore different time periods or filters
    2. Suggest comparisons with other categories
    3. Recommend deeper analysis or related insights

    Return exactly 3 follow-up questions, one per line, without numbering or bullets.
    """)
            else:
                # Generate 1 visualization + 2 data questions
                followup_prompt = ChatPromptTemplate.from_template("""
    Based on the user's question and the query results, generate exactly 3 follow-up questions.

    User Question: {user_question}
    SQL Query: {sql_query}
    Result Columns: {columns}
    Sample Data: {sample_data}
    Record Count: {record_count}
    Visualization Suggestion: {visualization_suggestion}

    Generate follow-up questions in this EXACT order:
    1. FIRST: A visualization question (use the provided visualization suggestion if available, or create one asking for a chart/graph to visualize the data)
    2. SECOND: A question exploring different dimensions or filters of the same data
    3. THIRD: A question suggesting deeper analysis or comparisons

    Return exactly 3 follow-up questions, one per line, without numbering or bullets.
    Make sure the FIRST question is always about creating a visualization.
    """)

            formatted_prompt = followup_prompt.format(
                user_question=user_question,
                sql_query=sql_query,
                columns=columns_info,
                sample_data=str(sample_data)[:500],  # Limit sample data size
                record_count=len(query_result),
                visualization_suggestion=visualization_suggestion if not user_asked_for_chart else ""
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])

            # Parse follow-up questions
            questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            questions = [q for q in questions if q and not q.startswith(('1.', '2.', '3.', '-', '•'))]

            # Clean up questions and ensure exactly 3
            clean_questions = []
            for q in questions[:3]:
                q = q.strip()
                if q and not q.endswith('?'):
                    q += '?'
                clean_questions.append(q)

            # If we don't have a visualization question as first and user didn't ask for chart, add one
            if not user_asked_for_chart:
                viz_keywords = ["chart", "plot", "graph", "visualiz", "diagram"]
                has_viz_question = any(any(keyword in q.lower() for keyword in viz_keywords) for q in clean_questions)
                
                if not has_viz_question:
                    # Create a default visualization question
                    default_viz = visualization_suggestion if visualization_suggestion else "Would you like to see a chart to visualize this data?"
                    clean_questions.insert(0, default_viz)
                    clean_questions = clean_questions[:3]  # Keep only 3

            # Ensure we have exactly 3 questions
            while len(clean_questions) < 3:
                clean_questions.append("What other insights would you like to explore from this data?")

            state["follow_up_questions"] = clean_questions[:3]
            logger.info(f"Generated {len(clean_questions)} follow-up questions")

        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            # Fallback: ensure we have at least a visualization question if user didn't ask for chart
            chart_keywords = ["chart", "plot", "graph", "visualize", "diagram", "bar", "pie", "line"]
            user_asked_for_chart = any(keyword in user_question.lower() for keyword in chart_keywords)
            
            if not user_asked_for_chart and visualization_suggestion:
                state["follow_up_questions"] = [
                    visualization_suggestion,
                    "What trends can you identify in this data?",
                    "How does this compare to other time periods?"
                ]
            else:
                state["follow_up_questions"] = [
                    "What other insights would you like to explore?",
                    "Would you like to filter this data differently?",
                    "How does this compare to other categories?"
                ]

        return state

    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle SQL execution errors with witty responses."""
        error_message = state["error_message"].lower()
        logger.info(f"Handling SQL execution error with witty response")

        # Handle the specific "no context" error first
        if error_message == "No previous query context available. Please provide a specific query first.":
            state["final_response"] = error_message
            state["follow_up_questions"] = [
                "What sales data would you like to explore?",
                "Can I help you with product performance metrics?",
                "Would you like to see customer analytics?"
            ]
            return state


        # Determine error type for appropriate witty response
        if "table" in error_message and ("doesn't exist" in error_message or "invalid" in error_message):
            error_type = "table_error"
        elif "column" in error_message and ("unknown" in error_message or "invalid" in error_message):
            error_type = "column_error"
        elif "syntax error" in error_message:
            error_type = "syntax_error"
        elif "type clash" in error_message or "incompatible" in error_message:
            error_type = "data_type_error"
        else:
            error_type = "general_error"
        
        state["final_response"] = self.response_manager.get_error_message(error_type)
        state["follow_up_questions"] = []
        return state

    def process_query(self, user_question: str, session_id: str = None, thread_id: str = None) -> Dict[str, Any]:
        """Process user query through the workflow with thread isolation."""
        logger.info(f"Processing user query: {user_question[:100]}...")

        initial_state = AgentState(
            messages=[HumanMessage(content=user_question)],
            user_question=user_question,
            corrected_question="",
            sql_query="",
            query_result=pd.DataFrame(),
            final_response="",
            chart_code="",
            follow_up_questions=[],
            session_id=session_id or str(uuid.uuid4()),
            error_message="",
            conversation_context=self.conversation_context,
            previous_queries=self.query_history[-3:] if self.query_history else [],
            is_valid_query=True,
            needs_clarification=False,
            clarification_prompt="",
            best_chart_type="",
            visualization_suggestion=""
        )

        # Add thread_id to the state if available
        if thread_id:
            initial_state["thread_id"] = thread_id

        try:
            final_state = self.graph.invoke(initial_state)

            result = {
                'success': True,
                'user_input': user_question,
                'sql_query': final_state["sql_query"],
                'dataframe': final_state["query_result"],
                'chart_code': final_state["chart_code"],
                'follow_up_questions': final_state["follow_up_questions"],
                'response': final_state["final_response"],
                'results_count': len(final_state["query_result"]) if not final_state["query_result"].empty else 0,
                'session_id': final_state["session_id"],
                'thread_id': final_state.get("thread_id", thread_id),
                'best_chart_type': final_state.get("best_chart_type", ""),
                'visualization_suggestion': final_state.get("visualization_suggestion", "")
            }

            logger.info(f"Query processed successfully. Results: {result['results_count']} rows")
            return result

        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {
                'success': False,
                'user_input': user_question,
                'sql_query': "Error occurred",
                'dataframe': pd.DataFrame(),
                'chart_code': "",
                'follow_up_questions': [],
                'response': "I apologize, but I encountered an unexpected error. Please try again or rephrase your question.",
                'results_count': 0,
                'session_id': session_id or str(uuid.uuid4()),
                'thread_id': thread_id,
                'best_chart_type': "",
                'visualization_suggestion': ""
            }


class SessionManager:
    """Enhanced session management for multi-user chat history with thread safety."""

    def __init__(self, db_path: str = "datachat_sessions.db"):
        self.db_path = db_path
        self._init_db()
        logger.info(f"SessionManager initialized with database: {db_path}")

    def _init_db(self):
        """Initialize SQLite database for session management."""
        logger.info("Initializing session database")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Updated sessions table with user_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_name TEXT DEFAULT 'New Conversation',
                thread_id TEXT
            )
        """)

        # Create index for sessions table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)
        """)

        # Updated messages table with user_id
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_id TEXT NOT NULL,
                message_type TEXT,
                content TEXT,
                sql_query TEXT,
                dataframe_json TEXT,
                chart_code TEXT,
                follow_up_questions TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)

        # Create indexes for messages table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_user_session ON messages(user_id, session_id)
        """)

        # Table for caching complete session states
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_cache (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                chat_history_json TEXT,
                agent_context_json TEXT,
                query_history_json TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            )
        """)

        # Create index for session_cache table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_cache_user_id ON session_cache(user_id)
        """)

        conn.commit()
        conn.close()
        logger.debug("Session database initialized with multi-user support")
    
    def create_session(self, user_id: str, thread_id: str = None) -> str:
        """Create a new chat session for a specific user."""
        session_id = str(uuid.uuid4())
        if not thread_id:
            thread_id = str(uuid.uuid4())
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO sessions (session_id, user_id, session_name, thread_id) VALUES (?, ?, ?, ?)",
            (session_id, user_id, f"Chat {datetime.now().strftime('%m/%d %H:%M')}", thread_id)
        )

        conn.commit()
        conn.close()
        logger.info(f"New session created for user {user_id}: {session_id}")
        return session_id
    
    def update_session_name(self, session_id: str, user_id: str, first_user_question: str):
        """Update session name based on first user question."""
        clean_question = re.sub(r'[^\w\s-]', '', first_user_question)
        session_name = clean_question[:50] + "..." if len(clean_question) > 50 else clean_question

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "UPDATE sessions SET session_name = ? WHERE session_id = ? AND user_id = ?",
            (session_name, session_id, user_id)
        )

        conn.commit()
        conn.close()
        logger.info(f"Session {session_id} renamed to: {session_name}")

    def delete_session(self, session_id: str, user_id: str):
        """Delete a chat session and its messages for a specific user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM messages WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        cursor.execute("DELETE FROM sessions WHERE session_id = ? AND user_id = ?", (session_id, user_id))
        cursor.execute("DELETE FROM session_cache WHERE session_id = ? AND user_id = ?", (session_id, user_id))

        conn.commit()
        conn.close()
        logger.info(f"Session deleted for user {user_id}: {session_id}")

    def save_message(self, session_id: str, user_id: str, message_type: str, content: str, 
                sql_query: str = None, dataframe: pd.DataFrame = None, chart_code: str = None, 
                follow_up_questions: List[str] = None):
        """Save a message to the session with complete data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Serialize DataFrame more robustly
        dataframe_json = None
        if dataframe is not None and not dataframe.empty:
            try:
                # Use orient='records' for more reliable serialization
                dataframe_json = dataframe.to_json(orient='records', date_format='iso', force_ascii=False)
                
                # Validate that the JSON is valid by trying to parse it
                
                test_parse = json.loads(dataframe_json)  # This will raise an exception if invalid
                
                logger.debug(f"DataFrame serialized successfully: {len(dataframe)} rows, JSON length: {len(dataframe_json)}")
            except Exception as e:
                logger.error(f"Failed to serialize DataFrame with orient='records': {e}")
                # Try alternative serialization method
                try:
                    dataframe_json = dataframe.to_json(orient='split', date_format='iso', force_ascii=False)
                    json.loads(dataframe_json)  # Validate
                    logger.debug("DataFrame serialized using 'split' orientation as fallback")
                except Exception as e2:
                    logger.error(f"All DataFrame serialization methods failed: {e2}")
                    # Store as CSV as last resort (for debugging)
                    try:
                        dataframe_json = dataframe.to_csv(index=False)
                        logger.warning("Stored DataFrame as CSV instead of JSON due to serialization issues")
                    except Exception as e3:
                        logger.error(f"Even CSV serialization failed: {e3}")
                        dataframe_json = None

        follow_up_json = json.dumps(follow_up_questions) if follow_up_questions else None

        cursor.execute("""
            INSERT INTO messages (session_id, user_id, message_type, content, sql_query, 
                                dataframe_json, chart_code, follow_up_questions) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (session_id, user_id, message_type, content, sql_query, 
            dataframe_json, chart_code, follow_up_json))

        cursor.execute(
            "UPDATE sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ? AND user_id = ?",
            (session_id, user_id)
        )

        conn.commit()
        conn.close()
        logger.debug(f"Message saved for user {user_id}, session {session_id}: {message_type}")

    def get_session_history(self, session_id: str, user_id: str) -> List[Dict]:
        """Get chat history for a session belonging to a specific user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT message_type, content, sql_query, dataframe_json, chart_code, 
                follow_up_questions, timestamp 
            FROM messages 
            WHERE session_id = ? AND user_id = ?
            ORDER BY timestamp ASC
        """, (session_id, user_id))

        history = []
        for row in cursor.fetchall():
            # Safely deserialize DataFrame
            dataframe = pd.DataFrame()  # Default empty DataFrame
            if row[3] and row[3].strip():  # Check if dataframe_json exists and is not empty
                try:
                    
                    # First try to parse as JSON
                    json_data = json.loads(row[3])
                    if json_data:  # Make sure it's not empty
                        dataframe = pd.DataFrame(json_data)
                        logger.debug(f"Successfully deserialized DataFrame with {len(dataframe)} rows")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    logger.warning(f"Failed to deserialize DataFrame for session {session_id}: {e}")
                    # Try alternative approach with StringIO
                    try:
                        from io import StringIO
                        dataframe = pd.read_json(StringIO(row[3]))
                        logger.debug("Successfully deserialized using StringIO fallback")
                    except Exception as e2:
                        logger.error(f"All DataFrame deserialization methods failed: {e2}")
                        dataframe = pd.DataFrame()

            # Safely deserialize follow-up questions
            follow_up_questions = []
            if row[5] and row[5].strip():  # Check if follow_up_questions exists and is not empty
                try:
                    follow_up_questions = json.loads(row[5])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to deserialize follow_up_questions for session {session_id}: {e}")
                    follow_up_questions = []
            
            history.append({
                'type': row[0],
                'content': row[1],
                'sql_query': row[2] if row[2] else '',
                'dataframe': dataframe,
                'chart_code': row[4] if row[4] else '',
                'follow_up_questions': follow_up_questions,
                'timestamp': row[6]
            })

        conn.close()
        logger.debug(f"Retrieved {len(history)} messages for user {user_id}, session {session_id}")
        return history

    def get_all_sessions(self, user_id: str) -> List[Dict]:
        """Get all chat sessions for a specific user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT session_id, session_name, created_at, last_activity, thread_id
            FROM sessions 
            WHERE user_id = ?
            ORDER BY last_activity DESC
        """, (user_id,))

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'session_id': row[0],
                'name': row[1],
                'created_at': row[2],
                'last_activity': row[3],
                'thread_id': row[4]
            })

        conn.close()
        logger.debug(f"Retrieved {len(sessions)} sessions for user {user_id}")
        return sessions

    def cache_session_state(self, session_id: str, user_id: str, chat_history: List[Dict], 
                           agent_context: Dict, query_history: List[str]):
        """Cache complete session state for fast loading."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        chat_history_json = json.dumps(chat_history, default=str)
        agent_context_json = json.dumps(agent_context, default=str)
        query_history_json = json.dumps(query_history)

        cursor.execute("""
            INSERT OR REPLACE INTO session_cache 
            (session_id, user_id, chat_history_json, agent_context_json, query_history_json, last_updated)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (session_id, user_id, chat_history_json, agent_context_json, query_history_json))

        conn.commit()
        conn.close()
        logger.debug(f"Session state cached for user {user_id}, session {session_id}")

    def load_cached_session(self, session_id: str, user_id: str) -> Dict:
        """Load cached session state for fast restoration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT chat_history_json, agent_context_json, query_history_json
            FROM session_cache 
            WHERE session_id = ? AND user_id = ?
        """, (session_id, user_id))

        result = cursor.fetchone()
        conn.close()

        if result:
            try:
                return {
                    'chat_history': json.loads(result[0]),
                    'agent_context': json.loads(result[1]),
                    'query_history': json.loads(result[2])
                }
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to load cached session: {e}")
                return None
        
        return None

# Professional CSS styling

def load_professional_css():

    # Inject custom CSS for expanders
    st.markdown(
        """
        <style>
        /* Target the summary element (expander header) */
        summary[class*="st-emotion-cache"] {
            background-color: #f9f9f9 !important;
            color: black !important;
            padding: 0.5rem 1rem !important;
            border-radius: 8px !important;
        }
        
        summary[class*="st-emotion-cache"]:hover {
            background-color: #333333 !important;
            color: white !important;
        }
        
        /* Keep the details content area normal */
        details[class*="st-emotion-cache"] {
            background-color: transparent !important;
        }
        
        /* Ensure the content inside details remains white */
        details[class*="st-emotion-cache"] > div {
            background-color: white !important;
            color: black !important;
        }
        
        /* Alternative more specific targeting */
        .st-emotion-cache-1ah0apa summary,
        .st-emotion-cache-1v6pjqr summary {
            background-color: black !important;
            color: white !important;
        }
        
        .st-emotion-cache-1ah0apa summary:hover,
        .st-emotion-cache-1v6pjqr summary:hover {
            background-color: #333333 !important;
            color: white !important;
        }

        /* Alternative approach - set specific width */
        summary[class*="st-emotion-cache"] {
            background-color: #f9f9f9 !important;
            color: black !important;
            border-radius: 4px !important;
            padding: 0.5rem 1rem !important;
            width: 80% !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            display: block !important;
            border-radius: 12px !important;
        }


        /* Remove borders from expander content areas */
        details[class*="st-emotion-cache"] > div {
            border: none !important;
            box-shadow: none !important;
        }

        /* Remove borders from dataframe */
        div[data-testid="stDataFrame"] {
            border: none !important;
            box-shadow: none !important;
        }

        div[data-testid="stDataFrame"] table {
            border: none !important;
        }

        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
            border: none !important;
        }

        /* Remove borders from pyplot/chart containers */
        div[data-testid="stPyplot"] {
            border: none !important;
            box-shadow: none !important;
        }

        /* Remove borders from any element container inside expanders */
        details[class*="st-emotion-cache"] .element-container {
            border: none !important;
            box-shadow: none !important;
        }

        /* Remove borders from stMarkdown inside expanders */
        details[class*="st-emotion-cache"] .stMarkdown {
            border: none !important;
            box-shadow: none !important;
        }

        /* Target the summary element (expander header) - consolidated */
        summary[class*="st-emotion-cache"] {
            background-color: #f9f9f9 !important;
            color: black !important;
            padding: 0.5rem 1rem !important;
            border-radius: 12px !important;
            width: 80% !important;
            max-width: 800px !important;
            margin: 0 auto !important;
            display: block !important;
            border: none !important;
        }

        summary[class*="st-emotion-cache"]:hover {
            background-color: #333333 !important;
            color: white !important;
        }

        /* Keep the details content area normal and remove borders */
        details[class*="st-emotion-cache"] {
            background-color: transparent !important;
            border: none !important;
        }

        /* Ensure the content inside details remains white and no borders */
        details[class*="st-emotion-cache"] > div {
            background-color: white !important;
            color: black !important;
            border: none !important;
            box-shadow: none !important;
        }

        /* Remove borders from all content inside expanders */
        details[class*="st-emotion-cache"] .element-container,
        details[class*="st-emotion-cache"] div[data-testid="stDataFrame"],
        details[class*="st-emotion-cache"] div[data-testid="stPyplot"],
        details[class*="st-emotion-cache"] .stMarkdown {
            border: none !important;
            box-shadow: none !important;
        }
        </style>
        
        """,
        unsafe_allow_html=True,
    )


    # Inject custom CSS for expanders
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    .stApp {
        background-color: #ffffff;
        max-width: 100%;
        padding: 0;
        margin: 0;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* Center content in streamlit app */
    .stApp > .withScreencast {
        max-width: 800px;
    }

    /* Ensure elements stay centered */
    .element-container, .stMarkdown {
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
        
        
    }
    
    /* Header without logo - ChatGPT style */
    .app-header {
        background:  linear-gradient(135deg, #8B0000 0%, #F40009 100%);
        color: #2d2d2d;
        padding: 0.8rem 1rem;
        border-bottom: 1px solid #e5e5e5;
        margin: 0;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: center;
        justify-content: space-between;
        position: sticky;
        top: 0;
        border-radius: 40px 0 40px 0;
        z-index: 1000;
    }
    
    .header-content {
        display: flex;
        align-items: center;
        gap: 2rem;
    }
    
    .app-title {
        display: flex;
        flex-direction: column;
        color: white;
    }
    
    .app-title h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
        color: white;
    }
    
    .app-title p {
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .bot-name {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 500;
        font-size: 0.9rem;
        color: white;
    }
    
    /* Sidebar logo */
    .sidebar-logo {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar-logo img {
        max-height: 50px;
        max-width: 150px;
        object-fit: contain;
    }
    
    .sidebar-logo-text {
        color: #1e3a8a;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    /* Welcome message */
    .welcome-message {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 800px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .welcome-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 1rem;
    }
    
    .welcome-text {
        color: #4b5563;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Follow-up questions - ChatGPT style */
    .followup-container {
        background: transparent;
        padding: 0.5rem 1.5rem;
        margin: 0.5rem 0;
    }
    
    .followup-title {
        font-weight: 600;
        color: #2d2d2d;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    .followup-question {
        background: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 4px;
        padding: 0.6rem 1rem;
        margin: 0.4rem 0;
        cursor: pointer;
        transition: all 0.2s ease;
        font-size: 0.9rem;
        color: #2d2d2d;
    }
    
    .followup-question:hover {
        background: #f0f0f1;
        border-color: #d1d1d1;
    }
    
    /* Input area styling */
    .stTextInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 1rem;
        background: white;
        border-top: 1px solid #e5e5e5;
        z-index: 1000;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        background: white;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: white !important;
        color: black !important;
        font-size: 0.9rem !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 4px !important;
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Fix expander hover state */
    .streamlit-expanderHeader:hover {
        background-color: #f8f9fa !important;
        color: black !important;
        border-color: #d1d5db !important;
    }

    /* Style download button hover */
    .stDownloadButton:hover button {
        background-color: #f8f9fa !important;
        color: black !important;
        border-color: #d1d5db !important;
    }

    /* Style download button normal state */
    .stDownloadButton button {
        background-color: white !important;
        color: black !important;
        border: 1px solid #e5e7eb !important;
        transition: all 0.2s ease !important;
    }

    /* Ensure expander content is visible */
    .streamlit-expanderContent {
        background-color: white !important;
        color: black !important;
        border: 1px solid #e5e7eb !important;
        border-top: none !important;
        padding: 1rem !important;
        margin-top: -0.5rem !important;
    }
    
    /* Style for expanded state */
    .streamlit-expanderHeader[data-expanded="true"] {
        border-bottom: none !important;
        border-bottom-left-radius: 0 !important;
        border-bottom-right-radius: 0 !important;
    }
    
    /* DataFrame styling */
    .dataframe {
        border: 1px solid #e5e5e5 !important;
        border-radius: 4px !important;
    }
    
    .dataframe th {
        background: #f7f7f8 !important;
        color: #2d2d2d !important;
        font-weight: 600 !important;
    }
    
    .followup-question:hover {
        background: #f1f5f9;
        border-color: #3b82f6;
        transform: translateY(-1px);
    }
    
    /* Message containers */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 1rem;
        position: relative;
        left: 50%;
        transform: translateX(-50%);
    }
    
    /* Ensure content stays centered when sidebar is collapsed */
    .main > .block-container {
        max-width: 800px;
        
        
        margin: 0 auto;
    }

    /* Ensure app content stays centered */
    .stApp > header {
        max-width: 800px;
        margin: 0 auto;
        
        
    }
    
    .user-message {
        background-color: white;
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 4px 18px;
        margin: 1rem 0 1rem auto;
        max-width: 70%;
        width: fit-content;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        border: 1px solid #e5e7eb;
    }
    
    .bot-message {
        background-color: white;
        color: black;
        padding: 1rem 1.5rem;
        border-radius: 18px 18px 18px 4px;
        margin: 1rem auto 1rem 0;
        max-width: 80%;
        width: fit-content;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
        border-left: 4px solid #F40009;
    }
    
    /* Add alternating background effect */
    .message-row {
        width: 100%;
        margin: 0;
        padding: 0;
    }
    
    .message-row:nth-child(odd) {
        background-color: #f7f7f8;
    }
    
    .message-row:nth-child(even) {
        background-color: white;
    }
    
    .bot-label {
        color: #F40009;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* SQL Query display */
    .sql-container {
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    /* Data table */
    .dataframe {
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
    
    /* Session items */
    .session-item {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .session-item:hover {
        background: #f3f4f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .session-item.active {
        background: #dbeafe;
        border-color: #3b82f6;
    }
    
    /* General Button Styling */
    .stButton > button {
        background: #f9f9f9;
        color: black;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 200;
        transition: all 0.2s ease;
    }
    
    /* Sidebar Button Layouts */
    .sidebar .stButton > button {
        width: 100%;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* New Chat Button */
    .sidebar .element-container:has(button:contains("New Chat")) button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 0.7rem 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
    }
    
    .sidebar .element-container:has(button:contains("New Chat")) button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.3);
    }
    
    /* Delete Button */
    .sidebar .element-container:has(button:contains("🗑️")) button {
        background: #ef4444;
        padding: 0.5rem;
        width: auto;
    }
    
    .sidebar .element-container:has(button:contains("🗑️")) button:hover {
        background: #dc2626;
        box-shadow: 0 2px 4px rgba(239, 68, 68, 0.3);
    }
    
    /* Database Status Button */
    .sidebar .element-container:has(button:contains("Database")) button {
        background: #10b981;
        font-size: 0.8rem;
        padding: 0.4rem 0.8rem;
        opacity: 0.9;
    }
    
    .sidebar .element-container:has(button:contains("Database")) button:hover {
        opacity: 1;
        background: #059669;
    }
    
    /* Session Buttons */
    .sidebar .element-container:has(button:contains("Chat")) button {
        background: white;
        color: #1f2937;
        border: 1px solid #e5e7eb;
        padding: 0.8rem 1rem;
        text-align: left;
        justify-content: flex-start;
        font-weight: normal;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .sidebar .element-container:has(button:contains("Chat")) button:hover {
        background: #f3f4f6;
        border-color: #d1d5db;
        transform: translateY(0);
    }
    
    /* Active Session Button */
    .sidebar .element-container:has(button.active) button {
        background: #f0f9ff;
        border-color: #3b82f6;
        color: #1d4ed8;
        font-weight: 500;
    }
    
    /* Hover effects for all sidebar buttons */
    .sidebar .stButton > button:hover {
        transform: translateY(-1px);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #9ca3af;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)



def load_logo_base64(image_path):
    """Load and encode logo image to base64."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode()
                logger.info(f"Logo loaded successfully from {image_path}")
                return f"data:image/png;base64,{encoded}"
        else:
            logger.warning(f"Logo file not found at {image_path}")
            return None
    except Exception as e:
        logger.error(f"Error loading logo: {e}")
        return None

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
        logger.info("Initializing DataChat AI agent")
        return ContextAwareSynapseAgent(AZURE_OPENAI_CONFIG)
    except Exception as e:
        logger.error(f"Failed to initialize DataChat AI: {e}")
        st.error(f"Failed to initialize DataChat AI: {e}")
        st.stop()

def init_session_state():
    """Initialize Streamlit session state with user isolation."""
    logger.info("Initializing session state with user isolation")

    # Generate unique user ID for session isolation
    if 'user_id' not in st.session_state:
        st.session_state.user_id = get_user_id()

    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None

    if 'current_thread_id' not in st.session_state:
        st.session_state.current_thread_id = None

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'agent' not in st.session_state:
        st.session_state.agent = init_agent()

    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()

    if 'show_welcome' not in st.session_state:
        st.session_state.show_welcome = True

    if 'current_followup_questions' not in st.session_state:
        st.session_state.current_followup_questions = []

    if 'first_message_sent' not in st.session_state:
        st.session_state.first_message_sent = False

    # Cache for loaded sessions to avoid reloading
    if 'loaded_sessions_cache' not in st.session_state:
        st.session_state.loaded_sessions_cache = {}

def display_header():
    """Display header without logo."""
    st.markdown("""
    <div class="app-header">
        <div class="header-content">
            <div class="app-title">
                <h1>DataChat AI</h1>
                <p>Data Analytics Assistant</p>
            </div>
        </div>
        <div class="bot-name">
             Assistant: DataBot
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_welcome_message():
    """Display welcome message for new sessions."""
    welcome_msg = st.session_state.agent.response_manager.get_welcome_message()
    
    st.markdown(f"""
    <div class="welcome-message">
        <div class="welcome-text">
            {welcome_msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_followup_questions(questions):
    """Display follow-up questions as clickable buttons."""
    if questions and len(questions) > 0:
        st.markdown("""
        <div class="followup-container">
            <div class="followup-title">Suggested follow-up questions:</div>
        </div>
        """, unsafe_allow_html=True)

        for i, question in enumerate(questions):
            if st.button(question, key=f"followup_{i}", width='stretch'):
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

def display_chat_message(message_data, is_user=True, message_index=0):
    """Display chat message with professional styling."""
    if is_user:
        st.markdown(f"""
        <div class="user-message">
            <strong>You:</strong><br>
            {message_data}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Get response text
        response_text = message_data.get('response', message_data.get('content', ''))
        
        # Display bot response inside the styled box
        st.markdown(f"""
        <div class="bot-message">
            <div class="bot-label"> DataBot</div>
            <div style="margin-top: 10px; line-height: 1.6;">
                {response_text.replace('**', '<strong>').replace('**', '</strong>').replace('\n', '<br>')}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # # Show SQL query
        # if message_data.get('sql_query') and message_data.get('sql_query') not in ["Error occurred", "Error in SQL generation", ""]:
        #     with st.expander("🔍 Generated T-SQL Query", expanded=False):
        #         st.code(message_data.get('sql_query'), language='sql')

        # Initialize dataframe variable
        dataframe = pd.DataFrame()  # Start with empty DataFrame
        
        # Safely get and validate dataframe
        raw_dataframe = message_data.get('dataframe')
        
        if raw_dataframe is not None:
            if isinstance(raw_dataframe, pd.DataFrame):
                # It's already a DataFrame
                dataframe = raw_dataframe
                logger.debug(f"Using existing DataFrame with {len(dataframe)} rows")
            elif isinstance(raw_dataframe, str) and raw_dataframe.strip():
                # It's a non-empty string, try multiple parsing approaches
                dataframe = parse_dataframe_from_string(raw_dataframe)
            elif isinstance(raw_dataframe, (list, dict)):
                # It's already parsed JSON data
                try:
                    dataframe = pd.DataFrame(raw_dataframe)
                    logger.debug(f"Successfully converted parsed data to DataFrame with {len(dataframe)} rows")
                except Exception as e:
                    logger.warning(f"Could not convert parsed data to DataFrame: {e}")
                    dataframe = pd.DataFrame()

        # Display results table only if we have data
        if not dataframe.empty:
            # Show data with pagination for large results
            if len(dataframe) > 20:
                st.dataframe(dataframe.head(20), use_container_width=True, hide_index=True)
                st.caption(f"Showing first 20 rows of {len(dataframe)} total results")
            else:
                st.dataframe(dataframe, use_container_width=True, hide_index=True)

            # Download button
            csv = dataframe.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"synapse_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"download_{message_index}"
            )

        # Display generated chart
        chart_code = message_data.get('chart_code')
        if chart_code and chart_code.strip() and not chart_code.startswith("#"):
            with st.expander("📊 Generated Chart", expanded=True):
                try:
                    if not dataframe.empty:
                        # Prepare the execution scope with figure size settings
                        plt.rcParams['figure.figsize'] = [3.6, 3.5]
                        plt.rcParams['figure.dpi'] = 100
                        exec_scope = {
                            'df': dataframe,
                            'pd': pd,
                            'plt': plt,
                            'np': np
                        }
                        # Execute the generated code
                        exec(chart_code, exec_scope)
                        
                        # Retrieve and style the figure object
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

        # Display follow-up questions for the latest message
        if message_index == len(st.session_state.chat_history) - 1:
            follow_up_questions = message_data.get('follow_up_questions', [])
            if follow_up_questions:
                st.session_state.current_followup_questions = follow_up_questions

def process_user_input(user_input):
    """Process user input and update chat history with user isolation."""
    logger.info(f"User {st.session_state.user_id} action: Submitted query - {user_input[:100]}...")

    # Hide welcome message after first input
    st.session_state.show_welcome = False

    # Check if this is the first message in the session
    is_first_message = not st.session_state.first_message_sent

    # Add user message to chat
    user_message = {
        'type': 'user',
        'content': user_input,
        'timestamp': datetime.now().isoformat()
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

    # Process query with context-aware agent
    processing_msg = st.session_state.agent.response_manager.get_processing_message()
    with st.spinner(processing_msg):
        result = st.session_state.agent.process_query(
            user_input, 
            st.session_state.current_session_id,
            st.session_state.current_thread_id
        )

    print(f"DEBUG: Agent result keys: {result.keys()}")
    print(f"DEBUG: Response from agent: {result.get('response', 'NO RESPONSE')[:200]}...")

    # Store bot response with all data
    bot_message = {
        'type': 'bot',
        'content': result['response'],
        'response': result['response'],
        'sql_query': result.get('sql_query', ''),
        'dataframe': result.get('dataframe', pd.DataFrame()),
        'chart_code': result.get('chart_code', ''),
        'follow_up_questions': result.get('follow_up_questions', []),
        'timestamp': datetime.now().isoformat()
    }

    print(f"DEBUG: Bot message response: {bot_message['response'][:200]}...")
    
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

def sidebar_interface():
    """Sidebar with session management and logo with user isolation."""
    with st.sidebar:
        logo_path = "img.png" 
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
            if st.button("➕ New Chat", width='stretch'):
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

    # Display current follow-up questions
    if st.session_state.current_followup_questions:
        display_followup_questions(st.session_state.current_followup_questions)

    # Input area
    st.markdown("---")

    user_input = st.chat_input("💬 Ask DataBot about your data...", key="main_input")

    if user_input:
        # Clear follow-up questions when new input is entered
        st.session_state.current_followup_questions = []
        process_user_input(user_input)
        st.rerun()

def main():
    """Main application entry point."""
    try:
        logger.info("Starting DataChat AI application")

        # Load CSS
        load_professional_css()

        # Initialize session state
        init_session_state()

        # Sidebar
        sidebar_interface()

        # Main chat interface
        main_chat_interface()

        # # Footer
        # st.markdown("---")
        # st.markdown("""
        # <div style="text-align: center; color: #666; font-size: 0.85rem;">
        #     <p>Powered by YASH Technologies | DataChat AI v2.1 | Enterprise Analytics Solution</p>
        # </div>
        # """, unsafe_allow_html=True)

        logger.info("DataChat AI application loaded successfully")

    except Exception as e:
        logger.error(f"Main application error: {e}", exc_info=True)
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    # Run the application
    main()
