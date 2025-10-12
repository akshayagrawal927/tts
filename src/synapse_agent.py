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
import asyncio
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

from .state import AgentState
from .fuzzy_search import FuzzySearch
from .response_manager import WittyResponseManager
from .synapse_connection import SynapseConnectionManager
from .system_prompts import ENHANCED_FEW_SHOT_EXAMPLES,MAPPING_SCHEMA_PROMPT

logger = logging.getLogger(__name__)

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

        self.pending_clarification = None

        self.response_manager = WittyResponseManager()

        logger.info("Context-Aware Azure Synapse Agent initialized successfully")
        
    def _rephrase_message_with_llm(self, original_message: str, user_query: str = "", 
                                message_type: str = "general") -> str:
        """
        Uses LLM to rephrase messages with full context awareness.
        
        Args:
            original_message: The base message to rephrase
            user_query: The original user question for context
            message_type: Type of message (greeting, error, success, invalid_query)
        """
        logger.info(f"Rephrasing {message_type} message with user context")
        
        if not original_message or not isinstance(original_message, str):
            return ""

        rephrase_prompt = ChatPromptTemplate.from_template("""
    You are a professional communication specialist for Pulse AI, an AI data analytics assistant.

    **CONTEXT:**
    - Message Type: {message_type}
    - User's Original Query: "{user_query}"
    - Base Response Template: "{original_message}"

    **YOUR TASK:**
    Rephrase the base response to be:
    1. **Contextually relevant** to the user's specific query
    2. **Personalized** - reference specific elements from their question
    3. **Warm and professional** - friendly but business-appropriate
    4. **Culturally sensitive** - suitable for Middle East Gulf Countries audience
    5. **Factually accurate** - preserve all data, numbers, and technical details

    **RESPONSE GUIDELINES BY TYPE:**

    **GREETING_ONLY:**
    - Welcome warmly and briefly introduce Pulse AI's capabilities
    - Reference any greeting in their query naturally
    - Offer specific examples of what you can help with
    -Example : 👋 Hello Leader! You're now plugged into Pulse AI—where questions meet instant intelligence. Consider me your strategy co-pilot.”
                                                           
    **MIXED_INTENT (greeting + data request):**
    - Acknowledge greeting briefly (1 sentence max)
    - Immediately pivot to addressing their data request
    - Show you understood both parts of their query

    **INVALID_QUERY:**
    - Politely acknowledge their question
    - Explain you specialize in data analytics for their business
    - Redirect to data-related capabilities with specific examples
    - Keep tone helpful, not dismissive

    **ERROR:**
    - Acknowledge what they were trying to do specifically
    - Explain the issue in simple terms
    - Provide actionable next steps

    **SUCCESS:**
    - Celebrate the successful query with relevant context
    - Highlight key findings from their specific request
    - Maintain professional enthusiasm



    **CRITICAL RULES:**
    - Preserve ALL factual information: numbers, names, dates
    - Keep markdown formatting for emphasis
    - Don't add information not in the original message
    - Reference the user's query naturally where appropriate
    - Keep response concise (2-4 sentences for greetings, longer for data responses)

    **OUTPUT:**
    Provide only the rephrased message with no meta-commentary.
    """)

        try:
            formatted_prompt = rephrase_prompt.format(
                message_type=message_type,
                user_query=user_query,
                original_message=original_message
            )
            
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            rephrased_content = response.content.strip()

            if not rephrased_content:
                logger.warning("LLM rephrasing returned empty. Using original.")
                return original_message

            logger.info(f"Successfully rephrased {message_type} message")
            return rephrased_content
            
        except Exception as e:
            logger.error(f"Failed to rephrase message: {e}")
            return original_message

    def _check_for_no_context_error(self, state: AgentState) -> Literal["error", "continue"]:
        """Checks if the SQL generation step resulted in a NO_CONTEXT_ERROR."""
        if state.get("sql_query") == "NO_CONTEXT_ERROR":
            return "error"
        return "continue"

    def _handle_no_context_error(self, state: AgentState) -> AgentState:
        """Handles the specific error when context is needed but not available."""
        logger.warning("Handling NO_CONTEXT_ERROR. Bypassing SQL execution.")
        
        base_response = "No previous query context available. Please provide a specific query first."
        state["final_response"] = self._rephrase_message_with_llm(
    base_response,
    user_query=state.get("user_question", ""),
    message_type="ERROR"
)
        state["query_result"] = pd.DataFrame() # Ensure result is empty
        state["chart_code"] = ""
        state["follow_up_questions"] = [
            "How many orders have been placed year-to-date for Fanta Orange 1.49L 2x6 PET AED?",
            "What is the average number of orders per day this quarter?",
            "What's the breakdown of orders by sales channel?"
        ]
        return state
    
    def _classify_intent_with_llm(self, state: AgentState) -> AgentState:
        """Use LLM to classify user intent and route appropriately."""
        user_question = state["user_question"]
        
        # Check for pending clarification first
        pending_clarification = None
        if hasattr(self, 'pending_clarification') and self.pending_clarification:
            pending_clarification = self.pending_clarification
        elif 'pending_clarification' in st.session_state:
            pending_clarification = st.session_state['pending_clarification']
        
        logger.info(f"Classifying intent for: {user_question[:100]}...")
        
        intent_prompt = ChatPromptTemplate.from_template("""
You are an intent classification expert for a data analytics chatbot.

Analyze the user's query and classify it into ONE primary intent:

USER QUERY: "{user_question}"

HAS PENDING CLARIFICATION: {has_pending}

INTENT CATEGORIES:
1. **CLARIFICATION_RESPONSE** - User responding to a clarification question
2. **GREETING_ONLY** - Simple greetings without questions
   - Examples: "hi", "hello", "hey", "good morning", "good evening", "hola"
3. **CAPABILITIES_QUERY** - Questions about bot capabilities/introduction
   - Examples: "what can you do?", "how can you help", "introduce yourself", 
               "what is this bot", "how to start", "help me get started",
               "tell me about yourself", "what are your features"
4. **CONVERSATIONAL** - General conversation, language preferences, feedback
   - Examples: "in english pls", "speak spanish", "thanks", "that's helpful"
5. **MIXED_INTENT** - Greeting + Data request (e.g., "hi, show me sales data")
6. **DATA_QUERY** - Any request for data, analysis, or SQL query
7. **CHART_MODIFICATION** - Request to modify existing chart
8. **INVALID_QUERY** - Non-data questions (trivia, general knowledge)

CRITICAL RULES:
- "what can you do" = CAPABILITIES_QUERY (NOT DATA_QUERY)
- "who are you" = CAPABILITIES_QUERY
- "tell me about yourself" = CAPABILITIES_QUERY
- "how to start" = CAPABILITIES_QUERY
- "help me" = CAPABILITIES_QUERY
- Simple greetings alone ("hi", "hola") = GREETING_ONLY
- Language requests = CONVERSATIONAL
- Thanks/feedback = CONVERSATIONAL
- If asking about BOT capabilities = CAPABILITIES_QUERY
- If asking about DATA/business metrics = DATA_QUERY

Respond ONLY with valid JSON:
{{
    "primary_intent": "CLARIFICATION_RESPONSE|GREETING_ONLY|CAPABILITIES_QUERY|CONVERSATIONAL|MIXED_INTENT|DATA_QUERY|CHART_MODIFICATION|INVALID_QUERY",
    "contains_greeting": true/false,
    "contains_capabilities_query": true/false,
    "contains_data_request": true/false,
    "reasoning": "Brief explanation",
    "extracted_data_query": "If MIXED_INTENT, extract only the data request part"
}}
""")

        try:
            formatted_prompt = intent_prompt.format(
                user_question=user_question,
                has_pending=bool(pending_clarification)
            )
            
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
                
                state["primary_intent"] = intent_data.get("primary_intent", "DATA_QUERY")
                state["contains_greeting"] = intent_data.get("contains_greeting", False)
                state["contains_data_request"] = intent_data.get("contains_data_request", False)
                state["intent_reasoning"] = intent_data.get("reasoning", "")
                
                # For MIXED_INTENT, extract the data query part
                if state["primary_intent"] == "MIXED_INTENT" and intent_data.get("extracted_data_query"):
                    extracted = intent_data["extracted_data_query"].strip()
                    if extracted:
                        logger.info(f"Mixed intent detected. Original: '{user_question}' -> Extracted: '{extracted}'")
                        state["user_question"] = extracted
                
                logger.info(f"Intent: {state['primary_intent']} | Reasoning: {state['intent_reasoning']}")
                
                # Set legacy flags for compatibility
                state["is_greeting"] = (state["primary_intent"] == "GREETING_ONLY")
                state["is_valid_query"] = (state["primary_intent"] not in ["GREETING_ONLY", "INVALID_QUERY", "CONVERSATIONAL"])
                
                return state
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Safe fallback
            state["primary_intent"] = "DATA_QUERY"
            state["contains_greeting"] = False
            state["contains_data_request"] = True
            state["is_greeting"] = False
            state["is_valid_query"] = True
            
        return state
    
    def _route_by_intent(self, state: AgentState) -> Literal["greeting", "conversational", "invalid", "clarification", "resolve_clarification", "chart_modification", "data_query"]:
        """Route based on LLM-classified intent."""
        intent = state.get("primary_intent", "DATA_QUERY")
        
        logger.info(f"Routing decision for intent: {intent}")
        
        # Priority routing
        if intent == "CLARIFICATION_RESPONSE":
            return "resolve_clarification" 
        elif intent == "GREETING_ONLY":
            return "greeting"
        elif intent == "CAPABILITIES_QUERY":  
            return "capabilities"
        elif intent == "CONVERSATIONAL":
            return "conversational"
        elif intent == "CHART_MODIFICATION":
            return "chart_modification"
        elif intent == "INVALID_QUERY":
            return "invalid"
        elif intent in ["DATA_QUERY", "MIXED_INTENT"]:
            return "data_query"
        
        # Default fallback
        return "data_query"

    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow with parallel processing."""
        logger.debug("Creating LangGraph workflow with parallel processing")
        workflow = StateGraph(AgentState)

        # Add all nodes
        workflow.add_node("classify_intent", self._classify_intent_with_llm)
        workflow.add_node("resolve_clarification", self._resolve_clarification)
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("fuzzy_search", self._fuzzy_search)
        workflow.add_node("extract_intent", self._extract_intent)
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("execute_sql", self._execute_sql)  
        workflow.add_node("parallel_processor", self._parallel_processor)  
        workflow.add_node("generate_followup", self._generate_followup_questions)
        workflow.add_node("handle_error", self._handle_error)
        workflow.add_node("handle_invalid", self._handle_invalid_query)
        workflow.add_node("handle_clarification", self._handle_clarification)
        workflow.add_node("modify_chart", self._modify_existing_chart)
        workflow.add_node("handle_greeting", self._handle_greeting)
        workflow.add_node("handle_conversational", self._handle_conversational)
        workflow.add_node("handle_capabilities", self._handle_capabilities)
        workflow.add_node("handle_no_context_error", self._handle_no_context_error)

        # Set entry point
        workflow.set_entry_point("classify_intent")

        # Intent routing
        workflow.add_conditional_edges(
        "classify_intent",
        self._route_by_intent,
        {
            "greeting": "handle_greeting",
            "capabilities": "handle_capabilities",  
            "conversational": "handle_conversational",
            "invalid": "handle_invalid",
            "clarification": "handle_clarification",
            "resolve_clarification": "resolve_clarification",
            "chart_modification": "modify_chart",
            "data_query": "analyze_query"
        }
    )
        
        # Terminal nodes
        workflow.add_edge("handle_greeting", END)
        workflow.add_edge("handle_capabilities", END) 
        workflow.add_edge("handle_conversational", END)
        workflow.add_edge("handle_invalid", END)
        workflow.add_edge("handle_clarification", END)
        workflow.add_edge("modify_chart", "parallel_processor")  
        workflow.add_edge("resolve_clarification", "analyze_query")

        # Main data processing flow
        workflow.add_edge("analyze_query", "fuzzy_search")

        workflow.add_conditional_edges(
            "fuzzy_search",
            self._check_clarification_needed,
            {
                "clarify": "handle_clarification",
                "continue": "extract_intent"
            }
        )

        workflow.add_edge("extract_intent", "check_chart_modification")
        workflow.add_node("check_chart_modification", lambda state: state)
        
        workflow.add_conditional_edges(
            "check_chart_modification",
            self._detect_chart_modification_request,
            {
                "modify_chart": "modify_chart",
                "generate_sql": "generate_sql"
            }
        )
        
        workflow.add_conditional_edges(
            "generate_sql",
            self._check_for_no_context_error,
            {
                "error": "handle_no_context_error",
                "continue": "execute_sql"
            }
        )
        
        workflow.add_edge("handle_no_context_error", END)
        
        # ✅ KEY CHANGE: After SQL execution, go to parallel processor
        workflow.add_conditional_edges(
            "execute_sql",
            self._check_sql_execution,
            {
                "success": "parallel_processor",  # ✅ Changed from "generate_response"
                "error": "handle_error"
            }
        )
        
        # ✅ NEW: Parallel processor handles response, chart, and followup in parallel
        workflow.add_edge("parallel_processor", "generate_followup")
        
        workflow.add_edge("handle_error", END)
        workflow.add_edge("generate_followup", END)

        return workflow.compile()

    def _extract_entities_for_fuzzy_search(self, user_question: str) -> Dict[str, List[str]]:
        """Use LLM to extract potential product and customer names from a query."""
        logger.info(f"Extracting entities from query: {user_question}")
        
        ner_prompt = ChatPromptTemplate.from_template("""
You are a highly accurate Named Entity Recognition (NER) model for a beverage/retail database.

User Query: "{user_question}"

Extract ONLY clear product names and customer/company names from the user's query.

RULES:
1. **Products**: Look for beverage names, brands, sizes, packaging types
   - Examples: "Sprite Zero", "Coca Cola 330ml", "Fanta Orange Can"
   - Include size/packaging if mentioned: "330ml", "Sleek Can", "4x6 NP"
   
2. **Customers**: Look for company names, business entities, trading names
   - Examples: "DUBAI GATE", "Trading Company", "F/S & TRADING"
   - Include business suffixes: "TRADING", "COMPANY", "HOTEL"

3. **DO NOT extract**:
   - Generic terms: "orders", "sales", "data", "revenue"
   - Time periods: "Q1", "Q2", "last year", "this month"
   - Actions: "show", "list", "compare", "analyze"

4. **BE CONSERVATIVE**: Only extract if you're confident it's a product or customer name

Return JSON with "products" and "customers" arrays. If unsure, return empty arrays.

Examples:
Query: "Show sales for Sprite Zero and DUBAI GATE TRADING"
Response: {{"products": ["Sprite Zero"], "customers": ["DUBAI GATE TRADING"]}}

Query: "Compare Q1 vs Q2 revenue trends"  
Response: {{"products": [], "customers": []}}

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
        """Perform fuzzy search and decide on a strategy: exact, like, or clarify."""
        user_question = state["user_question"]
        logger.info(f"Performing fuzzy search and strategy selection for: '{user_question}'")
 
        extracted_entities = self._extract_entities_for_fuzzy_search(user_question)
        product_entities = extracted_entities.get("products", [])
        customer_entities = extracted_entities.get("customers", [])
 
        clarifications_needed = {}
        like_suggestions = {}
        strategy = {}
        corrected_question = user_question
 
        all_entities = [("product", p) for p in product_entities] + [("customer", c) for c in customer_entities]
 
        # --- THE FIX IS HERE ---
        # The threshold was too high at 0.90, which rejected good matches in the 80-89% range.
        # Lowering it to 0.80 will allow the logic to trigger correctly.
        HIGH_CONFIDENCE_THRESHOLD = 0.80
 
        for entity_type, entity in all_entities:
            matches = self.fuzzy_search_system.find_best_matches(entity, entity_type)
 
            if not matches:
                strategy[entity] = "exact" # Fallback to use the original term
                logger.warning(f"No fuzzy matches for '{entity}'. Proceeding with original term.")
                continue
 
            # Filter matches that meet our new, more reasonable threshold
            high_confidence_matches = [m for m in matches if m[1] >= HIGH_CONFIDENCE_THRESHOLD]
            fuzz_suggestion_count = len(high_confidence_matches)
 
            # --- Improved Logging ---
            top_5_matches_log = [(name, f"{score:.2f}") for name, score in matches[:5]]
            logger.info(f"Fuzzy matches for '{entity}': {top_5_matches_log}")
            logger.info(f"Found {fuzz_suggestion_count} high-confidence (> {HIGH_CONFIDENCE_THRESHOLD}) matches.")
 
            # Now, with the lower threshold, this logic will trigger correctly.
            if fuzz_suggestion_count >= 3:
                # Case: Ambiguous, requires user input
                strategy[entity] = "clarify"
                clarifications_needed[entity] = [m[0] for m in high_confidence_matches[:5]]
                logger.info(f"Strategy for '{entity}': CLARIFY ({fuzz_suggestion_count} suggestions).")
 
            elif 1 <= fuzz_suggestion_count < 3:
                # Case: One or two strong matches, auto-correct with the best one
                best_match_name = high_confidence_matches[0][0]
                strategy[entity] = "like"
                like_suggestions[entity] = best_match_name
                logger.info(f"Strategy for '{entity}': LIKE (auto-correcting to '{best_match_name}')")
                # Use regex for safe, case-insensitive replacement
                corrected_question = re.sub(re.escape(entity), best_match_name, corrected_question, flags=re.IGNORECASE)
 
            else:
                # Case: No high-confidence matches, use the original term
                strategy[entity] = "exact"
                logger.warning(f"No high-confidence matches for '{entity}'. Using original term.")
 
        if "clarify" in strategy.values():
            state["needs_clarification"] = True
            prompt_parts = ["I found a few possibilities for your query. To give you the most accurate results, please clarify by selecting a number:"]
            for entity, suggestions in clarifications_needed.items():
                prompt_parts.append(f"\nWhich **'{entity}'** did you mean?")
                for i, suggestion in enumerate(suggestions, 1):
                    prompt_parts.append(f"  {i}. {suggestion}")
            state["clarification_prompt"] = "\n".join(prompt_parts)
 
            clarification_context = {"original_query": user_question, "entity_suggestions": clarifications_needed}
            self.pending_clarification = clarification_context
            st.session_state['pending_clarification'] = clarification_context
        else:
            state["needs_clarification"] = False
 
        state["fuzzy_search_strategy"] = strategy
        state["like_suggestions"] = like_suggestions
        state["corrected_question"] = corrected_question
        logger.info(f"Original: '{user_question}' -> Corrected for SQL: '{corrected_question}'")
        return state
    
    def _is_clarification_response(self, user_input: str) -> bool:
        """Check if user input is a clarification response using intelligent patterns."""
        
        # Log at entry
        print(f"DEBUG: _is_clarification_response CALLED with input: '{user_input}'")
        logger.info(f"DEBUG: _is_clarification_response CALLED with input: '{user_input}'")
        
        if not user_input:
            logger.info("❌ Empty input - not clarification")
            return False
            
        user_input_lower = user_input.lower().strip()
        
        # Check if we have pending clarification
        has_pending = bool(self.pending_clarification or st.session_state.get('pending_clarification'))
        
        logger.info(f"DEBUG: has_pending = {has_pending}")
        logger.info(f"DEBUG: self.pending_clarification = {self.pending_clarification is not None}")
        logger.info(f"DEBUG: st.session_state pending = {st.session_state.get('pending_clarification') is not None}")
        
        if not has_pending:
            logger.info("❌ No pending clarification - not clarification response")
            return False
        
        logger.info(f"🔍 Checking if clarification response: '{user_input}'")
        
        # **Word-to-number mapping**
        number_words = {
                'one': '1', 'first': '1','1st': '1',
                'two': '2', 'second': '2','2nd': '2',
                'three': '3', 'third': '3','3rd': '3',
                'four': '4', 'fourth': '4','4th': '4',
                'five': '5', 'fifth': '5','5th': '5',
            }
        
        # Pattern 1: Simple single number (1-10)
        logger.info(f"DEBUG: Testing Pattern 1")
        if user_input_lower.isdigit():
            num = int(user_input_lower)
            if 1 <= num <= 10:
                logger.info(f"✅ Pattern 1: Single number clarification: {num}")
                return True
        
        # Pattern 2: Single number word
        logger.info(f"DEBUG: Testing Pattern 2")
        if user_input_lower in number_words:
            logger.info(f"✅ Pattern 2: Single number word clarification: {user_input_lower}")
            return True
        
        # Pattern 3: Multiple numbers/words with separators
        logger.info(f"DEBUG: Testing Pattern 3")
        
        # Convert number words to digits first
        converted_input = user_input_lower
        for word, digit in number_words.items():
            converted_input = re.sub(r'\b' + word + r'\b', digit, converted_input)
        
        logger.info(f"   Converted: '{user_input_lower}' → '{converted_input}'")
        
        # Remove separators
        cleaned = re.sub(r'\b(and|or|&|,)\b', ' ', converted_input)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        tokens = cleaned.split()
        
        logger.info(f"   Cleaned: '{cleaned}'")
        logger.info(f"   Tokens: {tokens}")
        
        # Count numeric tokens
        number_tokens = [t for t in tokens if t.isdigit()]
        logger.info(f"   Number tokens: {number_tokens}")
        
        if number_tokens:
            ratio = len(number_tokens) / len(tokens) if tokens else 0
            logger.info(f"   Ratio: {ratio:.2f} (threshold: 0.4)")
            
            if ratio >= 0.4:
                logger.info(f"✅ Pattern 3: Multi-number clarification detected")
                return True
            else:
                logger.info(f"   Pattern 3 failed: ratio {ratio:.2f} < 0.4")
        else:
            logger.info(f"   Pattern 3 failed: no number tokens found")
        
        # Pattern 4: Explicit selection phrases
        logger.info(f"DEBUG: Testing Pattern 4")
        confirmation_phrases = [
            'yes', 'correct', 'right', 'exactly', 'yep', 'yeah',
            'first one', 'second one', 'third one', 'fourth one', 'fifth one',
            'that one', 'this one', 'both', 'all of them',
            'the first', 'the second', 'the third'
        ]
        
        matched_phrases = [phrase for phrase in confirmation_phrases if phrase in user_input_lower]
        if matched_phrases:
            logger.info(f"✅ Pattern 4: Confirmation phrase detected: {matched_phrases}")
            return True
        
        # Pattern 5: "I meant" or "I want" constructions
        logger.info(f"DEBUG: Testing Pattern 5")
        intent_phrases = ['i meant', 'i want', 'i choose', 'i pick', 'i select',
                        'select', 'choose', 'pick', 'go with', 'use']
        
        matched_intent = [phrase for phrase in intent_phrases if phrase in user_input_lower]
        if matched_intent:
            logger.info(f"✅ Pattern 5: Intent phrase detected: {matched_intent}")
            return True
        
        # Pattern 6: Very short responses with pending clarification
        logger.info(f"DEBUG: Testing Pattern 6 (length: {len(user_input)})")
        if len(user_input) <= 15:
            has_digit = bool(re.search(r'\d', user_input_lower))
            has_number_word = any(word in user_input_lower for word in number_words.keys())
            
            logger.info(f"   Has digit: {has_digit}, Has number word: {has_number_word}")
            
            if has_digit or has_number_word:
                logger.info(f"✅ Pattern 6: Short input with number element")
                return True
        
        logger.info(f"❌ No pattern matched - not a clarification response")
        return False
    
    def _process_clarification_response(self, user_response: str, original_query: str, suggestions: dict) -> str:
        """
        Use LLM to intelligently understand user's clarification intent.
        Handles: digits, number words, entity names, natural language.
        """
        
        logger.info(f"Processing clarification: '{user_response}'")
        logger.info(f"Original query: '{original_query}'")
        logger.info(f"Suggestions: {suggestions}")
        
        # Build clear mapping for LLM
        suggestion_mapping = []
        entity_order = list(suggestions.keys())
        
        for idx, (entity, options) in enumerate(suggestions.items(), 1):
            suggestion_mapping.append(f"\n**Entity {idx}: '{entity}'**")
            for opt_idx, option in enumerate(options, 1):
                suggestion_mapping.append(f"  {opt_idx}. {option}")
        
        suggestions_text = "\n".join(suggestion_mapping)
        
        clarification_prompt = ChatPromptTemplate.from_template("""
    You are resolving ambiguity in a database query.

    **ORIGINAL QUERY:**
    "{original_query}"

    **AMBIGUOUS ENTITIES WITH OPTIONS:**
    {suggestions_text}

    **USER'S CLARIFICATION:**
    "{user_response}"

    **RULES FOR MAPPING:**

    1. **Numbers (digits or words):** Map to option positions
    - "2" or "two" or "second" → Option 2
    - "2 and 1" or "second and one" → Entity 1 gets option 2, Entity 2 gets option 1
    - "first and third" → Entity 1 gets option 1, Entity 2 gets option 3

    2. **Natural Language:** Understand descriptive references
    - "the Dubai one" → Match to option containing "DUBAI"
    - "the 4x6 package" → Match to option containing "4x6"

    3. **Multiple Entities:** Numbers/words apply in order
    - With 2 entities, "2 and 1" means Entity 1→option 2, Entity 2→option 1

    4. **Ordinal/Cardinal Equivalence:**
    - "first" = "one" = "1"
    - "second" = "two" = "2"
    - "third" = "three" = "3"

    **OUTPUT:**
    Return ONLY the corrected query with exact option names substituted.

    In clarification_prompt add these examples :
 
    **EXAMPLES:**
 
    Example 1:
    User: "2 and 1"
    Entity 1: Sprite [1. Zero, 2. Regular]
    Entity 2: Customer [1. DUBAI GATE, 2. AL FATH]
    Output: Replace with "Regular" and "DUBAI GATE"
 
    Example 2:
    User: "second and one"
    (Same entities as above)
    Output: Replace with "Regular" and "DUBAI GATE"
 
    Example 3:
    User: "first product and the Dubai customer"
    Output: Replace with "Zero" and "DUBAI GATE"
                                                               
    Example 4:
    User: "I meant the 4x6 package and the trading company"
    Entity 1: Fanta [1. Orange Can, 2. Orange 4x6 NP]
    Entity 2: Customer [1. DUBAI GATE TRADING, 2    . AL FATH]
    Output: Replace with "Orange 4x6 NP" and "DUBAI GATE TRADING"
                                                               
    Example 5:
    User: "the first one"                          
    Entity 1: Coke [1. Diet, 2. Regular]
    Output: Replace with "Diet"
    Example 6:
    User:  "1 & 2"
    Entity 1: Sprite [1. Zero, 2. Regular]
    Entity 2: Customer [1. DUBAI GATE, 2. AL FATH]
    Output: Replace with "Zero" and "AL FATH"
                                                               
    Example 6
    User: "2 and first"
    Entity 1: Sprite [1. Zero, 2. Regular]
    Entity 2: Customer [1. DUBAI GATE, 2. AL FATH]
    Output: Replace with "Regular" and "DUBAI GATE"
    """)

        try:
            formatted_prompt = clarification_prompt.format(
                original_query=original_query,
                suggestions_text=suggestions_text,
                user_response=user_response
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            corrected_query = response.content.strip()
            
            # Clean up
            corrected_query = re.sub(r'^```.*\n|```$', '', corrected_query, flags=re.MULTILINE).strip()
            corrected_query = re.sub(r'^(Output:|Corrected Query:)\s*', '', corrected_query, flags=re.IGNORECASE).strip()
            corrected_query = corrected_query.strip('"\'')
            
            logger.info(f"✓ LLM clarification successful: '{corrected_query}'")
            return corrected_query

        except Exception as e:
            logger.error(f"LLM clarification failed: {e}")
            
            # **ENHANCED FALLBACK with word-to-number conversion**
            number_words = {
                'one': '1', 'first': '1',
                'two': '2', 'second': '2',
                'three': '3', 'third': '3',
                'four': '4', 'fourth': '4',
                'five': '5', 'fifth': '5'
            }
            
            # Convert number words to digits
            converted_response = user_response.lower()
            for word, digit in number_words.items():
                converted_response = re.sub(r'\b' + word + r'\b', digit, converted_response)
            
            # Extract numbers
            numbers = re.findall(r'\d+', converted_response)
            logger.info(f"Fallback: Converted '{user_response}' → numbers: {numbers}")
            
            if numbers:
                corrected_query = original_query
                entity_list = list(suggestions.items())
                
                for i, number_str in enumerate(numbers):
                    if i >= len(entity_list):
                        break
                        
                    entity, options = entity_list[i]
                    selection_num = int(number_str) - 1
                    
                    if 0 <= selection_num < len(options):
                        selected_option = options[selection_num]
                        corrected_query = re.sub(
                            re.escape(entity),
                            selected_option,
                            corrected_query,
                            count=1,
                            flags=re.IGNORECASE
                        )
                        logger.info(f"Fallback mapped: '{entity}' → '{selected_option}'")
                
                logger.info(f"✓ Fallback clarification: '{corrected_query}'")
                return corrected_query
            
            logger.error("All clarification methods failed")
            return original_query
        
    def _check_clarification_needed(self, state: AgentState) -> Literal["clarify", "continue"]:
        """Check if fuzzy search requires user clarification."""
        if state["needs_clarification"]:
            return "clarify"
        return "continue"

    def _handle_clarification(self, state: AgentState) -> AgentState:
        """Handle the case where fuzzy search needs user clarification."""
        logger.info("Handling clarification prompt for the user.")
        base_response = state["clarification_prompt"]
        state["final_response"] = self._rephrase_message_with_llm(
    base_response,
    user_query=state.get("user_question", ""),
    message_type="CLARIFICATION"
)
        # Clear other fields as we are stopping here to wait for user input
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["follow_up_questions"] = []
        state["chart_code"] = ""
        return state

    
    def _is_clarification_response(self, user_input: str) -> bool:
        """Check if user input is a clarification response."""
        user_input_lower = user_input.lower().strip()
        
        # Check for numbered responses
        if user_input_lower.isdigit() and int(user_input_lower) <= 5:
            return True
            
        # Check for confirmation words
        confirmation_words = ['yes', 'correct', 'right', 'exactly', 'that one', 'first one', 'second one', 'third one']
        if any(word in user_input_lower for word in confirmation_words):
            return True
            
        # Check if the input contains "i meant" or similar clarification phrases
        clarification_phrases = ['i meant', 'i want', 'i choose', 'select', 'pick']
        if any(phrase in user_input_lower for phrase in clarification_phrases):
            return True
            
        return False

    def _handle_invalid_query(self, state: AgentState) -> AgentState:
        """Handle invalid queries with context-aware LLM response."""
        user_query = state["user_question"]
        logger.info(f"Handling invalid query with personalization: {user_query[:50]}")
        
        base_response = self.response_manager.get_error_message('invalid_query')
        state["final_response"] = self._rephrase_message_with_llm(
            base_response,
            user_query=user_query,
            message_type="INVALID_QUERY"
        )
        
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["follow_up_questions"] = []
        state["chart_code"] = ""
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        return state
        
    def _handle_greeting(self, state: AgentState) -> AgentState:
        """Handle greetings with comprehensive capability explanation."""
        user_query = state["user_question"]
        logger.info(f"Handling greeting with capability explanation: {user_query}")
        
        # Generate a detailed, helpful greeting response using LLM
        greeting_prompt = ChatPromptTemplate.from_template("""
    You are Pulse AI, a professional data analytics assistant for a Coca-Cola business intelligence system.

    **USER'S MESSAGE:**
    "{user_query}"

    **YOUR TASK:**
    Respond naturally and appropriately to the user's specific message. This could be:
    - A simple greeting ("hi", "hello") → Provide a warm welcome and brief overview
    - A question about your capabilities ("what can you do?") → Explain your features in detail
    - A question about getting started ("how do I start?") → Guide them on how to interact with you
    - A general question about you ("who are you?", "tell me about yourself") → Introduce yourself professionally

    **YOUR CAPABILITIES (use when relevant):**

    1. **SQL Query Generation**: Convert natural language questions into Azure Synapse T-SQL queries
    - Example: "Show sales for Sprite Zero last month" → Generates and executes SQL

    2. **Data Analysis & Insights**: Analyze query results and provide business intelligence
    - Identify trends, patterns, and anomalies
    - Calculate key metrics (totals, averages, growth rates)
    - Provide actionable recommendations

    3. **Visualizations**: Create charts and graphs when requested
    - Bar charts for comparisons
    - Line charts for trends over time
    - Pie charts for distributions
    - Histograms for data distributions
    - Box plots for statistical analysis

    4. **Available Data**: You have access to:
    - Sales data (orders, revenue, volumes)
    - Customer information (names, locations, segments)
    - Product catalog (beverages, packages, SKUs)
    - Time-based analysis (daily, monthly, quarterly, yearly)

    5. **Smart Conversation**: Context-aware follow-ups and clarifications
    - Remembers previous queries in the conversation
    - Asks for clarification when needed
    - Suggests relevant next questions

    **RESPONSE GUIDELINES:**
    - **Match their tone**: If casual, be friendly. If formal, be professional.
    - **Be concise for simple greetings**: A simple "hi" deserves a brief, warm response
    - **Be detailed for capability questions**: If they ask "what can you do?", explain thoroughly
    - **Be specific**: Reference their actual words when appropriate
    - **Don't over-explain**: Only mention features relevant to their question
    - **Always end with an invitation**: Encourage them to ask something

    **EXAMPLES:**

    User: "hi"
    Response: "👋 Hello! I'm Pulse AI, your data analytics assistant. I can help you query your Coca-Cola business data, generate insights, and create visualizations. What would you like to explore today?"

    User: "what can you do?"
    Response: "Great question! I'm Pulse AI, and I specialize in helping you make sense of your business data. Here's what I can do:

     **Query Your Data**: Ask me questions in plain English like 'What's the breakdown of orders by sales channel?' and I'll generate and execute SQL queries

     **Provide Insights**: I analyze your data to find trends, patterns, and key metrics automatically

     **Create Visualizations**: Request charts and graphs to visualize your data

     **Smart Conversations**: I remember our conversation context and suggest relevant follow-up questions

    I have access to your sales data, customer information, and product catalog. What would you like to analyze first?"

    User: "how to start"
    Response: "Getting started is easy! Just ask me a question about your data in plain English. For example:

    - 'Show me sales for Sprite Zero this quarter'
    - 'Which customers ordered the most last month?'
    - 'What's the trend in Coca-Cola sales over the past year?'

    I'll handle the database querying, analysis, and even create charts if you'd like. Try asking me something about your sales, customers, or products!"

    User: "tell me about yourself"
    Response: "I'm Pulse AI, your AI-powered analytics companion built specifically for Coca-Cola's business intelligence needs. I'm designed to make data analysis simple and conversational.

    **What makes me different:**
    - I speak your language—no need to know SQL or technical jargon
    - I provide insights automatically, not just raw numbers
    - I can create visualizations on demand
    - I remember our conversation, so you can ask follow-up questions naturally

    Think of me as your personal data analyst who's always available. What business question can I help you answer today?"

    **Now respond to the user's message above:**
    """)

        try:
            formatted_prompt = greeting_prompt.format(user_query=user_query)
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            greeting_response = response.content.strip()
            
            state["final_response"] = greeting_response
            logger.info("Context-aware greeting response generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate greeting response: {e}")
            # Fallback to a contextual default
            if any(word in user_query.lower() for word in ['what', 'can', 'do', 'capabilities', 'features']):
                # User asking about capabilities
                state["final_response"] = """Great question! I'm Pulse AI, and here's what I can help you with:

    📊 **Query Your Data**: Ask questions in plain English, and I'll generate SQL queries to fetch the data you need

    🔍 **Analyze & Provide Insights**: I analyze results to highlight trends, patterns, and key metrics automatically

    📈 **Create Visualizations**: Request charts and graphs to visualize your data (bar charts, line charts, pie charts, etc.)

    💬 **Smart Conversations**: I remember our conversation context and suggest relevant follow-up questions

    **Available Data:**
    - Sales & Orders (revenue, volumes, order sources)
    - Customer Analytics (names, locations, preferences, segments)
    - Product Catalog (beverages, packages, SKUs)
    - Time-Based Trends (daily, monthly, quarterly, yearly analysis)

    What would you like to explore first?"""
            elif any(word in user_query.lower() for word in ['start', 'begin', 'how', 'help']):
                # User asking how to start
                state["final_response"] = """Getting started is simple! Just ask me a question about your data in natural language.

    **Try asking things like:**
    - "What's the breakdown of orders by sales channel?",
    - "Which customers placed the most orders in previous quarter?",
    - "Which day of the week sees the highest order volume?"

    I'll handle the database querying, analysis, and visualization. What would you like to know?"""
            else:
                # Simple greeting
                state["final_response"] = """👋 Hello! I'm Pulse AI, your intelligent data analytics assistant for Coca-Cola business insights.

    I can help you query your data, analyze trends, and create visualizations—all through natural conversation. What would you like to explore today?"""
        
        # Clear other response fields
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["chart_code"] = ""
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        state["follow_up_questions"] = [
            "What's the breakdown of orders by sales channel?",
            "Which customers placed the most orders in previous quarter?",
            "Which day of the week sees the highest order volume?",
        ]
        
        return state
    
    def _handle_capabilities(self, state: AgentState) -> AgentState:
        """Handle queries about bot capabilities."""
        user_query = state["user_question"]
        logger.info(f"Handling capabilities query: {user_query}")
        
        capabilities_prompt = ChatPromptTemplate.from_template("""
    You are Pulse AI explaining your capabilities to a user.

    **USER'S QUESTION:**
    "{user_query}"

    **YOUR CAPABILITIES:**

    1. **SQL Query Generation**: Convert natural language questions into Azure Synapse T-SQL queries
    - Example: "Show sales for Sprite Zero last month" → Generates and executes SQL

    2. **Data Analysis & Insights**: Analyze query results and provide business intelligence
    - Identify trends, patterns, and anomalies
    - Calculate key metrics (totals, averages, growth rates)
    - Provide actionable recommendations

    3. **Visualizations**: Create charts and graphs when requested
    - Bar charts for comparisons
    - Line charts for trends over time
    - Pie charts for distributions
    - Histograms for data distributions
    - Box plots for statistical analysis

    4. **Available Data**: You have access to:
    - Sales data (orders, revenue, volumes)
    - Customer information (names, locations, segments)
    - Product catalog (beverages, packages, SKUs)
    - Time-based analysis (daily, monthly, quarterly, yearly)

    5. **Smart Conversation**: Context-aware follow-ups and clarifications
    - Remembers previous queries in the conversation
    - Asks for clarification when needed
    - Suggests relevant next questions

    **YOUR TASK:**
    Respond naturally to the user's specific question about your capabilities. Be:
    - **Concise**: Don't list everything if they ask something specific
    - **Relevant**: Focus on what they're asking about
    - **Encouraging**: Invite them to try you out with an example
    - **Professional but friendly**

    **EXAMPLES:**

    User: "what can you do?"
    Response: "Great question! I'm Pulse AI, your data analytics assistant. Here's what I can do:

    📊 **Query Your Data**: Ask me questions in plain English like 'What's the breakdown of orders by sales channel?' and I'll generate and execute SQL queries

    🔍 **Provide Insights**: I analyze your data to find trends, patterns, and key metrics automatically

    📈 **Create Visualizations**: Request charts and graphs to visualize your data

    💬 **Smart Conversations**: I remember our conversation context and suggest relevant follow-up questions

    I have access to your sales data, customer information, and product catalog. What would you like to analyze first?"

    User: "how can you help me"
    Response: "I'm here to make data analysis simple! I can help you:

    - **Explore your data** through conversational questions (no SQL knowledge needed)
    - **Generate insights** automatically from your sales, customers, and products
    - **Create visualizations** to spot trends and patterns
    - **Answer follow-up questions** by remembering our conversation

    Just ask me something like 'Show me sales for Sprite Zero this quarter' or 'Which customers ordered the most last month' and I'll handle the rest. What would you like to know?"

    **Now respond to: "{user_query}"**

    Output only your response text, no meta-commentary.
    """)

        try:
            formatted_prompt = capabilities_prompt.format(user_query=user_query)
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            capabilities_response = response.content.strip()
            
            state["final_response"] = capabilities_response
            logger.info("Capabilities response generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate capabilities response: {e}")
            # Fallback response
            state["final_response"] = """Great question! I'm Pulse AI, and here's what I can help you with:

    📊 **Query Your Data**: Ask questions in plain English, and I'll generate SQL queries to fetch the data you need

    🔍 **Analyze & Provide Insights**: I analyze results to highlight trends, patterns, and key metrics automatically

    📈 **Create Visualizations**: Request charts and graphs to visualize your data (bar charts, line charts, pie charts, etc.)

    💬 **Smart Conversations**: I remember our conversation context and suggest relevant follow-up questions

    **Available Data:**
    - Sales & Orders (revenue, volumes, order sources)
    - Customer Analytics (names, locations, preferences, segments)
    - Product Catalog (beverages, packages, SKUs)
    - Time-Based Trends (daily, monthly, quarterly, yearly analysis)

    What would you like to explore first?"""
        
        # Clear query-related fields
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["chart_code"] = ""
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        state["follow_up_questions"] = [
            "What's the breakdown of orders by sales channel?",
            "Which customers placed the most orders this quarter?",
            "Show me sales trends for the past 6 months"
        ]
        
        return state
    
    def _handle_conversational(self, state: AgentState) -> AgentState:
        """Handle conversational messages that don't require data queries."""
        user_query = state["user_question"]
        logger.info(f"Handling conversational message: {user_query}")
        
        # Get conversation history for context
        recent_messages = []
        if st.session_state.chat_history:
            # Get last 3 messages for context
            recent_messages = st.session_state.chat_history[-3:]
        
        conversational_prompt = ChatPromptTemplate.from_template("""
    You are Pulse AI, a professional and friendly data analytics assistant.

    **CONVERSATION HISTORY (for context):**
    {conversation_history}

    **USER'S MESSAGE:**
    "{user_query}"

    **YOUR TASK:**
    Respond naturally to the user's conversational message. This could be:
    - A language preference request ("in english pls", "speak spanish") → Acknowledge and confirm language preference
    - Feedback or thanks ("thanks", "helpful", "great") → Respond graciously and offer to help more
    - A request for clarification ("I don't understand", "explain differently") → Offer to rephrase or explain
    - General conversation → Respond naturally while staying helpful

    **RESPONSE GUIDELINES:**
    1. **Be natural and conversational** - respond like a helpful colleague
    2. **Acknowledge their request** - show you understood what they said
    3. **Stay in role** - you're a data analytics assistant
    4. **Keep it brief** - 2-3 sentences max for simple messages
    5. **Offer to help** - end with an invitation to continue

    **EXAMPLES:**

    User: "in english pls"
    Response: "Of course! I'll respond in English from now on. What would you like to explore in your data?"

    User: "thanks"
    Response: "You're welcome! I'm here to help. Is there anything else you'd like to analyze or any other questions about your data?"

    User: "that's helpful"
    Response: "Glad I could help! Feel free to ask if you need any other insights or want to explore different aspects of your data."

    User: "I don't understand"
    Response: "No problem—let me explain it differently. Which part would you like me to clarify?"

    User: "can you speak spanish?"
    Response: "¡Por supuesto! Puedo responder en español. ¿Qué te gustaría explorar en tus datos?"

    **Now respond naturally to the user's message:**
    """)

        try:
            # Format conversation history for context
            history_text = ""
            for msg in recent_messages[-3:]:
                if msg['type'] == 'user':
                    history_text += f"User: {msg['content'][:100]}\n"
                elif msg['type'] == 'bot':
                    history_text += f"Assistant: {msg.get('response', msg.get('content', ''))[:100]}\n"
            
            formatted_prompt = conversational_prompt.format(
                conversation_history=history_text if history_text else "No previous conversation",
                user_query=user_query
            )
            
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            conversational_response = response.content.strip()
            
            state["final_response"] = conversational_response
            logger.info("Conversational response generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate conversational response: {e}")
            # Fallback responses based on common patterns
            user_lower = user_query.lower()
            
            if any(word in user_lower for word in ['english', 'inglés', 'ingles']):
                state["final_response"] = "Of course! I'll respond in English from now on. What would you like to explore?"
            elif any(word in user_lower for word in ['spanish', 'español', 'espanol']):
                state["final_response"] = "¡Por supuesto! Puedo responder en español. ¿Qué te gustaría saber?"
            elif any(word in user_lower for word in ['thanks', 'thank you', 'gracias']):
                state["final_response"] = "You're welcome! Happy to help. Anything else you'd like to know?"
            elif any(word in user_lower for word in ["don't understand", 'confused', 'unclear']):
                state["final_response"] = "No problem! Let me explain it differently. What specifically would you like me to clarify?"
            else:
                state["final_response"] = "I understand. How can I help you with your data today?"
        
        # Clear query-related fields
        state["sql_query"] = ""
        state["query_result"] = pd.DataFrame()
        state["chart_code"] = ""
        state["best_chart_type"] = ""
        state["visualization_suggestion"] = ""
        state["follow_up_questions"] = []
        
        return state

    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze query with intelligent LLM-based context awareness."""
        user_question = state["user_question"]
        logger.info(f"Analyzing query for context: {user_question[:50]}...")

        state["conversation_context"] = self.conversation_context
        state["previous_queries"] = self.query_history[-3:] if self.query_history else []

        # **IMPROVED: Better pattern matching for contextual queries**
        contextual_patterns = [
            r'\b(also|too)\b.*\b(show|display|get|fetch)',
            r'\b(first|next|previous)\s+\d+',  # "first 5", "next 10"
            r'\b(those|these|that|this|same)\b',
            r'\b(change|modify|update|adjust)\b.*(color|colour|chart|graph)',
        ]
        
        # **NEW: Patterns that are NOT contextual even if they match above**
        non_contextual_indicators = [
            r'\b(this year|last year|this month|last month|this quarter|last quarter)\b',  # Time references
            r'\b(show all|list all|get all)\b',  # Explicit "all" requests
            r'\bfor\b.*\b(product|customer)',  # "for [specific entity]"
        ]
        
        is_likely_contextual = any(
            re.search(pattern, user_question.lower()) 
            for pattern in contextual_patterns
        )
        
        # **NEW: Override contextual detection if non-contextual indicators present**
        if is_likely_contextual:
            has_non_contextual = any(
                re.search(pattern, user_question.lower())
                for pattern in non_contextual_indicators
            )
            if has_non_contextual:
                logger.info("Query has contextual keywords but is actually independent")
                is_likely_contextual = False
        
        if is_likely_contextual and self.query_history:
            logger.info("Quick pattern match: Query is contextual")
            if "conversation_context" not in state:
                state["conversation_context"] = {}
            state["conversation_context"]["needs_context"] = True
            state["conversation_context"]["context_type"] = "continuation"
            state["conversation_context"]["reasoning"] = "Contains contextual keywords and follows previous query"
            return state

        context_detection_prompt = ChatPromptTemplate.from_template("""
    You are an expert at understanding conversational context in data analytics queries.

    CONVERSATION HISTORY:
    Previous SQL Queries: {previous_queries}
    Last Entity Referenced: {last_entity}
    Last Table Used: {last_table}
    Last Filters: {last_conditions}

    CURRENT USER QUERY: "{current_query}"

    Analyze whether this query is:
    1. **INDEPENDENT** - A standalone question that doesn't need previous context
    2. **CONTEXTUAL** - Refers to or builds upon previous conversation

    INDEPENDENT queries (NEEDS CONTEXT = false):
    - Complete questions with all entities specified
    - Fresh topics unrelated to history
    - Explicit comparisons: "Q1 vs Q2", "product A vs product B"
    - **Time-based queries with full context: "this year", "last month", "Q2 2024"**
    - **Queries with "all" or "list all": "show all orders", "list all customers"**
    - **Queries specifying both customer AND product: "orders by [customer] for [product]"**

    CONTEXTUAL queries (NEEDS CONTEXT = true):
    - Contains words like: "also", "too", "as well" (without complete context)
    - Uses pronouns without antecedent: "that", "those", "this", "these", "them", "it", "same"
    - Continuation: "show more", "next results", "continue"
    - Modifications: "change color", "make it bigger", "update the chart"
    - Incomplete references: "first 5" (without saying "first 5 of what")

    **CRITICAL**: If query mentions "this year", "last year", "this quarter" with a specific customer/product, it is INDEPENDENT.

    Respond in JSON format:
    {{
        "needs_context": true/false,
        "reasoning": "Brief explanation",
        "context_type": "none/continuation/reference/drill_down/modification"
    }}
    """)

        try:
            formatted_prompt = context_detection_prompt.format(
                previous_queries='; '.join(self.query_history[-2:]) if self.query_history else 'None',
                last_entity=self.conversation_context.get('last_entity', 'None'),
                last_table=self.conversation_context.get('last_table', 'None'),
                last_conditions=self.conversation_context.get('last_conditions', 'None'),
                current_query=user_question
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                context_analysis = json.loads(json_match.group())
                needs_context = context_analysis.get('needs_context', False)
                
                logger.info(f"Context analysis: {context_analysis.get('reasoning', 'No reasoning')}")
                
                if "conversation_context" not in state or not isinstance(state["conversation_context"], dict):
                    state["conversation_context"] = {}
                
                state["conversation_context"]["needs_context"] = needs_context
                state["conversation_context"]["context_type"] = context_analysis.get('context_type', 'none')
                state["conversation_context"]["reasoning"] = context_analysis.get('reasoning', '')
            else:
                logger.warning("Failed to parse JSON from context detection - defaulting to independent")
                state["conversation_context"]["needs_context"] = False

        except Exception as e:
            logger.error(f"Context detection failed: {e} - using pattern matching")
            state["conversation_context"]["needs_context"] = is_likely_contextual

        return state
    
    def _extract_intent(self, state: AgentState) -> AgentState:
        """
        Uses an LLM to distill the core user intent from a potentially conversational query.
        """
        user_question = state.get("corrected_question") or state["user_question"]
        logger.info(f"Extracting core intent from query: {user_question[:100]}...")

        # If the query is already short and direct, no need to process it.
        if len(user_question.split()) < 8:
            logger.info("Query is short and direct. Using as is.")
            state["user_intent"] = user_question
            return state

        intent_prompt = ChatPromptTemplate.from_template("""
You are an expert at understanding user requests. Your task is to extract the core, direct command from the user's potentially conversational question.

Convert the user's question into a concise, direct instruction for a data analyst.

**Examples:**
- User Question: "Would you like to visualize this with a bar chart showing the total sales volume for Sprite Zero across different order sources?"
- Your Output: "Show total sales volume for Sprite Zero by order source"

- User Question: "Can we create a line chart showing the monthly trend of orders for Fanta Orange year-to-date?"
- Your Output: "Show the monthly trend of orders for Fanta Orange year-to-date"
            
- User Question: "How many orders have been placed for Fanta Orange 1.49L?"
- Your Output: "How many orders have been placed for Fanta Orange 1.49L?"

**User Question:**
"{user_question}"

**Direct Command:**
""")
        
        try:
            formatted_prompt = intent_prompt.format(user_question=user_question)
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            intent = response.content.strip()
            
            # Clean up potential markdown or prefixes
            if intent.lower().startswith("direct command:"):
                intent = intent[15:].strip()
            
            logger.info(f"Successfully extracted intent: {intent}")
            state["user_intent"] = intent
        except Exception as e:
            logger.error(f"Failed to extract intent: {e}. Using original query.")
            state["user_intent"] = user_question
            
        return state
    
    def _resolve_clarification(self, state: AgentState) -> AgentState:
        """
        Orchestrates the resolution of a clarification by calling the processing helper
        and updating the agent state with the corrected query.
        """
        logger.info("Resolving user's clarification response using the corrected query.")
        user_response = state["user_question"]
        
        # Retrieve the pending clarification context from the session
        pending_context = self.pending_clarification or st.session_state.get('pending_clarification')
        
        if not pending_context:
            logger.error("Attempted to resolve clarification, but no pending context was found.")
            # Fallback in case context is lost
            state["final_response"] = "I'm sorry, I seem to have lost the context for that clarification. Could you please ask your original question again?"
            state["error_message"] = "No pending clarification context."
            return state

        # *** Use the EXISTING helper function to get the corrected query ***
        corrected_query = self._process_clarification_response(
            user_response,
            pending_context["original_query"],
            pending_context["entity_suggestions"]
        )
        
        logger.info(f"Clarification resolved. New query to process: '{corrected_query}'")
        
        # Update the state to reflect the resolution
        state["user_question"] = corrected_query
        state["corrected_question"] = corrected_query
        state["needs_clarification"] = False  # Mark clarification as handled
        
        # CRITICAL: Clear the pending clarification state to exit the loop
        self.pending_clarification = None
        if 'pending_clarification' in st.session_state:
            del st.session_state['pending_clarification']
            
        return state

    def _generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL with intelligent context awareness for Azure Synapse."""
        user_question = state["user_intent"]
        current_date = datetime.now().strftime('%Y-%m-%d')
        context = state.get("conversation_context", {})
        previous_queries = state.get("previous_queries", [])

        logger.info(f"Generating SQL for: {user_question[:50]}...")

        # **NEW: Check if this is a chart modification request**
        chart_keywords = ['change color', 'change colour', 'make it', 'modify chart', 
                        'update chart', 'different color', 'different colour']
        is_chart_mod = any(keyword in user_question.lower() for keyword in chart_keywords)
        
        # Check for recent chart
        has_recent_chart = False
        if st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history[-3:]):
                if msg.get('type') == 'bot' and msg.get('chart_code'):
                    has_recent_chart = True
                    break
        
        if is_chart_mod and has_recent_chart and previous_queries:
            # Reuse the last SQL query for chart modification
            state["sql_query"] = previous_queries[-1]
            logger.info("Chart modification detected - reusing last SQL query")
            return state

        if context.get("needs_context", False) and not previous_queries:
            state["sql_query"] = "NO_CONTEXT_ERROR"
            state["error_message"] = "No previous query context available. Please provide a specific query first."
            logger.warning("Context needed but no previous queries available. Setting NO_CONTEXT_ERROR.")
            return state
        
        context_info = ""
        if context.get("needs_context", False) and previous_queries:
            context_info = f"""
    IMPORTANT - This query requires context from previous conversation:
    - Last entity referenced: {context.get('last_entity', 'None')}
    - Last table used: {context.get('last_table', 'None')}
    - Last filter conditions: {context.get('last_conditions', 'None')}
    - Previous SQL queries: {'; '.join(previous_queries[-2:])}

    **CONTEXT USAGE RULES:**
    - If user says "also show first 5" or "first 5 too", use same CustomerName/ProductName from previous query
    - If user says "also", "too", or "as well", maintain same entity filters
    - For "first" vs "last", only change ORDER BY direction (ASC vs DESC)
    - Keep all WHERE conditions from previous query unless explicitly changed
    """
        else:
            context_info = "This is an independent query that should be answered without reference to previous context."

        sql_prompt = ChatPromptTemplate.from_template("""
    You are a Text-to-SQL code generator for Azure Synapse.

    **CRITICAL RULES:**
    1. **OUTPUT FORMAT:** Return ONLY the SQL query. No markdown, no explanations.
    2. **SYNTAX:** Use Azure Synapse (SQL Server) syntax. Use `TOP` instead of `LIMIT`. Use `GETDATE()` for current date.
    3. **SCHEMA:** Always use `dbo.` schema prefix (e.g., `dbo.sales_data`).
    4. **CHART REQUESTS:** If user asks for visualization (chart, graph, plot), generate SQL to fetch the necessary data. Do NOT return INVALID.
    5. **CONTEXTUAL QUERIES:** Pay attention to context. If user says "also show first 5" after a query about a customer, use that same customer.

    **Database Schema:**
    {schema}

    {examples}

    **Context for this Query:**
    {context_info}

    **Current date:** {current_date}

    Generate Azure Synapse SQL query for: "{user_question}"

    Remember: For chart/visualization requests, return SQL that fetches the data needed for the chart.
    """)

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

            # Clean up markdown
            sql_query = re.sub(r'^```sql\n?', '', sql_query)
            sql_query = re.sub(r'\n?```$', '', sql_query)
            sql_query = sql_query.strip()

            if sql_query.upper() == "INVALID":
                raise ValueError("LLM determined the request is not a valid SQL query.")

            if context.get("needs_context", False) or self._establishes_new_context(sql_query):
                self._update_context_from_sql(sql_query)

            state["sql_query"] = sql_query
            self.query_history.append(sql_query)
            logger.info(f"SQL generated successfully: {sql_query[:100]}...")

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            state["error_message"] = f"Failed to generate SQL query: {str(e)}"
            state["sql_query"] = "Error in SQL generation"

        return state
    
    def _validate_sql_structure(self, sql_query: str) -> Dict[str, Any]:
        """Validate SQL structure without executing - Azure Synapse specific."""
        logger.info("Validating SQL structure for Azure Synapse compliance")
        
        sql_upper = sql_query.upper().strip()
        
        # Basic structure checks
        checks = {
            "has_select": "SELECT" in sql_upper,
            "has_from": "FROM" in sql_upper,
            "uses_top_not_limit": "LIMIT" not in sql_upper,
            "has_schema_prefix": "DBO." in sql_upper,
            "no_dangerous_keywords": not any(kw in sql_upper for kw in ['DROP', 'DELETE', 'TRUNCATE', 'ALTER']),
        }
        
        # Detailed validation
        if not checks["has_select"]:
            return {
                "is_valid": False, 
                "reason": "Missing SELECT statement. Azure Synapse queries must start with SELECT.",
                "suggestion": "Add SELECT statement to query"
            }
        
        if not checks["has_from"]:
            return {
                "is_valid": False, 
                "reason": "Missing FROM clause. Query must specify which table to query from.",
                "suggestion": "Add FROM dbo.table_name clause"
            }
        
        if not checks["uses_top_not_limit"]:
            return {
                "is_valid": False, 
                "reason": "Azure Synapse uses TOP instead of LIMIT for row limiting.",
                "suggestion": "Replace LIMIT with TOP (e.g., SELECT TOP 10 instead of LIMIT 10)"
            }
        
        if not checks["has_schema_prefix"]:
            logger.warning("Query missing dbo. schema prefix - may cause ambiguity")
            return {
                "is_valid": True,  # Warning, not error
                "reason": "Consider adding dbo. schema prefix for clarity",
                "suggestion": "Use dbo.table_name instead of just table_name"
            }
        
        if not checks["no_dangerous_keywords"]:
            return {
                "is_valid": False, 
                "reason": "Query contains dangerous keywords (DROP/DELETE/TRUNCATE/ALTER)",
                "suggestion": "Only SELECT queries are allowed"
            }
        
        # Additional Azure Synapse specific checks
        if "CURDATE()" in sql_upper or "NOW()" in sql_upper:
            return {
                "is_valid": False, 
                "reason": "Use GETDATE() instead of CURDATE() or NOW() for Azure Synapse",
                "suggestion": "Replace with GETDATE()"
            }
        
        logger.info("SQL validation passed")
        return {"is_valid": True, "reason": "SQL structure is valid"}


    def _regenerate_sql_with_feedback(self, state: AgentState, validation_error: str) -> AgentState:
        """Regenerate SQL with validation feedback."""
        logger.info(f"Regenerating SQL with feedback: {validation_error}")
        
        retry_prompt = ChatPromptTemplate.from_template("""
    You are a Text-to-SQL expert for Azure Synapse. Your previous SQL query had issues.

    **ORIGINAL USER QUESTION:**
    "{user_question}"

    **PREVIOUS SQL ATTEMPT:**
    ```sql
    {previous_sql}
    VALIDATION ERROR:
    {validation_error}
    DATABASE SCHEMA:
    {schema}
    CRITICAL AZURE SYNAPSE RULES:

    Use SELECT TOP N instead of LIMIT N
    Use GETDATE() instead of CURDATE() or NOW()
    Always use dbo. schema prefix (e.g., dbo.sales_data)
    Use SQL Server syntax, not MySQL or PostgreSQL
    Every query MUST have SELECT and FROM clauses

    Generate a CORRECTED SQL query that fixes the validation error.
    Return ONLY the SQL query with no explanations.
    """)
        try:
            formatted_prompt = retry_prompt.format(
                user_question=state["user_intent"],
                previous_sql=state["sql_query"],
                validation_error=validation_error,
                schema=self.schema_info[:1000]  # Limit schema size
            )
            
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            corrected_sql = response.content.strip()
            
            # Clean up markdown
            corrected_sql = re.sub(r'^```sql\n?', '', corrected_sql)
            corrected_sql = re.sub(r'\n?```$', '', corrected_sql)
            corrected_sql = corrected_sql.strip()
            
            state["sql_query"] = corrected_sql
            logger.info(f"SQL regenerated successfully: {corrected_sql[:100]}...")
            
        except Exception as e:
            logger.error(f"SQL regeneration failed: {e}")
            state["sql_validation_error"] = str(e)

        return state

    def _establishes_new_context(self, sql_query: str) -> bool:
        """Check if the SQL query establishes new context for future queries."""
        context_indicators = [
            r"WHERE.*=.*'[^']+'",
            r"WHERE.*LIKE.*'%.*%'",
            r"ProductName\s*=",
            r"CustomerName\s*=",
            r"TOP\s+\d+",
        ]

        for pattern in context_indicators:
            if re.search(pattern, sql_query, re.IGNORECASE):
                return True

        return False

    def _update_context_from_sql(self, sql_query: str):
        """Extract and update conversation context from SQL query."""
        logger.debug("Updating conversation context from SQL query")

        table_pattern = r'FROM\s+(?:dbo\.)?(\w+)'
        tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
        if tables:
            self.conversation_context['last_table'] = tables[0]

        entity_pattern = r"'([^']+)'"
        entities = re.findall(entity_pattern, sql_query)
        if entities:
            self.conversation_context['last_entity'] = entities[0]

        where_pattern = r'WHERE\s+(.+?)(?:GROUP|ORDER|OFFSET|$)'
        conditions = re.findall(where_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        if conditions:
            self.conversation_context['last_conditions'] = conditions[0].strip()

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

    def _get_column_distinct_values(self, table_name: str, column_name: str, limit: int = 10) -> List[str]:
        """
        Fetch distinct values from a specific column to suggest alternatives.
        
        Args:
            table_name: Name of the table (e.g., 'sales_data')
            column_name: Name of the column (e.g., 'OrderSource')
            limit: Maximum number of values to return
        
        Returns:
            List of distinct values
        """
        try:
            query = f"""
            SELECT DISTINCT TOP {limit} [{column_name}]
            FROM dbo.{table_name}
            WHERE [{column_name}] IS NOT NULL
            ORDER BY [{column_name}]
            """
            
            logger.info(f"Fetching distinct values for {table_name}.{column_name}")
            result_df = self.db_manager.execute_query(query)
            
            if not result_df.empty:
                values = result_df.iloc[:, 0].tolist()
                logger.info(f"Found {len(values)} distinct values in {column_name}")
                return values
            else:
                logger.warning(f"No values found in {table_name}.{column_name}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to fetch distinct values for {column_name}: {e}")
            return []
        
    def _extract_and_suggest_alternatives(self, sql_query: str, user_question: str) -> str:
        """
        Analyze SQL query to extract filter conditions and suggest available values.
        ONLY suggest alternatives if the filtered value doesn't exist in the database.
        
        Args:
            sql_query: The executed SQL query
            user_question: Original user question
        
        Returns:
            Helpful message with suggestions, or empty string if no suggestions needed
        """
        try:
            # Extract table name
            table_match = re.search(r'FROM\s+(?:dbo\.)?(\w+)', sql_query, re.IGNORECASE)
            if not table_match:
                return ""
            
            table_name = table_match.group(1)
            
            # **NEW: Check if this is a time-based query with valid filters**
            time_keywords = ['year', 'month', 'quarter', 'day', 'date', 'ytd', 'mtd']
            has_time_filter = any(keyword in sql_query.lower() for keyword in time_keywords)
            
            if has_time_filter:
                # For time-based queries, don't suggest alternatives - it's likely just no data for that period
                logger.info("Empty result due to time filter - no suggestions needed")
                return ""
            
            # Extract WHERE conditions - look for column = 'value' or column LIKE '%value%'
            where_patterns = [
                r"WHERE\s+(\w+)\s*=\s*'([^']+)'",  # WHERE Column = 'Value'
                r"WHERE\s+(\w+)\s+LIKE\s*'%([^']+)%'",  # WHERE Column LIKE '%Value%'
            ]
            
            suggestions_text = ""
            
            for pattern in where_patterns:
                matches = re.finditer(pattern, sql_query, re.IGNORECASE)
                
                for match in matches:
                    column_name = match.group(1)
                    searched_value = match.group(2)
                    
                    logger.info(f"Checking if '{searched_value}' exists in {table_name}.{column_name}")
                    
                    # **NEW: First verify if the value actually exists**
                    value_exists_query = f"""
                        SELECT COUNT(*) as count
                        FROM dbo.{table_name}
                        WHERE [{column_name}] = '{searched_value}'
                    """
                    
                    try:
                        result_df = self.db_manager.execute_query(value_exists_query)
                        value_exists = result_df.iloc[0]['count'] > 0
                        
                        if value_exists:
                            # Value exists in database, empty result is due to other filters (like time)
                            logger.info(f"Value '{searched_value}' exists but no results due to other filters")
                            return ""
                        
                        # Value doesn't exist - suggest alternatives
                        logger.info(f"Value '{searched_value}' not found - suggesting alternatives")
                        
                        # Get available values from that column
                        available_values = self._get_column_distinct_values(table_name, column_name, limit=10)
                        
                        if available_values:
                            # Format the suggestion message
                            values_list = ", ".join([f"**'{val}'**" for val in available_values[:5]])
                            
                            suggestions_text = f""" **No records found** for **{column_name} = '{searched_value}'**

     **Available values in {column_name}:**
    {values_list}

     **Tip:** Try rephrasing your query using one of the available values above. For example:
    - "Show me orders from {available_values[0]}"
    """
                            
                            logger.info(f"Generated suggestions for {column_name}: {available_values}")
                            return suggestions_text
                        
                    except Exception as e:
                        logger.error(f"Failed to check value existence: {e}")
                        continue
            
            return ""
            
        except Exception as e:
            logger.error(f"Failed to extract suggestions from SQL: {e}")
            return ""

    def _execute_sql(self, state: AgentState) -> AgentState:
        """Execute SQL with intelligent retry logic for malformed queries."""
        MAX_RETRIES = 2
        
        if state["sql_query"] == "NO_CONTEXT_ERROR":
            logger.error("Skipping SQL execution due to NO_CONTEXT_ERROR.")
            return state
        
        if state["sql_query"] == "Error in SQL generation":
            state["error_message"] = "Cannot execute query due to SQL generation error"
            logger.error("Cannot execute query due to SQL generation error")
            return state
        
        # Initialize retry counter if not present
        if "sql_retry_count" not in state:
            state["sql_retry_count"] = 0
        
        for attempt in range(MAX_RETRIES):
            try:
                logger.info(f"SQL Execution Attempt {attempt + 1}/{MAX_RETRIES}")
                
                # STEP 1: Validate SQL structure BEFORE executing
                validation_result = self._validate_sql_structure(state["sql_query"])
                
                if not validation_result["is_valid"]:
                    logger.warning(f"Attempt {attempt + 1}: SQL validation failed - {validation_result['reason']}")
                    
                    # If we have retries left, regenerate SQL
                    if attempt < MAX_RETRIES - 1:
                        state["sql_retry_count"] = attempt + 1
                        state = self._regenerate_sql_with_feedback(
                            state, 
                            validation_result['reason']
                        )
                        continue  # Try again with corrected SQL
                    else:
                        # Last attempt failed validation
                        state["error_message"] = f"SQL validation failed after {MAX_RETRIES} attempts: {validation_result['reason']}"
                        state["query_result"] = pd.DataFrame()
                        return state
                
                # STEP 2: SQL is valid - execute it
                logger.info(f"Executing validated SQL query: {state['sql_query'][:100]}...")
                result_df = self.db_manager.execute_query(state["sql_query"])
                
                # STEP 3: Check results
                if not result_df.empty:
                    # Success - we have data
                    state["query_result"] = result_df
                    state["error_message"] = ""
                    self.conversation_context['last_result_count'] = len(result_df)
                    logger.info(f"✅ SQL execution successful: {len(result_df)} rows returned")
                    return state
                
                else:
                    # Empty result - check if this is legitimate or a SQL issue
                    logger.info(f"Attempt {attempt + 1}: Empty result returned")
                    
                    if attempt < MAX_RETRIES - 1:
                        # Check if empty result might be due to wrong table/column names
                        # by analyzing the SQL query
                        sql_upper = state["sql_query"].upper()
                        
                        # If query has specific WHERE conditions, empty might be legitimate
                        has_specific_filters = any(keyword in sql_upper for keyword in [
                            "WHERE", "LIKE", "=", "BETWEEN", "IN ("
                        ])
                        
                        if has_specific_filters:
                            # Likely a legitimate empty result - accept it
                            logger.info("Empty result appears legitimate (query has specific filters)")
                            state["query_result"] = result_df
                            state["error_message"] = ""
                            return state
                        else:
                            # No filters and empty result - might be wrong table/column
                            logger.warning("Empty result with no filters - possible SQL issue")
                            state = self._regenerate_sql_with_feedback(
                                state,
                                "Query returned no data. Please verify table and column names are correct."
                            )
                            continue
                    else:
                        # Last attempt - accept empty result
                        state["query_result"] = result_df
                        state["error_message"] = ""
                        return state
            
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Attempt {attempt + 1}: SQL execution error - {error_msg[:200]}")
                
                # Check if it's a retryable error
                retryable_keywords = [
                    'syntax', 'invalid', 'unexpected', 'column', 'table', 
                    'does not exist', 'not found', 'ambiguous'
                ]
                
                is_retryable = any(keyword in error_msg for keyword in retryable_keywords)
                
                if is_retryable and attempt < MAX_RETRIES - 1:
                    logger.info(f"Retryable error detected - regenerating SQL")
                    state["sql_retry_count"] = attempt + 1
                    state = self._regenerate_sql_with_feedback(state, str(e))
                    continue
                else:
                    # Non-retryable error or last attempt
                    state["error_message"] = str(e)
                    state["query_result"] = pd.DataFrame()
                    return state
        
        # If we exhausted all retries
        logger.error(f"SQL execution failed after {MAX_RETRIES} attempts")
        state["error_message"] = f"Failed to execute query after {MAX_RETRIES} attempts"
        state["query_result"] = pd.DataFrame()
        return state

    def _check_sql_execution(self, state: AgentState) -> Literal["success", "error"]:
        """Check if SQL execution was successful."""
        return "success" if not state.get("error_message") else "error"
    
    async def _generate_response_async(self, state: AgentState) -> str:
        """Generate response asynchronously."""
        logger.info("Starting async response generation")
        
        loop = asyncio.get_event_loop()
        
        # Run the sync method in thread pool
        response = await loop.run_in_executor(
            None,
            self._generate_response_sync,
            state
        )
        
        return response


    def _generate_response_sync(self, state: AgentState) -> str:
        """Synchronous response generation with smart empty result handling."""
        query_result = state["query_result"]
        user_question = state["user_question"]
        sql_query = state.get("sql_query", "")

        logger.info("Generating response with LLM insights")

        if query_result.empty:
            # **IMPROVED: Check if this is a filter-based query that returned no results**
            is_filter_query = any(keyword in sql_query.upper() for keyword in ['WHERE', 'LIKE', '='])
            
            if is_filter_query:
                # Try to extract suggestions ONLY if the filter value doesn't exist
                suggestions = self._extract_and_suggest_alternatives(sql_query, user_question)
                
                if suggestions:
                    # Value doesn't exist - show suggestions
                    base_response = suggestions
                else:
                    # Value exists but no results due to other filters (like time)
                    # Use standard empty message without suggestions
                    date_keywords = ["today", "yesterday", "last", "this", "month", "year", "week", "day", "date", "ytd", "quarter", "mtd"]
                    is_date_query = any(keyword in user_question.lower() for keyword in date_keywords)
                    base_response = self.response_manager.get_empty_message(is_date_query)
            else:
                # Standard empty result (no filters applied)
                date_keywords = ["today", "yesterday", "last", "this", "month", "year", "week", "day", "date", "ytd", "quarter", "mtd"]
                is_date_query = any(keyword in user_question.lower() for keyword in date_keywords)
                base_response = self.response_manager.get_empty_message(is_date_query)
            
            return self._rephrase_message_with_llm(
                base_response,
                user_query=user_question,
                message_type="SUCCESS"
            )
        else:
            # SUCCESS CASE: Has results
            try:
                response_parts = []
                success_msg = self.response_manager.get_success_message(len(query_result))
                response_parts.append(success_msg)
                
                # Generate LLM-based insights
                insights = self._generate_automated_insights_llm(query_result, user_question, sql_query)
                if insights:
                    response_parts.append("\n\n**Business Intelligence Summary:**")
                    response_parts.extend([f"\n{insight}" for insight in insights])
                
                if len(query_result) > 20:
                    response_parts.append(f"\n\n*Displaying top 20 records from {len(query_result):,} total results.*")
                
                base_response = "".join(response_parts)
                base_response = self.response_manager.add_personality_touch(
                    base_response, 
                    {"result_count": len(query_result), "query": user_question}
                )
                
                return self._rephrase_message_with_llm(base_response)

            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                return self.response_manager.get_success_message(len(query_result))


    async def _generate_chart_async(self, state: AgentState) -> str:
        """Generate chart code asynchronously."""
        logger.info("Starting async chart generation")
        
        # Check if chart should be generated
        user_question = state["user_question"].lower()
        chart_keywords = ["chart", "plot", "graph", "visualize", "diagram", "bar", "pie", "line"]
        
        if not any(keyword in user_question for keyword in chart_keywords):
            logger.info("No chart request detected - skipping chart generation")
            return ""
        
        if state["query_result"].empty:
            logger.info("Empty result - skipping chart generation")
            return ""
        
        loop = asyncio.get_event_loop()
        
        # Create a temporary state for chart generation
        chart_state = state.copy()
        
        # Run chart generation in thread pool
        updated_state = await loop.run_in_executor(
            None,
            self._generate_chart_sync,
            chart_state
        )
        
        return updated_state.get("chart_code", "")
    
    def _generate_chart_sync(self, state: AgentState) -> AgentState:
        """Synchronous chart generation (existing logic)."""
        # This is your existing _generate_chart method logic
        # Copy the ENTIRE contents of your current _generate_chart method here
        user_question = state["user_question"]
        df = state["query_result"]
        
        user_question = state["user_question"]
        df = state["query_result"]
        columns = df.columns.tolist()

            # **Check if this is a chart modification request**
        modification_keywords = ['change color', 'change colour', 'make', 'modify', 'update', 
                                    'different color', 'yellow', 'red', 'green', 'blue']
        is_modification = any(keyword in user_question.lower() for keyword in modification_keywords)
            
            # Get the last chart code if this is a modification
        last_chart_code = ""
        if is_modification and st.session_state.chat_history:
                for msg in reversed(st.session_state.chat_history):
                    if msg.get('type') == 'bot' and msg.get('chart_code'):
                        last_chart_code = msg['chart_code']
                        break
            
        if is_modification and last_chart_code:
                logger.info(f"Modifying existing chart based on: {user_question[:50]}...")
                
                modification_prompt = ChatPromptTemplate.from_template("""
        You are modifying an existing chart based on a user request.

        **EXISTING CHART CODE:**
        ```python
        {existing_code}
        USER'S MODIFICATION REQUEST: "{user_request}"
        TASK: Modify ONLY the styling elements requested by the user. Keep all data processing and structure intact.
        Common modifications:

        Colors: Update colors= parameter in plotting functions
        Size: Change figsize= in plt.subplots() or plt.figure()
        Labels: Modify xlabel(), ylabel(), title() calls
        Legend: Adjust legend() parameters

        Return ONLY the complete modified Python code. No explanations.
        """)
                try:
                    formatted_prompt = modification_prompt.format(
                        existing_code=last_chart_code,
                        user_request=user_question
                    )

                    response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
                    modified_code = self._extract_code_from_response(response.content)

                    state["chart_code"] = modified_code
                    state["final_response"] = f"I've updated the chart colors as requested! Here's your modified visualization."
                    
                    logger.info(f"Chart modified successfully")
                    return state

                except Exception as e:
                    logger.error(f"Chart modification failed: {e}")
                    # Fall through to normal chart generation

            
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
    ```---Line Chart Code Example (including Time Series)---
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#F0F0F6')

    # For time series data, convert dates first
    if 'OrderDate' in df.columns or any('date' in col.lower() for col in df.columns):
        date_col = [col for col in df.columns if 'date' in col.lower()]
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
    
    async def _generate_followup_async(self, state: AgentState) -> List[str]:
        """Generate follow-up questions asynchronously with RICH CONTEXT."""
        logger.info("Starting async follow-up generation with enhanced context")
        
        loop = asyncio.get_event_loop()
        
        # Build rich context for follow-ups
        followup_context = self._build_rich_followup_context(state)
        
        # Run in thread pool
        questions = await loop.run_in_executor(
            None,
            self._generate_followup_sync,
            state,
            followup_context
        )
        
        return questions


    def _build_rich_followup_context(self, state: AgentState) -> Dict[str, Any]:
        """Build comprehensive context for intelligent follow-ups."""
        query_result = state["query_result"]
        
        context = {
            "user_question": state["user_question"],
            "sql_query": state["sql_query"],
            "conversation_history": state.get("previous_queries", [])[-3:],
            "result_metadata": {
                "row_count": len(query_result) if not query_result.empty else 0,
                "columns": query_result.columns.tolist() if not query_result.empty else [],
                "has_dates": any('date' in col.lower() for col in query_result.columns) if not query_result.empty else False,
                "has_numeric": len(query_result.select_dtypes(include=[np.number]).columns) > 0 if not query_result.empty else False
            },
            "extracted_entities": {
                "products": self._extract_entities_from_sql(state["sql_query"], "product"),
                "customers": self._extract_entities_from_sql(state["sql_query"], "customer"),
                "tables_used": self._extract_tables_from_sql(state["sql_query"])
            },
            "aggregation_level": self._detect_aggregation_level(state["sql_query"]),
            "time_dimension": self._extract_time_dimension(state["sql_query"]),
            "chart_suggestion": state.get("visualization_suggestion", "")
        }
        
        return context


    def _extract_entities_from_sql(self, sql_query: str, entity_type: str) -> List[str]:
        """Extract specific entities from SQL WHERE clauses."""
        entities = []
        
        # Extract from WHERE clause
        where_pattern = r"WHERE\s+(.+?)(?:GROUP|ORDER|$)"
        where_match = re.search(where_pattern, sql_query, re.IGNORECASE | re.DOTALL)
        
        if where_match:
            where_clause = where_match.group(1)
            
            if entity_type == "product":
                # Look for ProductDesc or ProductName
                product_pattern = r"ProductDesc\s*(?:LIKE|=)\s*'([^']+)'"
                entities = re.findall(product_pattern, where_clause, re.IGNORECASE)
            
            elif entity_type == "customer":
                # Look for CustomerName
                customer_pattern = r"CustomerName\s*(?:LIKE|=)\s*'([^']+)'"
                entities = re.findall(customer_pattern, where_clause, re.IGNORECASE)
        
        return entities
    
    def _generate_followup_sync(self, state: AgentState, context: Dict[str, Any]) -> List[str]:
        """Synchronous follow-up generation with rich context."""
        logger.info("Generating context-aware follow-ups")
        
        # Check if user already asked for visualization
        chart_keywords = ["chart", "plot", "graph", "visualize"]
        user_asked_for_chart = any(kw in state["user_question"].lower() for kw in chart_keywords)
        
        followup_prompt = ChatPromptTemplate.from_template("""
    You are generating intelligent follow-up questions based on a data query.

    **RICH CONTEXT:**
    ```json
    {context_json}
    RULES FOR CONTEXTUAL FOLLOW-UPS:

    Reference Specific Entities: Mention products, customers, or time periods from the current query
    Suggest Drill-Downs: If showing aggregated data, suggest viewing details
    Offer Comparisons: Compare with other time periods, categories, or segments
    Time-Based Evolution: If query has dates, suggest trends or time comparisons
    Visualization: {"Skip visualization question (user already asked for it)" if user_asked_for_chart else "First question should suggest a relevant chart"}

    CONTEXT-AWARE PATTERNS:

    If query filtered by product X: "How does X compare to [similar product]?"
    If query shows monthly data: "Would you like to see the yearly trend?"
    If query aggregates by category: "Which specific items in [category] are top performers?"
    If query shows totals: "Can we break this down by [region/customer/product]?"

    Generate exactly 3 follow-up questions that feel like natural conversation.
    {"The first question should suggest a visualization." if not user_asked_for_chart else "All 3 should be data analysis questions."}
    Return questions one per line, no numbering.
    """)
        try:
            formatted_prompt = followup_prompt.format(
                context_json=json.dumps(context, indent=2),
                user_asked_for_chart=user_asked_for_chart
            )
            
            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            
            # Parse questions
            questions = [q.strip() for q in response.content.strip().split('\n') if q.strip()]
            questions = [q for q in questions if q and not q.startswith(('1.', '2.', '3.', '-', '•'))]
            
            # Clean up and ensure 3 questions
            clean_questions = []
            for q in questions[:3]:
                q = q.strip()
                if q and not q.endswith('?'):
                    q += '?'
                clean_questions.append(q)
            
            # Ensure exactly 3
            while len(clean_questions) < 3:
                clean_questions.append("What other insights would you like to explore?")
            
            logger.info(f"Generated {len(clean_questions)} context-aware follow-ups")
            return clean_questions[:3]
            
        except Exception as e:
            logger.error(f"Follow-up generation failed: {e}")
            return [
                "Which day of the week sees the highest order volume?",
                "What's the breakdown of orders by sales channel?",
                "Which Customer ordered Coke 155ml 3x10 Can in the past 6 months?"
            ]
        
    async def _run_parallel_tasks(self, state: AgentState) -> AgentState:
        """Run response, chart, and follow-up generation in parallel."""
        logger.info("Starting parallel task execution")
        
        # Create tasks
        tasks = [
            self._generate_response_async(state),
            self._generate_chart_async(state),
            self._generate_followup_async(state)
        ]
        
        # Run in parallel and collect results
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        state["final_response"] = results[0] if not isinstance(results[0], Exception) else "Error generating response"
        state["chart_code"] = results[1] if not isinstance(results[1], Exception) else ""
        state["follow_up_questions"] = results[2] if not isinstance(results[2], Exception) else []
        
        # Mark tasks as ready
        state["response_ready"] = not isinstance(results[0], Exception)
        state["chart_ready"] = not isinstance(results[1], Exception)
        state["followup_ready"] = not isinstance(results[2], Exception)
        
        logger.info(f"Parallel tasks completed - Response: {state['response_ready']}, Chart: {state['chart_ready']}, Followups: {state['followup_ready']}")
        
        return state


    def _parallel_processor(self, state: AgentState) -> AgentState:
        """Wrapper to run async tasks from sync context."""
        return asyncio.run(self._run_parallel_tasks(state))



    def _extract_tables_from_sql(self, sql_query: str) -> List[str]:
        """Extract table names from SQL query."""
        # Pattern: FROM dbo.table_name or JOIN dbo.table_name
        table_pattern = r'(?:FROM|JOIN)\s+(?:dbo\.)?(\w+)'
        return list(set(re.findall(table_pattern, sql_query, re.IGNORECASE)))


    def _detect_aggregation_level(self, sql_query: str) -> str:
        """Detect if query uses aggregation."""
        sql_upper = sql_query.upper()
        
        if any(agg in sql_upper for agg in ['SUM(', 'COUNT(', 'AVG(', 'MAX(', 'MIN(']):
            if 'GROUP BY' in sql_upper:
                return "grouped_aggregation"
            else:
                return "total_aggregation"
        else:
            return "detail_level"


    def _extract_time_dimension(self, sql_query: str) -> Dict[str, Any]:
        """Extract time-related information from SQL."""
        sql_upper = sql_query.upper()
        
        time_info = {
            "has_time_filter": False,
            "time_granularity": None,
            "relative_time": None
        }
        
        # Check for time filters
        if any(keyword in sql_upper for keyword in ['YEAR', 'MONTH', 'DAY', 'DATE', 'GETDATE']):
            time_info["has_time_filter"] = True
            
            if 'YEAR' in sql_upper and 'MONTH' in sql_upper:
                time_info["time_granularity"] = "month"
            elif 'YEAR' in sql_upper:
                time_info["time_granularity"] = "year"
            elif 'DAY' in sql_upper or 'GETDATE' in sql_upper:
                time_info["time_granularity"] = "day"
        
        return time_info

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate intelligent response with LLM-based insights."""
        query_result = state["query_result"]
        user_question = state["user_question"]
        sql_query = state.get("sql_query", "")

        logger.info("Generating response with LLM insights")

        if query_result.empty:
            date_keywords = ["today", "yesterday", "last", "this", "month", "year", "week", "day", "date", "ytd", "quarter", "mtd"]
            is_date_query = any(keyword in user_question.lower() for keyword in date_keywords)
            
            base_response = self.response_manager.get_empty_message(is_date_query)
            state["final_response"] = self._rephrase_message_with_llm(
    base_response,
    user_query=user_question,
    message_type="SUCCESS"
)
        else:
            try:
                response_parts = []
                success_msg = self.response_manager.get_success_message(len(query_result))
                response_parts.append(success_msg)
                
                # Generate LLM-based insights
                insights = self._generate_automated_insights_llm(query_result, user_question, sql_query)
                if insights:
                    response_parts.append("\n\n**Business Intelligence Summary:**")
                    response_parts.extend([f"\n{insight}" for insight in insights])
                
                if len(query_result) > 20:
                    response_parts.append(f"\n\n*Displaying top 20 records from {len(query_result):,} total results.*")
                
                base_response = "".join(response_parts)
                base_response = self.response_manager.add_personality_touch(
                    base_response, 
                    {"result_count": len(query_result), "query": user_question}
                )
                
                state["final_response"] = self._rephrase_message_with_llm(base_response)
                logger.info(f"Generated response with LLM insights for {len(query_result)} records")

            except Exception as e:
                logger.error(f"Response generation failed: {e}")
                state["final_response"] = self.response_manager.get_success_message(len(query_result))

        return state
    
    def _detect_chart_modification_request(self, state: AgentState) -> Literal["modify_chart", "generate_sql"]:
        """Detect if user wants to modify an existing chart."""
        user_question = state["user_question"].lower()
        
        # Check if there's a recent chart in history
        has_recent_chart = False
        if st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history[-3:]):
                if msg.get('type') == 'bot' and msg.get('chart_code'):
                    has_recent_chart = True
                    break
        
        if not has_recent_chart:
            return "generate_sql"
        
        # Chart modification keywords
        modification_keywords = [
            "change color", "change colour", "make it", "modify", "update", "adjust",
            "different color", "blue", "red", "green", "larger", "smaller",
            "title", "label", "legend", "size", "width", "height"
        ]
        
        if any(keyword in user_question for keyword in modification_keywords):
            logger.info("Detected chart modification request")
            return "modify_chart"
        
        return "generate_sql"

    def _modify_existing_chart(self, state: AgentState) -> AgentState:
        """Modify existing chart based on user request using LLM."""
        user_request = state["user_question"]
        
        # Get the last chart code from history
        last_chart_code = ""
        last_dataframe = pd.DataFrame()
        last_query = ""
        
        for msg in reversed(st.session_state.chat_history):
            if msg.get('type') == 'bot' and msg.get('chart_code'):
                last_chart_code = msg['chart_code']
                last_dataframe = msg.get('dataframe', pd.DataFrame())
                last_query = msg.get('sql_query', '')
                break
        
        if not last_chart_code:
            state["error_message"] = "No recent chart found to modify"
            return state
        
        modification_prompt = ChatPromptTemplate.from_template("""
    You are a Python visualization expert. Modify the existing chart code based on the user's request.

    EXISTING CHART CODE:
    ```python
    {existing_code}
    USER'S MODIFICATION REQUEST: "{user_request}"
    RULES:

    Keep the same data and chart structure
    Only modify styling elements requested by user (colors, sizes, labels, etc.)
    Maintain all variable names and data processing logic
    Ensure fig variable is still created for display
    Return ONLY the complete modified Python code in a code block

    COMMON MODIFICATIONS:

    Colors: Update color parameters in plotting functions
    Size: Change figsize in plt.subplots() or figure()
    Labels: Modify xlabel, ylabel, title
    Legend: Adjust legend() parameters
    Grid: Add/modify grid settings

    Return only the Python code block, no explanations.
    """)
        try:
            formatted_prompt = modification_prompt.format(
                existing_code=last_chart_code,
                user_request=user_request
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            modified_code = self._extract_code_from_response(response.content)

            state["chart_code"] = modified_code
            state["query_result"] = last_dataframe
            state["sql_query"] = last_query
            state["final_response"] = f"I've modified the chart based on your request: '{user_request}'. Here's the updated visualization!"
            
            logger.info(f"Chart modified successfully based on: {user_request}")

        except Exception as e:
            logger.error(f"Chart modification failed: {e}")
            state["error_message"] = f"Failed to modify chart: {str(e)}"
            state["chart_code"] = last_chart_code  # Fallback to original

        return state
    
    def _generate_automated_insights_llm(self, df: pd.DataFrame, user_question: str, sql_query: str) -> List[str]:
        """Generate professional business insights using LLM."""
        
        if df.empty:
            return []
        
        # Prepare data summary
        summary_stats = {}
        for col in df.select_dtypes(include=[np.number]).columns[:5]:
            if df[col].notna().sum() > 0:
                summary_stats[col] = {
                    'mean': float(df[col].mean()),
                    'sum': float(df[col].sum()),
                    'max': float(df[col].max()),
                    'min': float(df[col].min()),
                    'count': int(df[col].count())
                }
        
        # **FIX: Handle categorical summary with date serialization**
        categorical_summary = {}
        for col in df.select_dtypes(include=['object', 'string']).columns[:3]:
            if df[col].notna().sum() > 0:
                value_counts = df[col].value_counts().head(5)
                # Convert to regular Python types to avoid serialization issues
                categorical_summary[col] = {
                    'unique_count': int(df[col].nunique()),
                    'top_values': {str(k): int(v) for k, v in value_counts.items()}
                }
        
        # **FIX: Handle date columns separately**
        date_summary = {}
        date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns
        if len(date_cols) > 0:
            for col in date_cols[:2]:
                if df[col].notna().sum() > 0:
                    date_summary[col] = {
                        'min_date': df[col].min().strftime('%Y-%m-%d'),
                        'max_date': df[col].max().strftime('%Y-%m-%d'),
                        'range_days': (df[col].max() - df[col].min()).days
                    }
        
        insight_prompt = ChatPromptTemplate.from_template("""
    You are a senior business analyst providing insights from data analysis.

    USER'S QUESTION: "{user_question}"
    SQL QUERY EXECUTED: "{sql_query}"

    DATA ANALYSIS:
    - Total Records: {row_count}
    - Columns: {columns}
    - Numeric Statistics: {summary_stats}
    - Categorical Summary: {categorical_summary}
    - Date Range: {date_summary}

    BUSINESS CONTEXT:
    {mapping_context}

    Generate 3-4 professional, actionable business insights that:
    1. Directly answer the user's question
    2. Highlight key patterns, trends, or outliers
    3. Provide business context with specific numbers
    4. Are concise (1-2 sentences each)

    Format each insight as: • [Insight text]

    EXAMPLES:
    - Revenue Analysis: Total sales of $1.2M with top product accounting for 35% ($420K)
    - Customer Concentration: Top 3 customers represent 45% of orders
    - Time Range: Data spans 180 days from Jan 2024 to June 2024
    """)

        try:
            formatted_prompt = insight_prompt.format(
                user_question=user_question,
                sql_query=sql_query[:300],
                row_count=len(df),
                columns=', '.join(df.columns.tolist()),
                summary_stats=json.dumps(summary_stats, indent=2) if summary_stats else 'No numeric data',
                categorical_summary=json.dumps(categorical_summary, indent=2) if categorical_summary else 'No categorical data',
                date_summary=json.dumps(date_summary, indent=2) if date_summary else 'No date information',
                mapping_context=MAPPING_SCHEMA_PROMPT[:400]
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            insights_text = response.content.strip()
            
            insights = [
                line.strip() 
                for line in insights_text.split('\n') 
                if line.strip().startswith('•')
            ]
            
            logger.info(f"Generated {len(insights)} LLM insights")
            return insights[:4]

        except Exception as e:
            logger.error(f"LLM insight generation failed: {e}")
            return [f"• Dataset analyzed: {len(df):,} records across {len(df.columns)} business metrics"]
        

    # def _generate_automated_insights(self, df: pd.DataFrame, user_question: str) -> List[str]:
    #     """Generate professional business insights from dataframe."""
    #     insights = []
        
    #     try:
    #         # Business-focused numeric column insights
    #         numeric_cols = df.select_dtypes(include=[np.number]).columns
    #         for col in numeric_cols[:3]:
    #             if df[col].notna().sum() > 0:
    #                 total = df[col].sum()
    #                 avg = df[col].mean()
    #                 max_val = df[col].max()
    #                 min_val = df[col].min()
                    
    #                 # Business context for different metric types
    #                 if any(keyword in col.lower() for keyword in ['sales', 'revenue', 'amount']):
    #                     insights.append(f"• **Revenue Analysis**: Total {col.replace('_', ' ').title()}: ${total:,.0f} with an average transaction value of ${avg:,.0f}")
    #                     if max_val > avg * 2:
    #                         insights.append(f"• **Performance Variance**: Highest {col.lower()} of ${max_val:,.0f} indicates significant growth opportunities in top-performing segments")
                    
    #                 elif any(keyword in col.lower() for keyword in ['volume', 'quantity', 'count', 'orders']):
    #                     insights.append(f"• **Volume Metrics**: Total {col.replace('_', ' ').title()}: {total:,.0f} units across {len(df)} records (Avg: {avg:,.0f} per record)")
                        
    #                 elif any(keyword in col.lower() for keyword in ['margin', 'profit', 'roi']):
    #                     insights.append(f"• **Profitability**: {col.replace('_', ' ').title()} ranges from {min_val:.1f}% to {max_val:.1f}% with average of {avg:.1f}%")
                    
    #                 else:
    #                     insights.append(f"• **Key Metric**: {col.replace('_', ' ').title()} - Range: {min_val:,.0f} to {max_val:,.0f} (Avg: {avg:,.0f})")

    #         # Business-focused categorical analysis
    #         categorical_cols = df.select_dtypes(include=['object', 'string']).columns
    #         for col in categorical_cols[:2]:
    #             if df[col].notna().sum() > 0:
    #                 unique_count = df[col].nunique()
                    
    #                 if any(keyword in col.lower() for keyword in ['customer', 'client', 'account']):
    #                     if unique_count <= 10:
    #                         top_customers = df[col].value_counts().head(3)
    #                         customer_list = ", ".join([f"{customer} ({count} transactions)" for customer, count in top_customers.items()])
    #                         insights.append(f"• **Customer Distribution**: Top customers by transaction volume - {customer_list}")
    #                     else:
    #                         insights.append(f"• **Customer Portfolio**: {unique_count} unique customers represented in this dataset")
                    
    #                 elif any(keyword in col.lower() for keyword in ['product', 'item', 'sku']):
    #                     if unique_count <= 10:
    #                         top_products = df[col].value_counts().head(3)
    #                         product_list = ", ".join([f"{product} ({count})" for product, count in top_products.items()])
    #                         insights.append(f"• **Product Performance**: Leading products - {product_list}")
    #                     else:
    #                         insights.append(f"• **Product Catalog**: {unique_count} different products analyzed")
                    
    #                 elif any(keyword in col.lower() for keyword in ['region', 'territory', 'location', 'country']):
    #                     if unique_count <= 10:
    #                         top_regions = df[col].value_counts().head(3)
    #                         region_list = ", ".join([f"{region} ({count} records)" for region, count in top_regions.items()])
    #                         insights.append(f"• **Geographic Distribution**: Primary markets - {region_list}")
    #                     else:
    #                         insights.append(f"• **Market Coverage**: Operations across {unique_count} geographic locations")
                    
    #                 else:
    #                     if unique_count <= 10:
    #                         value_counts = df[col].value_counts().head(3)
    #                         top_items = ", ".join([f"{val} ({count})" for val, count in value_counts.items()])
    #                         insights.append(f"• **{col.replace('_', ' ').title()} Analysis**: {top_items}")

    #         # Time-based business insights
    #         date_cols = [col for col in df.columns if any(date_word in col.lower() for date_word in ['date', 'time', 'created', 'updated'])]
    #         if date_cols and len(df) > 1:
    #             date_col = date_cols[0]
    #             try:
    #                 df_temp = df.copy()
    #                 df_temp[date_col] = pd.to_datetime(df_temp[date_col])
    #                 date_range = df_temp[date_col].max() - df_temp[date_col].min()
    #                 insights.append(f"• **Time Period**: Data spans {date_range.days} days from {df_temp[date_col].min().strftime('%B %Y')} to {df_temp[date_col].max().strftime('%B %Y')}")
                    
    #                 # Monthly trend if applicable
    #                 if len(df) > 10 and date_range.days > 30:
    #                     monthly_data = df_temp.groupby(pd.Grouper(key=date_col, freq='M')).size()
    #                     if len(monthly_data) > 1:
    #                         trend = "increasing" if monthly_data.iloc[-1] > monthly_data.iloc[0] else "decreasing"
    #                         insights.append(f"• **Trend Analysis**: Data shows {trend} activity pattern over the analyzed period")
    #             except:
    #                 pass

    #         # Business performance indicators
    #         if len(df) > 100:
    #             insights.append(f"• **Data Volume**: Robust dataset with {len(df):,} records providing statistically significant insights for decision-making")
    #         elif len(df) > 20:
    #             insights.append(f"• **Sample Size**: {len(df)} records analyzed - suitable for tactical business insights")

    #     except Exception as e:
    #         logger.error(f"Error generating business insights: {e}")
    #         # Fallback to basic business insight
    #         insights.append(f"• **Dataset Overview**: {len(df):,} business records analyzed across {len(df.columns)} key metrics")

    #     return insights[:4]

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

        # **Check if this is a chart modification request**
        modification_keywords = ['change color', 'change colour', 'make', 'modify', 'update', 
                                'different color', 'yellow', 'red', 'green', 'blue']
        is_modification = any(keyword in user_question.lower() for keyword in modification_keywords)
        
        # Get the last chart code if this is a modification
        last_chart_code = ""
        if is_modification and st.session_state.chat_history:
            for msg in reversed(st.session_state.chat_history):
                if msg.get('type') == 'bot' and msg.get('chart_code'):
                    last_chart_code = msg['chart_code']
                    break
        
        if is_modification and last_chart_code:
            logger.info(f"Modifying existing chart based on: {user_question[:50]}...")
            
            modification_prompt = ChatPromptTemplate.from_template("""
    You are modifying an existing chart based on a user request.

    **EXISTING CHART CODE:**
    ```python
    {existing_code}
    USER'S MODIFICATION REQUEST: "{user_request}"
    TASK: Modify ONLY the styling elements requested by the user. Keep all data processing and structure intact.
    Common modifications:

    Colors: Update colors= parameter in plotting functions
    Size: Change figsize= in plt.subplots() or plt.figure()
    Labels: Modify xlabel(), ylabel(), title() calls
    Legend: Adjust legend() parameters

    Return ONLY the complete modified Python code. No explanations.
    """)
        try:
            formatted_prompt = modification_prompt.format(
                existing_code=last_chart_code,
                user_request=user_question
            )

            response = self.llm.invoke([HumanMessage(content=formatted_prompt)])
            modified_code = self._extract_code_from_response(response.content)

            state["chart_code"] = modified_code
            state["final_response"] = f"I've updated the chart colors as requested! Here's your modified visualization."
            
            logger.info(f"Chart modified successfully")
            return state

        except Exception as e:
            logger.error(f"Chart modification failed: {e}")
            # Fall through to normal chart generation

        
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
```---Line Chart Code Example (including Time Series)---
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(12, 8), facecolor='#F0F0F6')

# For time series data, convert dates first
if 'OrderDate' in df.columns or any('date' in col.lower() for col in df.columns):
    date_col = [col for col in df.columns if 'date' in col.lower()]
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
        if "no previous query context available" in error_message:
            base_response = "It looks like you're asking a follow-up, but I don't have the context of a previous query. Could you please state your full question?"
            state["final_response"] = self._rephrase_message_with_llm(
    base_response,
    user_query=state.get("user_question", ""),
    message_type="ERROR"
)
            state["follow_up_questions"] = []
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
        
        base_response = self.response_manager.get_error_message(error_type)
        state["final_response"] = self._rephrase_message_with_llm(base_response)
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
            
            # Intent fields
            primary_intent="",
            contains_greeting=False,
            contains_data_request=False,
            intent_reasoning="",
            
            # Legacy fields
            is_valid_query=True,
            is_greeting=False,
            needs_clarification=False,
            clarification_prompt="",
            best_chart_type="",
            user_intent="",
            visualization_suggestion="",
            sql_retry_count=0,
            sql_validation_error="",
            response_ready=False,
            chart_ready=False,
            followup_ready=False,
            parallel_tasks_completed=0
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
            base_response = "I apologize, but I encountered an unexpected error. Please try again or rephrase your question."
            final_response = self._rephrase_message_with_llm(base_response)
            return {
                'success': False,
                'user_input': user_question,
                'sql_query': "Error occurred",
                'dataframe': pd.DataFrame(),
                'chart_code': "",
                'follow_up_questions': [],
                'response': final_response,
                'results_count': 0,
                'session_id': session_id or str(uuid.uuid4()),
                'thread_id': thread_id,
                'best_chart_type': "",
                'visualization_suggestion': ""
            }