import streamlit as st 
import base64
import os
import logging
logger = logging.getLogger(__name__)

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
