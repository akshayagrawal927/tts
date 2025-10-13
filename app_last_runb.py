import streamlit as st
import logging
from styles.css import load_professional_css
from utils.helper_components import init_session_state
from utils.helper_components import sidebar_interface
from utils.helper_components import main_chat_interface

logger = logging.getLogger(__name__)

def main():

    # Page configuration
    st.set_page_config(
        page_title="Pulse AI ",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded"
    )


    try:
        logger.info("Starting Pulse AI application")

        # Load CSS
        load_professional_css()

        # Initialize session state
        init_session_state()

        # Sidebar
        sidebar_interface()

        # Main chat interface
        main_chat_interface()


        logger.info("Pulse AI application loaded successfully")

    except Exception as e:
        logger.error(f"Main application error: {e}", exc_info=True)
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")

if __name__ == "__main__":
    # Run the application
    main()