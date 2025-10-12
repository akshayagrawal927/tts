import json 
import sqlite3
import uuid
from datetime import datetime
from typing import List, Dict,Set,Tuple
import regex as re
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class SessionManager:
    
    """Enhanced session management for multi-user chat history with thread safety."""

    def __init__(self, db_path: str = "Pulse_sessions.db"):
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
