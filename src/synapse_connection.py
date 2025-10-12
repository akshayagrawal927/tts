import pyodbc
from .config import SYNAPSE_CONFIG
import logging
import pandas as pd
from typing import List, Literal, Tuple
import streamlit as st


logger = logging.getLogger(__name__)
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