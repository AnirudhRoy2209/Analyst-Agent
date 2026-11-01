import psycopg2


class Databaseutil:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = None
        self.conn_error = None
        try:
            # 10-second timeout prevents infinite "Thinking..." hangs
            self.connection = psycopg2.connect(**db_config, connect_timeout=10)
        except Exception as e:
            self.conn_error = str(e)
            print(f"error connecting to the database: {e}")

    def schema_details(self, schema_name):
        if not self.connection:
            raise ConnectionError(f"Database connection failed: {self.conn_error}")

        schema_info_context = f"Database Schema: {schema_name}\n"
        cursor = None

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema=%s;",
                (schema_name,),
            )
            tables_list = cursor.fetchall()

            for table in tables_list:
                table_name = table[0]
                schema_info_context += f"\nTable: {table_name}\n"

                cursor.execute(
                    "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema=%s AND table_name=%s;",
                    (schema_name, table_name),
                )
                column_list = cursor.fetchall()

                for column_name, data_type in column_list:
                    schema_info_context += f" Column: {column_name}, Data Type: {data_type}\n"

                cursor.execute(f'SELECT * FROM "{schema_name}"."{table_name}" LIMIT 5;')
                sample_data = cursor.fetchall()
                schema_info_context += " Sample Data:\n"
                for row in sample_data:
                    schema_info_context += f" {row}\n"

        except Exception as e:
            print(f"error fetching schema details: {e}")
            schema_info_context = f"Error fetching schema details: {e}"
        finally:
            if cursor:
                cursor.close()
            if self.connection:
                self.connection.close()

        return schema_info_context

    def execute_sql(self, query):
        if not self.connection:
            return f"Database connection failed: {self.conn_error}"

        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()
            self.connection.commit()
            return str(result)
        except Exception as e:
            print(f"error executing query: {e}")
            return f"Error executing query: {e}"
        finally:
            if cursor:
                cursor.close()
            if self.connection:
                self.connection.close()
            
        