import os 
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llmpick import pickllm
from Model.schema import AgentSchema, JudgeSchema
from langchain_core.messages import HumanMessage, AIMessage
from utils.db import Databaseutil
from langgraph.graph import StateGraph, START, END

def content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return str(content)


def curate_ques(state:AgentSchema)->AgentSchema:
    user_qns=state.user_question

    llm= pickllm("low")

    response=content_to_text(
        llm.invoke(f"curate the following question: {user_qns}").content
    )

    state.curated_qns=response
    state.messages=state.messages+ [HumanMessage(content=f"{response}")]

    return state


def prompt_query_context(state:AgentSchema)->AgentSchema:
    curate_question= state.curated_qns

    conn_details = {
        "host": os.environ['host'],
        "port": os.environ['port'],
        "user": os.environ['user'],
        "password": os.environ['password'],
        "dbname": os.environ['database']
    }

    obj= Databaseutil(conn_details)
    schema_info=obj.schema_details("public")

    prompt = f"""
    You are an SQL analyst agent. Your task is to convert the user's natural language 
    query into Postgres SQL query that can be executed on the database. You are provided 
    with the user's original query and the schema details of the database, including
    table names, column names, data types, and sample data for each table so that 
    you can understand the structure of the database and generate an accurate SQL query.
    Unless user explicitly asks for specific number of rows, always limit the output to 10 rows.
    Note - Just generate the SQL query without any explanation or additional text because
    this query will be executed directly on the database. So, the output should be SQL
    ready to be executed without any modifications.  
    
    User's Original Query: {curate_question}

    Database Schema Details:
    {schema_info}
    
    """    

    state.prompt_query_context=prompt
    
    return state


def generate_sql(state:AgentSchema)->AgentSchema:
    prompt= state.prompt_query_context
    llm= pickllm("medium")
    raw_response = content_to_text(llm.invoke(prompt).content)
    
    # Strip markdown code blocks if the LLM added them
    clean_sql = raw_response.strip()
    if clean_sql.startswith("```"):
        lines = clean_sql.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]  # Remove top ```sql
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1] # Remove bottom ```
        clean_sql = "\n".join(lines).strip()
        
    state.generated_sql_query = clean_sql
    return state


def is_safe(state:AgentSchema)->AgentSchema:
    sql_query=state.generated_sql_query

    llm=pickllm("medium")
    llm_judge=llm.with_structured_output(JudgeSchema)

    prompt = f"""
    You are an SQL Judge for data security. Your task is to determine whether the SQL query is 
    safe or not. The SQL query should only be used for data retrieval and should not modify the 
    database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the
    database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change
    the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with 
    'No'. Additionally, provide comments explaining your decision.
    Here's the SQL query to evaluate:
    {sql_query}"""

    response=llm_judge.invoke(prompt)
    if isinstance(response, dict):
        state.is_safe = response.get("answer", "No")
        state.comments = response.get("comments", "")
    else:
        state.is_safe = getattr(response, "answer", "No")
        state.comments = getattr(response, "comments", "")

    return state

def cancelled_sql(state:AgentSchema)->AgentSchema:
    comments= state.comments

    state.final_answer= f"the generated SQL query was deemed unsafe to execute. The reason provided by the judge: {comments}"
    state.messages=state.messages+ [AIMessage(content=f"{state.final_answer}")]
    return state

def execute_sql(state:AgentSchema)->AgentSchema:
    sql_query=state.generated_sql_query

    conn_details = {
        "host": os.environ['host'],
        "port": os.environ['port'],
        "user": os.environ['user'],
        "password": os.environ['password'],
        "dbname": os.environ['database']
    }

    obj=Databaseutil(conn_details)

    execution_result=obj.execute_sql(sql_query)

    state.sql_query_execution_result=execution_result
    return state

def represent_final_answer(state:AgentSchema)->AgentSchema:
    sql_query=state.sql_query_execution_result
    curated_qns=state.curated_qns
    llm=pickllm("low")

    prompt= f"""
    You are an SQL analyst agent. Your task is to provide a final answer to the user based on the
    execution result of the SQL query and the user's original question. The final answer should be
    concise, clear, and directly address the user's query. Avoid including any SQL code or technical
    details in the final answer. The final answer should be in a user-friendly format that is easy to
    understand. If the execution result is empty or does not provide a clear answer to the user's question, explain this in the final answer. \n
    Here is the execution result: {sql_query} \n
    Here is the user's original question: {curated_qns}
    """
    llm_response=content_to_text(llm.invoke(prompt).content)
    
    state.final_answer=llm_response
    state.messages=state.messages+ [AIMessage(content=f"{llm_response}")]
    return state


sql_agent_graph=StateGraph(AgentSchema)

sql_agent_graph.add_node("curate_qns", curate_ques)
sql_agent_graph.add_node("prompt_query_context", prompt_query_context)
sql_agent_graph.add_node("generate_sql", generate_sql)
sql_agent_graph.add_node("is_safe", is_safe)
sql_agent_graph.add_node("cancelled_node", cancelled_sql)
sql_agent_graph.add_node("execute_sql", execute_sql)
sql_agent_graph.add_node("represent_node", represent_final_answer)

sql_agent_graph.add_edge(START, "curate_qns")
sql_agent_graph.add_edge("curate_qns", "prompt_query_context")
sql_agent_graph.add_edge("prompt_query_context", "generate_sql")
sql_agent_graph.add_edge("generate_sql", "is_safe")

def is_safe_sql(state:AgentSchema)->AgentSchema:
    is_safer=state.is_safe

    if is_safer.lower()=="yes":
        return "execute_sql"
    else:
        return "cancelled_node"


sql_agent_graph.add_conditional_edges("is_safe", is_safe_sql,
                                      {
                                          "execute_sql":"execute_sql",
                                          "cancelled_node":"cancelled_node"
                                      })



sql_agent_graph.add_edge("cancelled_node", END)
sql_agent_graph.add_edge("execute_sql", "represent_node")
sql_agent_graph.add_edge("represent_node", END)

sql_analyst=sql_agent_graph.compile()

if __name__ == "__main__":

    

    input_schema = {
        "messages": [],
        "user_question": "What are the different types of Payment Methods we have in our database",
        "curated_qns": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": ""
    }

    sql_analyst_response = sql_analyst.invoke(input_schema)
    print(sql_analyst_response['messages']) 
    print("********************************")

    print(sql_analyst_response['generated_sql_query'])  

    print("********************************")

    print(sql_analyst_response['sql_query_execution_result'])  

    print("********************************")

    print(sql_analyst_response['prompt_query_context'])





