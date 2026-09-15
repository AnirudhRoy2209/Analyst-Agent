import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llmpick import pickllm
from utils.etl_tools import etltools
from Model.schema import ETLAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


@tool
def extract_load_tool(url:str, output_folder:str, format:str):
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).
    
    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
            
    Returns:
        str: A message indicating the success or failure of the operation.
    
    """
    etl_tool=etltools()
    return etl_tool.extract_load(url, output_folder, format)

@tool
def transform_load_tool(input_file_path: str, output_folder: str, output_format: str, user_question: str):
    """
    This tool transforms the data from the specified file and loads it into the
    desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
        user_question (str): The user's transformation question.
    Returns:
        str: A message indicating the success or failure of the operation.
    """
    etl_tool = etltools()
    top_3_rows = etl_tool.transform_load_context(input_file_path)

    # Build the full output path with a filename
    output_filename = f"transformed_data.{output_format}"
    output_path = os.path.join(output_folder, output_filename)

    llm = pickllm("medium")

    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
            write the code to transform and save the data at {output_path}.
            The output path already includes the filename. Use it exactly as given. Do not change it.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n
        """

    ai_message = llm.invoke(prompt)
    raw = ai_message.content

    if isinstance(raw, list):
        text = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw
        )
    else:
        text = raw

    import re
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    pandas_code = match.group(1).strip() if match else text.strip()

    results = etl_tool.execute_code(pandas_code)

    return (
        f"The data is transformed and saved at {output_path} in {output_format} format. "
        f"\n\n Pandas Code Executed: \n {pandas_code} "
        f"\n\n Execution Result: \n {results}"
    )
    
tools=[extract_load_tool, transform_load_tool]

llm=pickllm("high")
llm_bind=llm.bind_tools(tools)

def llm_node(state:ETLAgentSchema):
    message=state.messages

    prompt = f"""
            You are a Python Data Analyst who has access to tools that can extract and load, 
            transform and load data. You will be provided with a user's question 
            and you would need to perform the right ETL operations as per the user's question. 
            If the operation is performed then inform the user and end the coversation.
            Here's the chat history: {message}\n
    """

    final_answer=llm_bind.invoke(prompt)

    state.messages=message+ [final_answer]

    return state

def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """

    tool_results=[]

    toolbyname={tool.name:tool for tool in tools}

    tool_calls=state.messages[-1].tool_calls

    for tool_call in tool_calls:
        tool=toolbyname[tool_call['name']]
        observation=tool.invoke(tool_call['args'])

        tool_results.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))

    state.messages=state.messages + tool_results

    return state



etl_analyst_graph=StateGraph(ETLAgentSchema)

etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)


etl_analyst_graph.add_edge(START, "llm_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls= state.messages[-1].tool_calls

    if tool_calls:
        return "tool_node"
    else:
        return "end"

etl_analyst_graph.add_conditional_edges(
    "llm_node",is_tool_call,
    {
        "tool_node":"tool_node",
        "end": END
    }
)

etl_analyst_graph.add_edge("tool_node", "llm_node")


etl_analyst=etl_analyst_graph.compile()

if __name__== "__main__":
    

    response=etl_analyst.invoke(
        {"messages": [HumanMessage(content=f"""
            I want to transform the data stored in the 'C:\\Users\\ANIDUDH\\OneDrive\\Desktop\\projects\\Analyst Agent\\Analyst Agent\\data\\extract\\extracted data.csv' file 
            and save the transformed data in the 'C:\\Users\\ANIDUDH\\OneDrive\\Desktop\\projects\\Analyst Agent\\Analyst Agent\\data\\transform' folder in the csv format.
            The transformation should filter the data to show bulbasaur pokemon
 """)]}
    )

    print(response)

