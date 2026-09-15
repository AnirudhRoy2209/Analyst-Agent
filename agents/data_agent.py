import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llmpick import pickllm
from utils.etl_tools import etltools
from Model.schema import RouterSchema, DataAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from agents.etl_analyst import etl_analyst
from agents.sql_analyst import sql_analyst

llm=pickllm("high")

llm_router=llm.with_structured_output(RouterSchema)

def router_node(state:DataAgentSchema):
    user_question=state.messages[-1].content

    prompt = f"""
    You are a routing classifier. Your ONLY job is to classify the user's question.

    Classify as "sql" if the question is about querying a database.
    Classify as "etl" if the question is about extracting, transforming, or loading data.

    DO NOT answer the question. DO NOT write code. DO NOT explain your reasoning.
    ONLY call the classification tool.

    User question: {user_question}
    """

    route_response_dict=llm_router.invoke(prompt).model_dump()

    route_response=route_response_dict['answer']

    state.route_response=route_response
    return {"route_response": route_response}

def etl_node(state:DataAgentSchema):
    message=state.messages[-1].content

    response=etl_analyst.invoke(
        {"messages":[HumanMessage(content=message)]}
    )

    new_messages= response["messages"][-1]

    return {"messages": new_messages}

    

def sql_node(state: DataAgentSchema):
    message = state.messages[-1].content

    input_schema = {
        "messages": [HumanMessage(content=message)],   # give it the question
        "user_question": message,
        "curated_qns": "",
        "prompt_query_context": "",
        "generated_sql_query": "",
        "is_safe": "No",
        "comments": "",
        "sql_query_execution_result": "",
        "final_answer": "",
    }

    response = sql_analyst.invoke(input_schema)

    # response is the SQL agent's final state; grab its messages
    new_messages = response["messages"][-1]

    return {"messages": new_messages}


data_agent_graph=StateGraph(DataAgentSchema)

data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("sql_node", sql_node)

data_agent_graph.add_edge(START, "router_node")

def router_edge(state:DataAgentSchema):
    if state.route_response=="etl":
        return "etl_node"
    elif state.route_response=="sql":
        return "sql_node"
    else:
        raise ValueError(f"Invalid route response: {state.route_response}")


data_agent_graph.add_conditional_edges("router_node", router_edge,
                                       {
                                           "etl_node":"etl_node",
                                           "sql_node":"sql_node"
                                       })

data_agent= data_agent_graph.compile()

if __name__=="__main__":
    response=data_agent.invoke({
        "messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")],
        "route_response": ""
    })

    print(response)
