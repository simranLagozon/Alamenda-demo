from langgraph.graph import Graph
from langchain.chat_models import ChatOpenAI
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from langgraph.graph import StateGraph,END,START
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from table_details import table_chain as select_table
from table_details import get_table_details, get_tables,  create_extraction_chain_pydantic, Table
from prompts import final_prompt
from langchain.chains import create_sql_query_chain
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Optional, TypedDict,Tuple
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
# Define the intellidoc tool
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
import logging
from fastapi.responses import JSONResponse
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv


import chromadb
import os
import json
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
db_user = os.getenv("db_user")
db_password = os.getenv("db_password")
db_host=os.getenv("db_host")
#db_warehouse=os.getenv("db_warehouse")
db_database=os.getenv("db_database")
db_port=os.getenv("db_port")
db_schema= os.getenv("db_schema")
adv_db_database = os.getenv("adv_db_database")
adv_db_schema = os.getenv("adv_db_schema")
adv_db_schema_hr = os.getenv("adv_db_schema_hr")
adv_db_schema_pe = os.getenv("adv_db_schema_pe")
adv_db_schema_purchase = os.getenv("adv_db_schema_purchase")
adv_db_schema_sales = os.getenv("adv_db_schema_sales")
#table_details_prompt = os.getenv('TABLE_DETAILS_PROMPT')
# Change if your schema is different
DOCSTORE = os.getenv("DOCSTORE").split(",")
COLLECTION = os.getenv("COLLECTION").split(",")
Chroma_DATABASE = os.getenv("Chroma_DATABASE").split(",")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

QA_PROMPT_STR = os.getenv("QA_PROMPT_STR")
LLM_INSTRUCTION = os.getenv("LLM_INSTRUCTION")
NO_METADATA = os.getenv("NO_METADATA")
METADATA_INSTRUCTION = os.getenv("METADATA_INSTRUCTION").split(",")
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)


class GraphState(TypedDict):
    question: str
    messages: list
    selected_model: str
    selected_subject: str
    chosen_tables: list
    SQL_Statement: str
    tables_data: dict
    selected_tools: list[str]  # Now a list of tools



def classify_intent(state: GraphState) -> str:
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    user_question = state["messages"][-1].content

    # FIX: Use correct key name "selected_tools"
    tool_selected = state.get("selected_tools", ["all"])  # ✅ Changed key


    intent_prompt = f"""You are an intent classifier. Your job is to determine which agent is most appropriate to answer the user's question.
    The possible agents are:
    - researcher: Use this agent when the question requires searching the internet for information about current events or general knowledge.
    - db_query: Use this agent when the question requires querying a database to retrieve specific data.
    - intellidoc: Use this agent when the question requires retrieving information from existing documents.

    Question: {user_question}
    Return the name of the agent only.
    """

    # Add a note about available tools
    if "all" not in tool_selected:
        available_tools = ", ".join(tool_selected)
        intent_prompt += f"\nNote: Only the following agents are available: {available_tools}."

    intent_chain = RunnablePassthrough() | llm | StrOutputParser()
    intent = intent_chain.invoke([HumanMessage(content=intent_prompt)]).strip().lower()

    # Validate the intent against the selected tools
    if "all" not in tool_selected and intent not in tool_selected:
        # If the intent is not in the selected tools, default to the first tool in the list
        intent = tool_selected[0]

    print(f"Intent Classification: {intent}")
    return intent
BING_API_KEY = os.getenv("BING_API_KEY")

# Bing Search Tool
def bing_search(query: str) -> str:
    """Query Bing Search API to get summarized search results."""
    api_key = os.getenv('BING_API_KEY')
    url = f'https://api.bing.microsoft.com/v7.0/search?q={query}'
    headers = {'Ocp-Apim-Subscription-Key': api_key}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        results = response.json()
        summaries = []
        for result in results.get('webPages', {}).get('value', []):
            title = result.get('name', 'No Title')
            snippet = result.get('snippet', 'No summary available.')
            summaries.append(f"**{title}**: {snippet}")

        return "\n\n".join(summaries) if summaries else "No results found."
    else:
        return f"Error: {response.status_code}"


bing_tool = Tool(
    name="Bing Search",
    func=bing_search,
    description="Useful for general web searches."
)



tools = [bing_tool]

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4o-mini",
    temperature=0.5
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # Better for conversations
    memory=memory,
    verbose=True
)

def researcher_node(state: GraphState) -> dict:
    """
    Handles the researcher intent by summarizing search results.
    """
    user_question = state["messages"][-1].content
    raw_results = agent.run(user_question)

    summarization_prompt = f"""Summarize the following search result content into concise, informative points without listing URLs:

{raw_results}
"""

    summary = llm.predict(summarization_prompt)

    return {
        "messages": [
            HumanMessage(content=summary, name="researcher")
        ]
    }


def hybrid_retrieve(query, docstore, vector_index, bm25_retriever, alpha=0.5):
    """Perform hybrid retrieval using BM25 and vector-based retrieval."""
    # Get results from BM25
    try:
        bm25_results = bm25_retriever.retrieve(query)
        # Get results from the vector store
        vector_results = vector_index.as_retriever(similarity_top_k=2).retrieve(query)
    except Exception as e:
        logging.error(e)
        return JSONResponse("Error with retriever")
    # Combine results with weighting
    combined_results = {}
    # Weight BM25 results
    for result in bm25_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + (1 - alpha)
    # Weight vector results
    for result in vector_results:
        combined_results[result.id_] = combined_results.get(result.id_, 0) + alpha

    # Sort results based on the combined score
    sorted_results = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)
    # Return the top N results
    return [docstore.get_document(doc_id) for doc_id, _ in sorted_results[:4]]


def init_chroma_collection(db_path, collection_name):
    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_or_create_collection(collection_name, embedding_function=openai_ef)
        logging.info(f"Initialized Chroma collection: {collection_name} at {db_path}")
        return collection
    except Exception as e:
        logging.error(f"Error initializing Chroma collection: {e}")
        raise



def intellidoc_tool(department: str, query_text: str):
    """
    Intellidoc Tool: Processes a query for a given department and returns the response.

    Args:
        department (str): The department for which the query is being processed.
        query_text (str): The user's query.

    Returns:
        Tuple[str, str, str]: A tuple containing the query_text, response_text, and source.
    """
    print(department)
    try:
        # Map the department to its index in the lists
        DEPARTMENT_TO_INDEX = {
            # "human_resources": 0,
            # "legal": 1,
            # "finance": 2,
            # "operations": 3,
            # "healthcare": 4,
            # "insurance": 5,
            # "learning_and_development": 6,
            # "others": 7
            # "Adv-Manufacturing":0,
            # "Adv-Inventory":1,
            "Regulatory":0,
            "Audit":1
           
        }
        department_index = DEPARTMENT_TO_INDEX.get(department)

        if department_index is None:
            return query_text, "Invalid department selected.", ""

        docstore_file = DOCSTORE[department_index]
        collection_name = COLLECTION[department_index]
        db_path = Chroma_DATABASE[department_index]

        if not os.path.exists(docstore_file):
            return query_text, f"Document store not found for {department}.", ""

        # Proceed with querying the documents
        collection = init_chroma_collection(db_path, collection_name)
        if "documents" in collection.get() and len(collection.get()['documents']) > 0:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            docstore = SimpleDocumentStore.from_persist_path(docstore_file)
            storage_context = StorageContext.from_defaults(docstore=docstore, vector_store=vector_store)
            vector_index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=OpenAIEmbedding(api_key=OPENAI_API_KEY))

            bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=2)
            retrieved_nodes = hybrid_retrieve(query_text, docstore, vector_index, bm25_retriever, alpha=0.8)
            ids = [doc.id_ for doc in retrieved_nodes]

            context_str = "\n\n".join([node.get_content().replace('{', '').replace('}', '') for node in retrieved_nodes])

            qa_prompt_str = QA_PROMPT_STR
            fmt_qa_prompt = qa_prompt_str.format(context_str=context_str, query_str=query_text)

            chat_text_qa_msgs = [
                ChatMessage(
                    role=MessageRole.SYSTEM,
                    content=LLM_INSTRUCTION
                ),
                ChatMessage(
                    role=MessageRole.USER,
                    content=fmt_qa_prompt
                ),
            ]

            text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)
            result = vector_index.as_query_engine(text_qa_template=text_qa_template, llm=OpenAI(model=LLM_MODEL)).query(query_text)
            response_text = result.response
            print("response_text:",response_text)
            source = " and ".join(ids)  # Default to joining IDs

            # Check if any instruction is present in the response
            for instruction in METADATA_INSTRUCTION:
                if instruction in str(response_text).lower():
                    source = NO_METADATA
                    break  # Exit the loop if a match is found

            return query_text, response_text, source

        return query_text, "No documents found in the collection.", ""

    except Exception as e:
        logging.error(f"Error in intellidoc_tool: {e}")
        return query_text, f"Error processing query: {str(e)}", ""


# Define the intellidoc node
def intellidoc_node(state: GraphState) -> dict:
    """
    Handles the intellidoc intent by retrieving information from documents.
    """
    user_question = state["messages"][-1].content  # Get the last user message
    department = state.get("selected_subject", "human_resources")  # Default to human_resources if not specified

    # Use the intellidoc tool to retrieve information
    search_results = intellidoc_tool(department, user_question)

    # Unpack the results correctly
    query_text, response_text, source = search_results
    # Format the results into a single string for HumanMessage content
    formatted_content = f"{response_text}\n\nSource: {source}"
    # Use the intellidoc tool to retrieve information

    return {
        "messages": [
            HumanMessage(content=formatted_content, name="intellidoc")
        ]
    }
# Define Node Functions
def extract_tables(data: GraphState) -> dict:
    question = data['question']
    selected_subject = data.get('selected_subject', 'Adv-Manufacturing')
    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    table_details = get_table_details(selected_subject)
    table_details_set_prompt = os.getenv('TABLE_DETAILS_SET_PROMPT')
    table_details_prompt = table_details_set_prompt.format(table=table_details)

    print("From utils.py Table_details_prompt:", table_details_prompt)

    table_chain = {"input": itemgetter("question")} | create_extraction_chain_pydantic(Table, llm, system_message=table_details_prompt) | get_tables
    chosen_tables = table_chain.invoke({"question": question})

    return {'chosen_tables': chosen_tables, 'question': question, 'selected_model': 'gpt-4o-mini', 'selected_subject': selected_subject, 'messages': data['messages']}


def generate_sql(data: GraphState)-> dict:
    selected_subject = data['selected_subject']
    print("This is selectedddd subject:",selected_subject)

    if selected_subject.startswith('Adv'):

        if selected_subject.endswith('HumanResources'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_hr
                    ,include_tables= data['chosen_tables']
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)
        elif selected_subject.endswith('Purchasing'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_purchase
                    ,include_tables= data['chosen_tables']
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)
        elif selected_subject.endswith('Sales'):
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                    ,schema=adv_db_schema_sales
                    ,include_tables= data['chosen_tables']
                    , view_support=True
                    ,sample_rows_in_table_info=1
                    ,lazy_table_reflection=True
                    )
            print("DB Connection Done for Adventureworks----",db._schema)

        else:
            db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}'
                                ,schema=adv_db_schema
                                ,include_tables= data['chosen_tables']
                                , view_support=True
                                ,sample_rows_in_table_info=1
                                ,lazy_table_reflection=True
                                )
            print("DB Connection Done for Adventureworks----",db._schema)

    else:
        db = SQLDatabase.from_uri(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}'
                                ,schema=db_schema
                                ,include_tables= data['chosen_tables']
                                , view_support=True
                                ,sample_rows_in_table_info=1
                                ,lazy_table_reflection=True
                                )
        print("DB Connection Done for PostGress---",db._schema)
        #for testing of synapse


    llm = ChatOpenAI(model=data['selected_model'], temperature=0)
    print("Generate Query Starting")

    generate_query = create_sql_query_chain(llm, db, final_prompt)
    SQL_Statement = generate_query.invoke({"question": data['question'], "messages": data['messages']})
    print(f"Generated SQL Statement before execution: {SQL_Statement}")

    return {'SQL_Statement': SQL_Statement, 'db': db, 'chosen_tables': data['chosen_tables'], 'question': data['question'], 'messages': data['messages']}


def execute_sql(data: GraphState) -> dict:
    print("dataaaa:", data)
    selected_subject = data['selected_subject']
    SQL_Statement = data['SQL_Statement'].replace("SQL Query:", "").strip()

    if selected_subject.startswith('Adv'):
        alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{adv_db_database}')
    else:
        alchemyEngine = create_engine(f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}')

    tables_data = {}
    for table in data['chosen_tables']:
        query = SQL_Statement + ";"
        print(f"Executing SQL Query: {query}")
        with alchemyEngine.connect() as conn:
            df = pd.read_sql(sql=query, con=conn.connection)
            tables_data[table] = df
            break  # Execute only once as in the original code

    # Create the rephrasing chain
    llm = ChatOpenAI(model=data['selected_model'], temperature=0)

    rephrase_answer = (
        RunnablePassthrough.assign(
            answer=lambda x: llm.invoke(
                f"""Rephrase this SQL result into a clear answer for the user's question.
                Question: {x["question"]}
                SQL Query: {x["query"]}
                Result Data:\n{x["result"].to_string()}

                Provide a concise answer in natural language.
                Also suggest 3 relevant follow-up questions the user might ask next.
                Format your response as JSON with these keys:
                - "answer": The main answer
                - "follow_up_1": First follow-up question
                - "follow_up_2": Second follow-up question
                - "follow_up_3": Third follow-up question
                """
            ).content
        )
    )

    # Invoke the chain
    response = rephrase_answer.invoke({
        "question": data['question'],
        "query": SQL_Statement,
        "result": df,  # The pandas DataFrame result
        "answer": "",
        "follow_up_1": "",
        "follow_up_2": "",
        "follow_up_3": ""
    })

    try:
        # Parse the JSON response
        response_data = json.loads(response["answer"])
        formatted_answer = response_data["answer"]
        follow_ups = [
            response_data["follow_up_1"],
            response_data["follow_up_2"],
            response_data["follow_up_3"]
        ]
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        formatted_answer = response["answer"]
        follow_ups = []

    return {
        'SQL_Statement': SQL_Statement,
        'chosen_tables': data['chosen_tables'],
        'tables_data': tables_data,
        'messages': [
            HumanMessage(content=formatted_answer, name="sql_answer"),
            *[HumanMessage(content=fq, name="follow_up") for fq in follow_ups]
        ]
    }


graph = StateGraph(GraphState)
print("Graph Created")
# Add nodes
graph.add_node("classify_intent", lambda state: {"intent": classify_intent(state)})
graph.add_node("extract_tables", extract_tables)
graph.add_node("generate_sql", generate_sql)
graph.add_node("execute_sql", execute_sql)
graph.add_node("researcher", researcher_node)
graph.add_node("intellidoc", intellidoc_node)

# Add edges
graph.add_edge(START, "classify_intent")

# Conditional edges based on intent and tool_selected
def conditional_edges(state: GraphState):
    intent = state["intent"]
    tool_selected = state.get("tool_selected", ["all"])

    if "all" in tool_selected:
        # All tools are available
        if intent == "db_query":
            return "extract_tables"
        elif intent == "researcher":
            return "researcher"
        elif intent == "intellidoc":
            return "intellidoc"
    else:
        # Only specific tools are available
        if intent in tool_selected:
            if intent == "db_query":
                return "extract_tables"
            elif intent == "researcher":
                return "researcher"
            elif intent == "intellidoc":
                return "intellidoc"
        else:
            # If the intent is not in the selected tools, default to the first tool in the list
            return tool_selected[0]

    return END  # Default to end if no matching intent

graph.add_conditional_edges("classify_intent", conditional_edges)

graph.add_edge("extract_tables", "generate_sql")
graph.add_edge("generate_sql", "execute_sql")
graph.add_edge("execute_sql", END)
graph.add_edge("researcher", END)
graph.add_edge("intellidoc", END)
