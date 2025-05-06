import pandas as pd
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.memory import ChatMessageHistory
from langchain_openai import ChatOpenAI
from base import *
from dotenv import load_dotenv
from state import session_state
load_dotenv()  # Load environment variables from .env file
from typing import Optional
import logging
from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException,Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.responses import JSONResponse
from typing import List
import os, time
from io import BytesIO, StringIO
import nest_asyncio
from io import BytesIO
import openai
from sqlalchemy.orm import sessionmaker
from prompts import insight_prompt
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext
from azure.storage.blob import BlobServiceClient
from llama_parse import LlamaParse
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import json


from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from fastapi.staticfiles import StaticFiles
import plotly.graph_objects as go
import plotly.express as px
from langchain_openai import ChatOpenAI
import openai, yaml
import base64
from pydantic import BaseModel
from io import BytesIO
import os, csv
import pandas as pd

from langchain.chains.openai_tools import create_extraction_chain_pydantic
from langchain_core.pydantic_v1 import Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Optional
# Setup environment variables
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
DOCSTORE = os.getenv("DOCSTORE").split(",")
COLLECTION = os.getenv("COLLECTION").split(",")
Chroma_DATABASE = os.getenv("Chroma_DATABASE").split(",")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
os.environ["LLAMA_CLOUD_API_KEY"] = LLAMA_API_KEY
Settings.llm = OpenAI(model=LLM_MODEL)
Settings.embed_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)
models = os.getenv('models')
databases = os.getenv('databases').split(',')
subject_areas2 = os.getenv('subject_areas2').split(',')
print("subject_areas2",subject_areas2)
openai_ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL)
AZURE_CONTAINER_NAME = os.getenv("AZURE_CONTAINER_NAME")
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

app = FastAPI()

# Set up static files and templates
app.mount("/stats", StaticFiles(directory="stats"), name="stats")
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI API key and model

question_dropdown = os.getenv('Question_dropdown')
llm = ChatOpenAI(model=models, temperature=0)  # Adjust model as necessary
from table_details import get_table_details  # Importing the function

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")



from urllib.parse import quote  



# Set up static files and templates
app.mount("/stats", StaticFiles(directory="stats"), name="stats")
templates = Jinja2Templates(directory="templates")
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
    print("Blob service client initialized successfully.")
except Exception as e:
    print(f"Error initializing BlobServiceClient: {e}")
    # Handle the error appropriately, possibly exiting the application
    raise  # Re-raise the exception to prevent the app from starting

# Initialize OpenAI API key and model



llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)  # Adjust model as necessary
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

from table_details import get_table_details  # Importing the function

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description="Name of table in SQL database.")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):

    # Extract table names dynamically
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("index.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })
    
    
    


from fastapi import FastAPI, HTTPException, Depends, status, Form
import psycopg2
from psycopg2 import sql
class ChartRequest(BaseModel):
    """
    Pydantic model for chart generation requests.
    """
    table_name: str
    x_axis: str
    y_axis: str
    chart_type: str

    class Config:  # This ensures compatibility with FastAPI
        json_schema_extra = {
            "example": {
                "table_name": "example_table",
                "x_axis": "column1",
                "y_axis": "column2",
                "chart_type": "Line Chart"
            }
        }
        
class QueryInput(BaseModel):
    """
    Pydantic model for user query input.
    """
    query: str
@app.post("/add_to_faqs")
@app.post("/add_to_faqs/")
async def add_to_faqs(data: QueryInput):
    """
    Adds a user query to the FAQ CSV file on Azure Blob Storage.

    Args:
        data (QueryInput): The user query.

    Returns:
        JSONResponse: A JSON response indicating success or failure.
    """
    query = data.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Invalid query!")

    blob_name = 'Regulatory_questions.csv'

    try:
        # Get the blob client
        blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=blob_name)

        try:
            # Download the blob content
            blob_content = blob_client.download_blob().content_as_text()
        except ResourceNotFoundError:
            # If the blob doesn't exist, create a new one with a header if needed
            blob_content = "question\n"  # Replace with your actual header

        # Append the new query to the existing CSV content
        updated_csv_content = blob_content + f"{query}\n"  # Append new query

        # Upload the updated CSV content back to Azure Blob Storage
        blob_client.upload_blob(updated_csv_content.encode('utf-8'), overwrite=True)

        return {"message": "Query added to FAQs successfully and uploaded to Azure Blob Storage!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def download_as_excel(data: pd.DataFrame, filename: str = "data.xlsx"):
    """
    Converts a Pandas DataFrame to an Excel file and returns it as a stream.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        data.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output        
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=db_host,
            database=db_database,
            user=db_user,
            password=db_password
        )
        # Check if the connection is successful
        conn.cursor().execute("SELECT 1")
        print("Database connection established successfully.")
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to the database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to connect to the database"
        )

get_db_connection()

# Login endpoint
# Login endpoint
@app.post("/login")
async def login(
    email: str = Form(...),
    password: str = Form(...),
    section: str = Form(...),
):
    if not email or not password or not section:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="All fields are required"
        )

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Check if the user exists and the password matches
        cur.execute(
            sql.SQL("""
                SELECT u.user_id, u.full_name, r.role_name
                FROM lz_users u
                JOIN lz_user_roles ur ON u.user_id = ur.user_id
                JOIN lz_roles r ON ur.role_id = r.role_id
                WHERE u.email = %s AND u.password_hash = %s
            """),
            (email, password)
        )
        user = cur.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        user_id, full_name, role_name = user

        # Validate department access
        if (role_name in ["reg-admin", "reg-user"] and section != "Regulatory") or \
           (role_name in ["audit-admin", "audit-user"] and section != "Audit"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not authorized to access this department"
            )

        # Redirect based on role
        if role_name in ["reg-admin", "audit-admin"]:
            # For audit-admin, use "audit admin" as the name
            display_name = "Alameda Audit Admin" if role_name == "audit-admin" else full_name
            encoded_name = quote(display_name)
            encoded_section = quote(section)
            return RedirectResponse(
                url=f"/role-select?name={encoded_name}&section={encoded_section}",
                status_code=status.HTTP_303_SEE_OTHER
    )

       
        elif role_name in ["reg-user", "audit-user"]:
                
                display_name = full_name.replace("Tracking ", "") if role_name == "audit-user" else full_name
                encoded_name = quote(display_name)
                encoded_section = quote(section)
                return RedirectResponse(
                    url=f"/user_more?name={encoded_name}&section={encoded_section}",
                    status_code=status.HTTP_303_SEE_OTHER
        )

        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Unauthorized access"
            )

    finally:
        cur.close()
        conn.close()

        
def generate_chart_figure(data_df: pd.DataFrame, x_axis: str, y_axis: str, chart_type: str):
    """
    Generates a Plotly figure based on the specified chart type.
    """
    fig = None
    if chart_type == "Line Chart":
        fig = px.line(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Bar Chart":
        fig = px.bar(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Scatter Plot":
        fig = px.scatter(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Pie Chart":
        fig = px.pie(data_df, names=x_axis, values=y_axis)
    elif chart_type == "Histogram":
        fig = px.histogram(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Box Plot":
        fig = px.box(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Heatmap":
        fig = px.density_heatmap(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Violin Plot":
        fig = px.violin(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Area Chart":
        fig = px.area(data_df, x=x_axis, y=y_axis)
    elif chart_type == "Funnel Chart":
        fig = px.funnel(data_df, x=x_axis, y=y_axis)
    return fig

@app.post("/generate-chart/")
async def generate_chart(request: ChartRequest):
    """
    Generates a chart based on the provided request data.
    """
    try:
        table_name = request.table_name
        x_axis = request.x_axis
        y_axis = request.y_axis
        chart_type = request.chart_type

        if "tables_data" not in globals() or table_name not in globals()["tables_data"]:
            return JSONResponse(
                content={"error": f"No data found for table {table_name}"},
                status_code=404
            )

        data_df = globals()["tables_data"][table_name]
        fig = generate_chart_figure(data_df, x_axis, y_axis, chart_type)

        if fig:
            return JSONResponse(content={"chart": fig.to_json()})
        else:
            return JSONResponse(content={"error": "Unsupported chart type selected."}, status_code=400)

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while generating the chart: {str(e)}"},
            status_code=500
        )

@app.get("/download-table/")
async def download_table(table_name: str):
    """
    Downloads a table as an Excel file.
    """
    if "tables_data" not in globals() or table_name not in globals()["tables_data"]:
        raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

    data = globals()["tables_data"][table_name]
    output = download_as_excel(data, filename=f"{table_name}.xlsx")

    response = StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    response.headers["Content-Disposition"] = f"attachment; filename={table_name}.xlsx"
    return response
@app.get("/role-select", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("admin_landing_page.html", {"request": request})

@app.get("/user_client", response_class=HTMLResponse)
async def user_client(request: Request):
    # return templates.TemplateResponse("user_more.html", {"request": request})
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("user_client.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })


@app.get("/customer_landing_page", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("customer_landing_page.html", {"request": request})

@app.get("/authentication", response_class=HTMLResponse)
async def user_page(request: Request):
    return templates.TemplateResponse("authentication.html", {"request": request})

@app.get("/user_more", response_class=HTMLResponse)
async def user_more(request: Request):
    # return templates.TemplateResponse("user_more.html", {"request": request})
    tables = []

    # Pass dynamically populated dropdown options to the template
    return templates.TemplateResponse("user.html", {
        "request": request,
        "models": models,
        "databases": databases,  # Dynamically populated database dropdown
        "section": subject_areas2,
        "tables": tables,        # Table dropdown based on database selection
        "question_dropdown": question_dropdown.split(','),  # Static questions from env
    })

@app.post("/submit_feedback/")
async def submit_feedback(request: Request):
    data = await request.json() # Corrected for FastAPI
    
    table_name = data.get("table_name")
    feedback_type = data.get("feedback_type")
    user_query = data.get("user_query")
    sql_query = data.get("sql_query")

    if not table_name or not feedback_type:
        return JSONResponse(content={"success": False, "message": "Table name and feedback type are required."}, status_code=400)

    try:
        # Create database connection
        engine = create_engine(
        f'postgresql+psycopg2://{quote_plus(db_user)}:{quote_plus(db_password)}@{db_host}:{db_port}/{db_database}'
        )
        Session = sessionmaker(bind=engine)
        session = Session()

        # Sanitize input (Escape single quotes)
        table_name = escape_single_quotes(table_name)
        user_query = escape_single_quotes(user_query)
        sql_query = escape_single_quotes(sql_query)
        feedback_type = escape_single_quotes(feedback_type)

        # Insert feedback into database
        insert_query = f"""
        INSERT INTO lz_feedbacks (department, user_query, sql_query, table_name, data, feedback_type, feedback)
        VALUES ('unknown', :user_query, :sql_query, :table_name, 'no data', :feedback_type, 'user feedback')
        """

        session.execute(insert_query, {
        "table_name": table_name,
        "user_query": user_query,
        "sql_query": sql_query,
        "feedback_type": feedback_type
        })

        session.commit()
        session.close()

        return JSONResponse(content={"success": True, "message": "Feedback submitted successfully!"})

    except Exception as e:
        session.rollback()
        session.close()
        return JSONResponse(content={"success": False, "message": f"Error submitting feedback: {str(e)}"}, status_code=500)

@app.post("/transcribe-audio/")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribes an audio file using OpenAI's Whisper API.

    Args:
        file (UploadFile): The audio file to transcribe.

    Returns:
        JSONResponse: A JSON response containing the transcription or an error message.
    """
    try:
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing OpenAI API Key.")
        audio_bytes = await file.read()
        audio_bio = BytesIO(audio_bytes)
        audio_bio.name = "audio.webm"

        # Fix: Using OpenAI API correctly
        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_bio
        )

        # Fix: Access `transcript.text` instead of treating it as a dictionary
        return {"transcription": transcript.text}

    except Exception as e:
        return JSONResponse(content={"error": f"Error transcribing audio: {str(e)}"}, status_code=500)

@app.get("/get_questions/")
async def get_questions(subject: str):
    """
    Fetches questions from a CSV file in Azure Blob Storage based on the selected subject.

    Args:
        subject (str): The subject to fetch questions for.

    Returns:
        JSONResponse: A JSON response containing the list of questions or an error message.
    """
    csv_file_name = f"{subject}_questions.csv"
    blob_client = blob_service_client.get_blob_client(container=AZURE_CONTAINER_NAME, blob=csv_file_name)

    try:
        # Check if the blob exists
        if not blob_client.exists():
            print(f"file not found {csv_file_name}")
            return JSONResponse(
                content={"error": f"The file {csv_file_name} does not exist."}, status_code=404
            )

        # Download the blob content
        blob_content = blob_client.download_blob().content_as_text()

        # Read the CSV content
        questions_df = pd.read_csv(StringIO(blob_content))
        
        if "question" in questions_df.columns:
            questions = questions_df["question"].tolist()
        else:
            questions = questions_df.iloc[:, 0].tolist()

        return {"questions": questions}

    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred while reading the file: {str(e)}"}, status_code=500
        )

@app.get("/get-tables/")
async def get_tables(selected_section: str):
    # Fetch table details for the selected section
    print("now starting...")
    table_details = get_table_details(selected_section)
    # Extract table names dynamically
    print("till table details")
    tables = [line.split("Table Name:")[1].strip() for line in table_details.split("\n") if "Table Name:" in line]
    # Return tables as JSON
    return {"tables": tables}

def display_table_with_styles(data, table_name):

    page_data = data # Get whole table data

    styled_table = page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"') \
        .set_table_styles(
            [{
                'selector': 'th',
                'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]
            },
            {
                'selector': 'td',
                'props': [('border', '2px solid black'), ('padding', '5px')]
            }]
        ).to_html(escape=False)
    print(styled_table)
    return styled_table


# Invocation Function
def invoke_chain(question, messages, selected_model, selected_subject, selected_tools):
    try:
        print(selected_tools)
        history = ChatMessageHistory()
        for message in messages:
            if message["role"] == "user":
                history.add_user_message(message["content"])
            else:
                history.add_ai_message(message["content"])
        
        runner = graph.compile()
        result = runner.invoke({
            'question': question,
            'messages': history.messages,
            'selected_model': selected_model,
            'selected_subject': selected_subject,
            'selected_tools': selected_tools
        })
        
        print(f"Result from runner.invoke:", result)
        
        # Initialize response with common fields
        response = {
            "messages": result.get("messages", []),
            "follow_up_questions": {}
        }
        
        # Extract follow-up questions from all messages
        for message in result.get("messages", []):
            if hasattr(message, 'content'):
                content = message.content
                # Try to extract JSON from code block
                json_match = re.search(r'```json\n({.*?})\n```', content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group(1))
                        for key, value in data.items():
                            if key.startswith('follow_up_') and value:
                                response["follow_up_questions"][key] = value
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON from message: {e}")
        
        # Handle different intents
        if result.get("SQL_Statement"):
            print("Intent Classification: db_query")
            response.update({
                "intent": "db_query",
                "SQL_Statement": result.get("SQL_Statement"),
                "chosen_tables": result.get("chosen_tables", []),
                "tables_data": result.get("tables_data", {}),
                "db": result.get("db")
            })
        elif result.get("messages") and len(result["messages"]) > 0:
            last_message = result["messages"][-1]
            if hasattr(last_message, 'name'):
                print(f"Intent Classification: {last_message.name}")
                response.update({
                    "intent": last_message.name,
                    "search_results": last_message.content
                })
            else:
                print("Intent Classification: Unknown (no message name)")
                response.update({
                    "intent": "unknown",
                    "message": "This intent is not yet implemented."
                })
        
        print("Final response with follow-ups:", response)
        return response

    except Exception as e:
        print("Error:", e)
        return {
            "error": str(e),
            "message": "Insufficient information to generate SQL Query."
        }
    
@app.post("/submit")
async def submit_query(
    section: str = Form(...),
    example_question: str = Form(...),
    user_query: str = Form(...),
    tool_selected: List[str] = Form(default=[]),
    page: Optional[int] = Query(1),
    records_per_page: Optional[int] = Query(5),
):
    selected_subject = section
    session_state['user_query'] = user_query

    prompt = user_query if user_query else example_question
    if 'messages' not in session_state:
        session_state['messages'] = []

    session_state['messages'].append({"role": "user", "content": prompt})

    try:
        # If no tools selected, default to all
        if not tool_selected:
            tool_selected = ["all"]
        
        # If "all" is selected, use all available tools
        if "all" in tool_selected:
            tool_selected = ["intellidoc", "db_query", "researcher"]

        result = invoke_chain(
            prompt, 
            session_state['messages'], 
            "gpt-4o-mini", 
            selected_subject, 
            tool_selected
        )

        response_data = {
            "user_query": session_state['user_query'],
            "follow_up_questions": {},
            "intent": result.get("intent", ""),
            "selected_tools": tool_selected
        }

        # Handle different response types based on intent
        if result["intent"] == "db_query":
            session_state['generated_query'] = result.get("SQL_Statement", "")
            session_state['chosen_tables'] = result.get("chosen_tables", [])
            session_state['tables_data'] = result.get("tables_data", {})
            
            tables_html = []
            for table_name, data in session_state['tables_data'].items():
                html_table = display_table_with_styles(data, table_name)
                tables_html.append({
                    "table_name": table_name,
                    "table_html": html_table,
                })

            chat_insight = None
            if result["chosen_tables"]:
                insights_prompt = insight_prompt.format(
                    sql_query=result["SQL_Statement"],
                    table_data=result["tables_data"]
                )
                chat_insight = llm.invoke(insights_prompt).content

            response_data.update({
                "query": session_state['generated_query'],
                "tables": tables_html,
                "chat_insight": chat_insight
            })

        elif result["intent"] == "researcher":
            response_data["search_results"] = result.get("search_results", "No results found.")
            if "messages" in result:
                for message in result["messages"]:
                    if hasattr(message, 'content'):
                        response_data["search_results"] = message.content

        elif result["intent"] == "intellidoc":
            response_data["search_results"] = result.get("search_results", "No results found.")
            if "messages" in result:
                for message in result["messages"]:
                    if hasattr(message, 'content'):
                        response_data["search_results"] = message.content

        # Extract follow-up questions from all messages
        if "messages" in result:
            for message in result["messages"]:
                if hasattr(message, 'content'):
                    content = message.content
                    follow_ups = extract_follow_ups(content)
                    if follow_ups:
                        response_data["follow_up_questions"].update(follow_ups)

        print(f"Final response data: {response_data}")
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing the prompt: {str(e)}"
        )
        
@app.get("/get_table_data/")
async def get_table_data(
    table_name: str = Query(...),
    page_number: int = Query(1),
    records_per_page: int = Query(10),
):
    """Fetch paginated and styled table data."""
    try:
        # Check if the requested table exists in session state
        if "tables_data" not in session_state or table_name not in session_state["tables_data"]:
            raise HTTPException(status_code=404, detail=f"Table {table_name} data not found.")

        # Retrieve the data for the specified table
        data = session_state["tables_data"][table_name]
        total_records = len(data)
        total_pages = (total_records + records_per_page - 1) // records_per_page

        # Ensure valid page number
        if page_number < 1 or page_number > total_pages:
            raise HTTPException(status_code=400, detail="Invalid page number.")

        # Slice data for the requested page
        start_index = (page_number - 1) * records_per_page
        end_index = start_index + records_per_page
        page_data = data.iloc[start_index:end_index]

        # Style the table as HTML
        styled_table = (
            page_data.style.set_table_attributes('style="border: 2px solid black; border-collapse: collapse;"')
            .set_table_styles([
                {'selector': 'th', 'props': [('background-color', '#333'), ('color', 'white'), ('font-weight', 'bold'), ('font-size', '16px')]},
                {'selector': 'td', 'props': [('border', '2px solid black'), ('padding', '5px')]},
            ])
            .to_html(escape=False)  # Render as HTML
        )

        return {
            "table_html": styled_table,
            "page_number": page_number,
            "total_pages": total_pages,
            "total_records": total_records,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating table data: {str(e)}")


class Session:
    def _init_(self):
        self.data = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value

    def pop(self, key, default=None):
        return self.data.pop(key, default)

    def _contains_(self, item):
        return item in self.data

    def items(self):
        return self.data.items()

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def _iter_(self):
        return iter(self.data)
session = Session()

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

@app.route("/admin", methods=["GET", "POST"])
async def admin_page(request: Request):
    """Admin page to manage documents."""
    try:
        if request.method == "POST":
            form_data = await request.form()  # Parse the form data
            selected_section = form_data.get("section")
            print ("Selected section:", selected_section)
            # Ensure selected_section is valid
            if selected_section not in subject_areas2:
                raise ValueError("Invalid section selected.")

            collection_name = COLLECTION[subject_areas2.index(selected_section)]
            db_path = Chroma_DATABASE[subject_areas2.index(selected_section)]

            logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")

            return templates.TemplateResponse(
                'admin.html',
                {
                    "request": request,
                    "section": selected_section,
                    "collection": collection_name,
                    "db_path": db_path
                }
            )

        logging.info('Rendering admin page')
        return templates.TemplateResponse(
            'admin.html',
            {
                "request": request,
                "sections":subject_areas2
            }
        )
    except Exception as e:
        logging.error(f"Error rendering admin page: {e}")
        return JSONResponse(
            {"status": "error", "message": f"Error rendering admin page: {str(e)}"},
            status_code=500
        )

def upload_to_blob_storage(
                    connect_str: str, container_name: str,collection_name, file_content, file_name
                ):
                    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

                    # Ensure the container exists and create if necessary
                    container_client = blob_service_client.get_container_client(container_name)
                    blob_name = f"{collection_name}/{file_name}"
                    blob_client = container_client.get_blob_client(blob_name)

                    print(f"Uploading {file_name} to {blob_name}...")
                    blob_client.upload_blob(file_content, overwrite=True)        
        


@app.post("/upload")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    section : str = Form(...)
):
    """Handle file uploads for documents."""
    try:

        selected_section = section
        print(selected_section)
        # Ensure selected_section is valid
        if selected_section not in subject_areas2:
            raise ValueError("Invalid section selected.")

        collection_name = COLLECTION[subject_areas2.index(selected_section)]
        print("collection name",collection_name)
        db_path = Chroma_DATABASE[subject_areas2.index(selected_section)]

        print(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")

        if files:
            # logging.info(f"Handling upload for collection: {collection}, DB Path: {db_path}")
            
            for file in files:
                file_content = await file.read()
                file_name = file.filename

                upload_to_blob_storage(
                    AZURE_STORAGE_CONNECTION_STRING,
                    AZURE_CONTAINER_NAME,
                    collection_name,
                    file_content,
                    file_name,
                )

                try:
                    # Parse the uploaded file using LlamaParse
                    parsed_text =   await use_llamaparse(file_content, file_name)

                    # Split the parsed document into chunks
                    base_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
                    nodes = base_splitter.get_nodes_from_documents([Document(text=parsed_text)])

                    # Initialize storage context (defaults to in-memory)
                    storage_context = StorageContext.from_defaults()

                    # Prepare for storing document chunks
                    base_file_name = os.path.basename(file_name)
                    chunk_ids = []
                    metadatas = []

                    for i, node in enumerate(nodes):
                        chunk_id = f"{base_file_name}_{i + 1}"
                        chunk_ids.append(chunk_id)

                        metadata = {"type": base_file_name, "source": file_name}
                        metadatas.append(metadata)

                        document = Document(text=node.text, metadata=metadata, id_=chunk_id)
                        storage_context.docstore.add_documents([document])

                    # Load existing documents from the .json file if it exists
                    for i in range(len(DOCSTORE)):
                        if collection_name in DOCSTORE[i]:
                            coll = DOCSTORE[i]
                            print("collection name",coll)
                            break
                    existing_documents = {}
                    if os.path.exists(coll):
                        with open(coll, "r") as f:
                            existing_documents = json.load(f)

                        # Persist the storage context (if necessary)
                        storage_context.docstore.persist(coll)

                        # Load new data from the same file (or another source)
                        with open(coll, "r") as f:
                            st_data = json.load(f)

                        # Update existing_documents with st_data
                        for key, value in st_data.items():
                            if key in existing_documents:
                                # Ensure the existing value is a list before extending
                                if isinstance(existing_documents[key], list):
                                    existing_documents[key].extend(
                                        value
                                    )  # Merge lists if key exists
                                else:
                                    # If it's not a list, you can choose to replace it or handle it differently
                                    existing_documents[key] = (
                                        [existing_documents[key]] + value
                                        if isinstance(value, list)
                                        else [existing_documents[key], value]
                                    )
                            else:
                                existing_documents[key] = value  # Add new key-value pair

                        merged_dict = {}
                        for d in existing_documents["docstore/data"]:
                            merged_dict.update(d)
                        final_dict = {}
                        final_dict["docstore/data"] = merged_dict

                        # Write the updated documents back to the JSON file
                        with open(coll, "w") as f:
                            json.dump(final_dict, f, indent=4)

                    else:
                        # Persist the storage context if the file does not exist
                        storage_context.docstore.persist(coll)


                    collection_instance = init_chroma_collection(db_path, collection_name)

                    embed_model = OpenAIEmbedding()
                    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
                    batch_size = 500
                    for i in range(0, len(nodes), batch_size):
                        batch_nodes = nodes[i : i + batch_size]
                        try:
                            collection_instance.add(
                                documents=[node.text for node in batch_nodes],
                                metadatas=metadatas[i : i + batch_size],
                                ids=chunk_ids[i : i + batch_size],
                            )
                            time.sleep(5)  # Add a retry with a delay
                            logging.info(f"Files uploaded and processed successfully for collection: {collection_name}")
                            return JSONResponse({"status": "success", "message": "Documents uploaded successfully."})


                        except:
                            # Handle rate limit by adding a delay or retry mechanism
                            print("Rate limit error has occurred at this moment")
                            return JSONResponse({"status": "error", "message": f"Error processing file {file_name}."})



                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {e}")
                    return JSONResponse({"status": "error", "message": f"Error processing file {file_name}."})

        logging.warning("No files uploaded.")
        return JSONResponse({"status": "error", "message": "No files uploaded."})
    except Exception as e:
        logging.error(f"Error in upload_files: {e}")
        return JSONResponse({"status": "error", "message": "Error during file upload."})

import json
import re

def extract_follow_ups(message_content):
    """Extract follow-up questions from the message content."""
    try:
        # Try to find JSON in code block first
        json_match = re.search(r'```json\n({.*?})\n```', message_content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1))
            return {
                k: v for k, v in data.items() 
                if k.startswith('follow_up_') and v and isinstance(v, str)
            }
        
        # If no code block, try parsing the whole content as JSON
        try:
            data = json.loads(message_content)
            return {
                k: v for k, v in data.items()
                if k.startswith('follow_up_') and v and isinstance(v, str)
            }
        except json.JSONDecodeError:
            return {}
            
    except Exception as e:
        print(f"Error extracting follow-ups: {e}")
        return {}

async def use_llamaparse(file_content, file_name):
    try:
        with open(file_name, "wb") as f:
            f.write(file_content)

        # Ensure the result_type is 'text', 'markdown', or 'json'
        parser = LlamaParse(result_type='text', verbose=True, language="en", num_workers=2)
        documents =  await parser.aload_data([file_name])

        os.remove(file_name)

        res = ''
        for i in documents:
            res += i.text + " "
        return res
    except Exception as e:
        logging.error(f"Error parsing file: {e}")
        raise

def init_chroma_collection(db_path, collection_name):
    try:
        db = chromadb.PersistentClient(path=db_path)
        collection = db.get_or_create_collection(collection_name, embedding_function=openai_ef)
        logging.info(f"Initialized Chroma collection: {collection_name} at {db_path}")
        return collection
    except Exception as e:
        logging.error(f"Error initializing Chroma collection: {e}")
        raise


@app.post("/show_documents")
async def show_documents(request: Request,
                          section: str = Form(...)):
    """Show available documents."""
    try:

        selected_section = section
        # Ensure selected_section is valid
        if selected_section not in subject_areas2:
            raise ValueError("Invalid section selected.")

        collection_name = COLLECTION[subject_areas2.index(selected_section)]
        db_path = Chroma_DATABASE[subject_areas2.index(selected_section)]

        logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")

        if not collection_name or not db_path:
            raise ValueError("Missing 'collection' or 'db_path' query parameters.")

        # Initialize the collection
        collection = init_chroma_collection(db_path, collection_name)

        # Retrieve metadata and IDs from the collection
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']

        # Create a dictionary mapping document names to IDs
        doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                doc_name = meta['source'].split('\\')[-1]
                if doc_name not in doc_name_to_id:
                    doc_name_to_id[doc_name] = []
                doc_name_to_id[doc_name].append(doc_id)

        # Get the unique document names
        doc_list = list(doc_name_to_id.keys())
        logging.info(f"Documents retrieved successfully for collection: {collection_name}")
        return doc_list

       

    except Exception as e:
        logging.error(f"Error showing documents: {e}")
        return JSONResponse({"status": "error", "message": "Error showing documents."})

@app.post("/delete_document")
async def delete_document(request: Request,
                         section: str = Form(...),
                          doc_name: str = Form(...)):
    """Handle document deletion."""
    try:
        selected_section = section
    # Ensure selected_section is valid
        if selected_section not in subject_areas2:
            raise ValueError("Invalid section selected.")

        collection_name = COLLECTION[subject_areas2.index(selected_section)]
        db_path = Chroma_DATABASE[subject_areas2.index(selected_section)]

        logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")
        # Initialize the collection
        collection = init_chroma_collection(db_path, collection_name)
        print("document to be deleted",doc_name)
        
        if doc_name:
              def delete_from_blob_storage(connect_str: str, container_name: str, file_name: str,collection_name):
                    # Create a BlobServiceClient to interact with the Azure Blob Storage
                    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

                    # Get the container client
                    container_client = blob_service_client.get_container_client(container_name)

                    # Create the blob name with the collection prefix
                    blob_name = f"{collection_name}/{file_name}"

                    # Get the blob client for the specific file (blob)
                    blob_client = container_client.get_blob_client(blob_name)

                    # Delete the specified blob (file)
                    try:
                        print(f"Deleting {blob_name} from blob storage...")
                        blob_client.delete_blob()
                        print(
                            f"Blob '{blob_name}' deleted successfully from container '{container_name}'."
                        )
                    except Exception as e:
                        print(f"Failed to delete blob: {e}")
        # Retrieve metadata and IDs from the collection
        docs = collection.get()['metadatas']
        ids = collection.get()['ids']

        # Create a dictionary mapping document names to IDs
        doc_name_to_id = {}
        for doc_id, meta in zip(ids, docs):
            if 'source' in meta:
                name = meta['source'].split('\\')[-1]
                if name not in doc_name_to_id:
                    doc_name_to_id[name] = []
                doc_name_to_id[name].append(doc_id)

        # Get the unique document names
        ids_to_delete = doc_name_to_id.get(doc_name, [])

        print("Document name: ", doc_name)
        print("IDs to delete: ", ids_to_delete)

        delete_from_blob_storage(
                        AZURE_STORAGE_CONNECTION_STRING,
                        AZURE_CONTAINER_NAME,
                        doc_name,
                        collection_name )
        
      
        
        if ids_to_delete:
            
            # Attempt deletio
            collection.delete(ids=ids_to_delete)

             # Step 1: Read the JSON file
            for i in range(len(DOCSTORE)):
                if collection_name in DOCSTORE[i]:
                    coll = DOCSTORE[i]
                    break
            with open(coll, 'r') as file:
                data = json.load(file)["docstore/data"]

            for i in ids_to_delete:
                del data[i]

            final_dict = {}
            final_dict["docstore/data"] = data


            with open(coll, 'w') as file:
                json.dump(final_dict, file, indent=4)

            logging.info(f"Document '{doc_name}' deleted successfully.")
            return JSONResponse({"status": "success", "message": f"Document '{doc_name}' deleted successfully."})
        else:
            logging.warning(f"Document '{doc_name}' not found for deletion.")
            return JSONResponse({"status": "error", "message": "Document not found."})
    except Exception as e:
        logging.error(f"Error deleting document '{doc_name}': {e}")
        print(f"Error deleting document: {e}")  # Print exception for debugging
        return JSONResponse({"status": "error", "message": "Error deleting document."})
