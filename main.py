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
import nest_asyncio
from io import BytesIO
import openai

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

        # Redirect based on role
        if role_name == "admin":
            encoded_name = quote(full_name)
            encoded_section = quote(section)
            # return RedirectResponse(url=f"/admin?section={encoded_section}", status_code=status.HTTP_303_SEE_OTHER)
            return RedirectResponse(url=f"/role-select?name={encoded_name}&section={encoded_section}", status_code=status.HTTP_303_SEE_OTHER)
            
        elif role_name == "user":
            # Use urllib.parse.quote to encode full_name and section
            encoded_name = quote(full_name)
            encoded_section = quote(section)
            return RedirectResponse(
                url=f"/user_more?name={encoded_name}&section={encoded_section}",
                status_code=status.HTTP_303_SEE_OTHER
            )
        elif role_name == "viewer":
            # Use urllib.parse.quote to encode full_name and section
            encoded_name = quote(full_name)
            encoded_section = quote(section)
            return RedirectResponse(
                url=f"/customer_landing_page?name={encoded_name}&section={encoded_section}",
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
    """Fetch questions from the selected subject's CSV file."""
    csv_file = f"{subject}_questions.csv"
    if not os.path.exists(csv_file):
        return JSONResponse(
            content={"error": f"The file {csv_file} does not exist."}, status_code=404
        )

    try:
        # Read the questions from the CSV
        questions_df = pd.read_csv(csv_file)
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
        result = invoke_chain(
            prompt, session_state['messages'], "gpt-4o-mini", selected_subject, tool_selected
        )

        response_data = {
            "user_query": session_state['user_query'],
            "follow_up_questions": {}  # Initialize as empty
        }

        # Extract follow-up questions from all messages
        if "messages" in result:
            for message in result["messages"]:
                if hasattr(message, 'content'):
                    content = message.content
                    print(f"Message content for follow-up extraction: {content}")  # Debug
                    follow_ups = extract_follow_ups(content)
                    if follow_ups:
                        response_data["follow_up_questions"].update(follow_ups)

        # Handle different intents
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

            response_data.update({
                "query": session_state['generated_query'],
                "tables": tables_html
            })

        elif result["intent"] == "researcher":
            response_data["search_results"] = result.get("search_results", "No results found.")

        elif result["intent"] == "intellidoc":
            response_data["search_results"] = result.get("search_results", "No results found.")

        print(f"Final response data with follow-ups: {response_data}")  # Debug
        return JSONResponse(content=response_data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the prompt: {str(e)}")

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
    section: str = Form(...)
):
    """Handle file uploads for documents."""
    try:
        selected_section = section
        if selected_section not in subject_areas2:
            raise ValueError("Invalid section selected.")

        collection_name = COLLECTION[subject_areas2.index(selected_section)]
        db_path = Chroma_DATABASE[subject_areas2.index(selected_section)]

        logging.info(f"Selected section: {selected_section}, Collection: {collection_name}, DB Path: {db_path}")

        if not files:
            logging.warning("No files uploaded.")
            return JSONResponse({"status": "error", "message": "No files uploaded."})

        # Initialize collection outside the file loop
        collection_instance = init_chroma_collection(db_path, collection_name)
        embed_model = OpenAIEmbedding()
        
        processed_files = []
        failed_files = []

        for file in files:
            try:
                file_content = await file.read()
                file_name = file.filename
                
                # Upload to blob storage
                upload_to_blob_storage(
                    AZURE_STORAGE_CONNECTION_STRING,
                    AZURE_CONTAINER_NAME,
                    collection_name,
                    file_content,
                    file_name,
                )

                # Parse the uploaded file using LlamaParse
                parsed_text = await use_llamaparse(file_content, file_name)

                # Split the parsed document into chunks
                base_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
                nodes = base_splitter.get_nodes_from_documents([Document(text=parsed_text)])

                # Prepare metadata and IDs
                base_file_name = os.path.basename(file_name)
                chunk_ids = [f"{base_file_name}_{i + 1}" for i in range(len(nodes))]
                metadatas = [{"type": base_file_name, "source": file_name} for _ in nodes]

                # Process in batches
                batch_size = 500
                for i in range(0, len(nodes), batch_size):
                    batch_nodes = nodes[i:i + batch_size]
                    batch_metadatas = metadatas[i:i + batch_size]
                    batch_ids = chunk_ids[i:i + batch_size]

                    try:
                        collection_instance.add(
                            documents=[node.text for node in batch_nodes],
                            metadatas=batch_metadatas,
                            ids=batch_ids,
                        )
                        time.sleep(2)  # Rate limit protection
                    except Exception as e:
                        logging.error(f"Error adding batch {i} of file {file_name}: {e}")
                        raise

                processed_files.append(file_name)
                logging.info(f"Successfully processed file: {file_name}")

            except Exception as e:
                logging.error(f"Error processing file {file_name}: {e}")
                failed_files.append(file_name)

        # Final response
        if failed_files:
            return JSONResponse({
                "status": "partial_success",
                "message": f"Processed {len(processed_files)} files successfully, {len(failed_files)} failed",
                "processed_files": processed_files,
                "failed_files": failed_files
            })
        
        return JSONResponse({
            "status": "success",
            "message": f"All {len(processed_files)} files processed successfully",
            "processed_files": processed_files
        })

    except Exception as e:
        logging.error(f"Error in upload_files: {e}")
        return JSONResponse({"status": "error", "message": f"Error during file upload: {str(e)}"})

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
