from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import os
from fastapi.middleware.cors import CORSMiddleware
import time
import base64
import pandas as pd
import re
import io
import uuid
import openai
import asyncio
from dataclass import *
import uvicorn
import soundfile as sf
import config
from pydub import AudioSegment

import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from os.path import join, dirname
from ibm_watson import SpeechToTextV1, TextToSpeechV1
from ibm_watson.websocket import RecognizeCallback, AudioSource, SynthesizeCallback
import threading

authenticator = IAMAuthenticator(config.api_key)

service = SpeechToTextV1(authenticator=authenticator)
service.set_service_url('https://api.us-south.speech-to-text.watson.cloud.ibm.com')
model = service.get_model('en-US_BroadbandModel').get_result()

tts_service = TextToSpeechV1(authenticator=authenticator)
tts_service.set_service_url('https://api.us-south.text-to-speech.watson.cloud.ibm.com')

assistant = AssistantV2(
    version='2018-09-20',
    authenticator=authenticator)
assistant.set_service_url('https://api.us-south.assistant.watson.cloud.ibm.com')

# Function to generate a unique identifier for each question
def generate_question_id():
    return str(uuid.uuid4())

# SQL Initiator and query functions

load_dotenv()
db_user = os.getenv("DB_USER")
db_pass = os.getenv("DB_PASS")  # e.g. 'my-db-password'
db_name = os.getenv("DB_NAME")  # e.g. 'my-database'
instance_connection_name = os.getenv("INSTANCE_CONNECTION_NAME")


def connect_with_db() -> sqlalchemy.engine.base.Engine:
    """
    Initializes a connection pool for a Cloud SQL instance of MySQL.

    Uses the Cloud SQL Python Connector package.
    """
    ip_type = IPTypes.PRIVATE if os.environ.get("PRIVATE_IP") else IPTypes.PUBLIC

    connector = Connector(ip_type)

    def getconn() -> pymysql.connections.Connection:
        conn: pymysql.connections.Connection = connector.connect(
            instance_connection_name,
            "pymysql",
            user=db_user,
            password=db_pass,
            db=db_name,
        )
        return conn

    pool = sqlalchemy.create_engine(
        "mysql+pymysql://",
        creator=getconn,
        # ...
    )
    return pool

def push_df_db(df, engine):
    for index, row in df.iterrows():
        email = row['email']
        subject = row['subject']
        weeks = row['weeks']
        card_number = row['card_number']
        front = row['front']
        back = row['back']
                
        with engine.connect() as connection:
            stmt = sqlalchemy.text(
                "INSERT INTO flash_cards (email, subject, weeks, card_number, front, back) VALUES (:email, :subject, :weeks, :card_number, :front, :back)"
            )
            connection.execute(stmt, parameters={"email": email, "subject": subject, "weeks": weeks, "card_number": card_number, "front": front, "back": back})
            connection.commit()


def push_quiz_db(df, engine):
    for index, row in df.iterrows():
        email = row['email']
        question_id = row['question_id']
        subject = row['subject']
        questions = row['questions']
        answers = row['answers']
        option_a = row['option_a']
        option_b = row['option_b']
        option_c = row['option_c']
        option_d = row['option_d']

        
        with engine.connect() as connection:
            stmt = sqlalchemy.text(
                "INSERT INTO quiz_data (question_id, email, subject, questions, answers, option_a, option_b, option_c, option_d) VALUES (:question_id, :email, :subject, :questions, :answers, :option_a, :option_b, :option_c, :option_d)"
            )
            connection.execute(stmt, parameters={"question_id": question_id, "email": email, "subject": subject, "questions": questions, "answers": answers, "option_a": option_a, "option_b": option_b, "option_c": option_c, "option_d": option_d})
            connection.commit()


def submit_quiz_to_db(engine, id, student_response):
    with engine.connect() as connection:
        stmt = sqlalchemy.text(
            "UPDATE quiz_data SET student_response = :student_response WHERE question_id = :question_id"
        )
        connection.execute(stmt, parameters={"student_response": student_response, "question_id": id})
        connection.commit()


def fetch_card_from_db(engine, email, subject):
    with engine.connect() as connection:
        user = connection.execute(
            sqlalchemy.text(
                "SELECT weeks, card_number, front, back FROM flash_cards WHERE email = :email and subject = :subject"
            ),
            {"email":email, "subject": subject}
        ).fetchall()
    return user


def fetch_subjects_from_db(engine, email):
    with engine.connect() as connection:
        user = connection.execute(
            sqlalchemy.text(
                "SELECT DISTINCT subject FROM flash_cards WHERE email = :email"
            ),
            {"email":email}
        ).fetchall()
    return user


def retrieve_student_responses_from_db(engine, email, subject):
    with engine.connect() as connection:
        user = connection.execute(
            sqlalchemy.text(
                "SELECT questions, answers, student_response FROM quiz_data WHERE email = :email and subject = :subject and student_response IS NOT NULL"
            ),
            {"email":email, "subject": subject}
        ).fetchall()
    return user

def fetch_quiz_from_db(engine, email, subject):
    with engine.connect() as connection:
        user = connection.execute(
            sqlalchemy.text(
                "SELECT question_id, questions, option_a, option_b, option_c, option_d FROM quiz_data WHERE email = :email and subject = :subject and student_response IS NULL or student_response = ''"
            ),
            {"email":email, "subject": subject}
        ).fetchall()
    return user


def generate_str(st):
    return st

def generate_weekly_str(st):
    result = []
    for item in st:
        result.extend([item] * 10)
    return result


            
BUCKET_URI = f"development_awarri"
PROJECT_ID = "development-416403"
LOCATION = "us-central1"
vertexai.init(project=PROJECT_ID, location=LOCATION)
MODEL_ID = "meta/llama3-405b-instruct-maas" 

chat_history = []


# Initialize the Google Cloud Storage client
storage_client = storage.Client()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello World"}


def upload_blob(file: UploadFile, destination_blob_name):
    """Uploads a file object directly to the bucket."""
    try:
        bucket = storage_client.bucket(BUCKET_URI)
        blob = bucket.blob(destination_blob_name)

        # Stream file content directly to Google Cloud Storage
        blob.upload_from_file(file.file)

        return f"File {file.filename} uploaded to {destination_blob_name} in bucket {BUCKET_URI}."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")
    
def rag_bucket_items(url_text):
    # print("url_text: ", url_text)

    blobs = storage_client.list_blobs(BUCKET_URI, prefix=url_text, delimiter=None)

    # Note: The call returns a response only when the iterator is consumed.
    names = []
    result = []
    for blob in blobs:
        file = blob.name
        result.append(file.split('/')[-1])
        # print("File: ", file)
        file_bucket_link = f"gs://{BUCKET_URI}/{file}"
        names.append(file_bucket_link)
    # print("names: ", names)
    return result, names

uploading_model = GenerativeModel(
    "gemini-1.5-pro-002",
)


@app.post("/register")
def register(task: RegisterationPage):
    email = task.email
    password = task.password
    disability = task.disability
    language = task.language

    engine = connect_with_db()
    with engine.connect() as connection:
        existing_user = connection.execute(
            sqlalchemy.text(
                "SELECT * FROM user_data WHERE email= :email"
            ),
            {"email":email}
        ).fetchone()

    if existing_user:
        return HTTPException(status_code=500, detail="This email already exists. Please choose another one.")
    
    with engine.connect() as connection:
        stmt = connection.execute(
            sqlalchemy.text(
                "INSERT INTO user_data (email, password, disability, language) VALUES (:email, :password, :disability, :language)"
            ),
            {"email":email, "password":password, "disability":disability, "language":language}
        )
        connection.commit()
        
    task.status  ="Registration Successful!!!"
            
    return task.status


@app.post("/login")
def register(task: LoginPage):
    email = task.email
    password = task.password

    engine = connect_with_db()
    with engine.connect() as connection:
        existing_user = connection.execute(
            sqlalchemy.text(
                "SELECT * FROM user_data WHERE email= :email and password = :password"
            ),
            {"email":email, "password": password}
        ).fetchone()

    if not existing_user:
        return HTTPException(status_code=500, detail="This email or password does not exists. Please Register!!!")
    
        
    task.status  = "Login Successful!!!"
            
    return task.status


@app.post("/upload_docs")
def upload_files(email: str = Form(...), subject: str = Form(...), langauge: str = Form(...), duration: int = Form(...), files: List[UploadFile] = File(...)):
    try:
        results = []
        for file in files:
            destination_blob_name = f"reserve/{email}/{subject}/{file.filename}"
            gs_path = f"gs://{BUCKET_URI}/{destination_blob_name}"
            upload_blob(file, destination_blob_name)
            results.append(gs_path)
        
        prompt = []
        for res in results:
            file_ext = res.split('.')[-1]
            if file_ext == 'pdf':
                # Assuming the PDF content is in base64
                document_part = Part.from_uri(mime_type="application/pdf", uri=res)
            elif file_ext == 'txt':
                # If it's a text file, directly pass the encoded content
                document_part = Part.from_uri(mime_type="text/plain", uri=res)
            prompt.append(document_part)
        GENERATION_PROMPT = f"""
        You are a tutor that helps people learn difficult and also simple things in a very intuitive manner and in their desired language. When you are teaching, you teach in only the language that the tutor specified.
        You normally create flash cards for students that help them to learn concepts in their chosen language.
        When you create the flash cards, you always put in context the learning timeframe of the student, so often you will give students that are studying for long periods of time easy flash cards
        in the first few weeks before they become really difficult. For example, students that want to study for just one week are given very complex cards to study, but students that want to study for three
        weeks are given easy flash cards in the first week, a medium difficulty flash card in the second week and then really difficult flash cards in the last week.
        Depening on the number of weeks, which is {duration}, Generate 10 flash cards based on the info ** I want to learn the basics of physics in English Language and in one week**. If only 1 week was specified, then generate for only one week, but if multiple
        weeks were specified then generate for the number of weeks. Generate the flash cards in {langauge} language
        An example of the format that I want the flash cards to come is:
        W_e_e_k 1: Easier Recall & Key Details**\n\nflash_card 1:\nF_r_o_n_t: What is ÌròyìnSpeech?\nB_a_c_k: ÌròyìnSpeech is a multi-purpose Yorùbá speech corpus created to increase the 
        amount of high-quality, contemporary Yorùbá speech data for text-to-speech (TTS) and automatic 
        speech recognition (ASR) tasks.\n\nflash_card 2:\nF_r_o_n_t: What is the goal of the TTS - 1000 
        African Voices project?\nB_a_c_k: The goal of the TTS - 1000 African Voices project is to advance 
        inclusive multi-speaker multi-accent speech synthesis and increase the representation of African English accents.
        Please let your response be in the exact same format as the above example, and when the chosen language is not English, ensure that there are no English text in your response except for the subseequent W_e_e_ks, flash_cards, F_r_o_n_ts and B_a_c_ks.""" 
        prompt.append(GENERATION_PROMPT)
        
        message = assistant.message(
            config.ASSISTANT_ID,
            config.SESSION_ID,
            input={'text': prompt},
            context={
                'metadata': {
                    'deployment': 'myDeployment'
                }
            }).get_result()

        response = json.dumps(message, indent=2)
        
        # Regex patterns
        week_pattern = r'W_e_e_k (\d+):'
        flash_card_pattern = r'flash_card (\d+):'
        front_pattern = r'F_r_o_n_t: (.*?)\n'
        back_pattern = r'B_a_c_k: (.*?)\n'

        # Extracting lists
        weekly_data = re.findall(week_pattern, response)
        flash_cards = re.findall(flash_card_pattern, response)
        fronts = re.findall(front_pattern, response)
        backs = re.findall(back_pattern, response)
        email = [generate_str(email) for _ in range(len(flash_cards))]
        subject = [generate_str(subject) for _ in range(len(flash_cards))]
        weeks = generate_weekly_str(weekly_data)
        # Print results
        print(len(weeks))
        print(len(flash_cards))
        print(len(fronts))
        print(len(backs))
        
        data = {
            'email': email,
            'subject': subject,
            'weeks': weeks,
            'card_number': flash_cards,
            'front': fronts,
            'back': backs
        }
        # print("Data: ", data)
        
        # print("Dict: ", data)
        df = pd.DataFrame(data)
        engine = connect_with_db()
        push_df_db(df, engine)
            
        return {"message": "DONE"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.post("/upload_audios")
def upload_audios(task : AudioUploader):
    try:
        audio_bytes = io.BytesIO(base64.b64decode(task.audio_base64_content.split(',')[1]))
        transcription = json.dumps(service.recognize(audio=audio_file,content_type='audio/wav',timestamps=True,word_confidence=True).get_result(),indent=2)
        audio = AudioSegment.from_file(audio_bytes, format="webm")

        GENERATION_PROMPT = f"""
        You are a tutor that helps people learn difficult and also simple things by creating flash cards for them in their desired language. When you are teaching, you teach in only the language that the student specified.
        You create 10 flash cards for each week depending on the time frame specified by the student.
        The student has said: {transcription}.
        
        An example of the format that I want the flash cards to come is:
        W_e_e_k 1: Easier Recall & Key Details**\n\nflash_card 1:\nF_r_o_n_t: What is ÌròyìnSpeech?\nB_a_c_k: ÌròyìnSpeech is a multi-purpose Yorùbá speech corpus created to increase the 
        amount of high-quality, contemporary Yorùbá speech data for text-to-speech (TTS) and automatic 
        speech recognition (ASR) tasks.\n\nflash_card 2:\nF_r_o_n_t: What is the goal of the TTS - 1000 
        African Voices project?\nB_a_c_k: The goal of the TTS - 1000 African Voices project is to advance 
        inclusive multi-speaker multi-accent speech synthesis and increase the representation of African English accents.
        Please let your response be in the exact same format as the above example, and when the chosen language is not English, ensure that there are no English text in your response except for the subseequent W_e_e_ks, flash_cards, F_r_o_n_ts and B_a_c_ks.""" 
        
        message = assistant.message(
            config.ASSISTANT_ID,
            config.SESSION_ID,
            input={'text': GENERATION_PROMPT},
            context={
                'metadata': {
                    'deployment': 'myDeployment'
                }
            }).get_result()

        response = json.dumps(message, indent=2)
        
        print("response: ", response)
        # Regex patterns
        week_pattern = r'W_e_e_k (\d+):'
        flash_card_pattern = r'flash_card (\d+):'
        front_pattern = r'F_r_o_n_t: (.*?)\n'
        back_pattern = r'B_a_c_k: (.*?)\n'

        # Extracting lists
        weekly_data = re.findall(week_pattern, response)
        flash_cards = re.findall(flash_card_pattern, response)
        fronts = re.findall(front_pattern, response)
        backs = re.findall(back_pattern, response)
        email = [generate_str("moses@awarri.com") for _ in range(len(flash_cards))]
        subject = [generate_str("astronomy") for _ in range(len(flash_cards))]
        weeks = generate_weekly_str(weekly_data)
        # Print results
        print(len(weeks))
        print(len(flash_cards))
        print(len(fronts))
        print(len(backs))
        
        data = {
            'email': email,
            'subject': subject,
            'weeks': weeks,
            'card_number': flash_cards,
            'front': fronts,
            'back': backs
        }
        # print("Data: ", data)
        
        # print("Dict: ", data)
        df = pd.DataFrame(data)
        engine = connect_with_db()
        push_df_db(df, engine)
            
        return {"message": "DONE"}
    
    
        # return {"message": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

################

@app.get("/get_flash_cards")
def get_flash_cards(email: str, subject: str):
    try:
        
        engine = connect_with_db()
        cards = fetch_card_from_db(engine, email, subject)
        
        # Format cards into a list of dictionaries for JSON response
        formatted_cards = [
            {"week": str(card[0]), "id": card[1], "question": card[2], "answer": card[3]}
            for card in cards
        ]
        
        return {"cards": formatted_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_audio_cards")
def get_audio_cards(email: str, subject: str):
    try:

        # Note: the voice can also be specified by name.
        # Names of voices can be retrieved with client.list_voices().
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Studio-O",
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=1)
    
        engine = connect_with_db()
        cards = fetch_card_from_db(engine, email, subject)
        
        # Format cards into a list of dictionaries for JSON response
        formatted_cards = [
            {"week": str(card[0]), "id": card[1], "question": tts_service.synthesize(card[2], accept='audio/wav',voice="en-US_AllisonVoice").get_result(), "answer": tts_service.synthesize(card[3], accept='audio/wav',voice="en-US_AllisonVoice").get_result()}
            for card in cards
        ]
        
        return {"cards": formatted_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/get_subjects")
def get_subjects(email: str):
    try:        
        engine = connect_with_db()
        # print("Email: ", email)
        cards = fetch_subjects_from_db(engine, email)
        
        # Format cards into a list of dictionaries for JSON response
        formatted_cards = [
            card[0]
            for card in cards
        ]
        # print("formatted_cards: ", formatted_cards)
        
        return {"cards": formatted_cards}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



@app.post("/create_quizes")
def create_quizes(email: str = Form(...), subject: str = Form(...)):
    try:
        
        engine = connect_with_db()
        cards = fetch_card_from_db(engine, email, subject)#[0][2]
        lang = cards[0][2]
        print("lang: ", lang)
        # print("Cards: ", cards)
        weeks = int(len(cards)/ 10)
        print("Number of weeks: ", weeks)

        CHAT_COMPLETIONS_PROMPT = f"""
        You are an expert in several langauges, I want you to answer in one word what language is this text:{lang}."""
        message = assistant.message(
            config.ASSISTANT_ID,
            config.SESSION_ID,
            input={'text': CHAT_COMPLETIONS_PROMPT},
            context={
                'metadata': {
                    'deployment': 'myDeployment'
                }
            }).get_result()

        response = json.dumps(message, indent=2)
        
        _, names = rag_bucket_items(f"reserve/{email}/{subject}")
        
        prompt = []
        for res in names:
            file_ext = res.split('.')[-1]
            if file_ext == 'pdf':
                # Assuming the PDF content is in base64
                document_part = Part.from_uri(mime_type="application/pdf", uri=res)
            elif file_ext == 'txt':
                # If it's a text file, directly pass the encoded content
                document_part = Part.from_uri(mime_type="text/plain", uri=res)
            prompt.append(document_part)
        GENERATION_PROMPT = f"""
        You are a tutor that helps people learn difficult and also simple things in a very intuitive manner and in their desired language by giving them quizes to take. When you are give quizes, you do so in only the language that the tutor specified.
        For each quiz question that you create, you also provide 4 multiple choice options for the student to pick as well as the correct answer to the question. You do all of this in the student's chosen language. 
        When you create the quizes, you always put in context the learning timeframe of the student, so you will often give students that are studying for long periods of time easy quizes
        in the first few weeks before they become really difficult. For example, students that want to study for just one week are given very complex quizes, but students that want to study for three
        weeks are given easy quizes in the first week, medium difficulty quizes in the second week and then really difficult quizes in the last week.
        Depening on the number of weeks, which is {weeks}, Generate 10 quizes from the attached documents for each week specified. If only 1 week was specified, then generate for only one week, but if multiple
        weeks were specified then generate for the number of weeks. The student has specified that they would like the quize in {langauge} language.
        
        An example of the format that I want the quizes to come is:
        W_e_e_k 1: Easier Recall & Key Details**\n\q_u_e_s_t_i_o_n: What is ÌròyìnSpeech?\o_p_t_i_o_n_a: ÌròyìnSpeech is a multi-purpose ... Lack of funding for research\n * o_p_t_i_o_n_b) Limited access to internet connectivity\n * o_p_t_i_o_n_c) Scarcity of data for African languages\n * o_p_t_i_o_n_d) Lack of skilled engineers\n* A_n_s_w_e_r: Scarcity of data for African languages)
        **\n\q_u_e_s_t_i_o_n: What is ÌròyìnSpeech?\o_p_t_i_o_n_a: ÌròyìnSpeech is a multi-purpose ... Lack of funding for research\n * o_p_t_i_o_n_b) Limited access to internet connectivity\n * o_p_t_i_o_n_c) Scarcity of data for African languages\n * o_p_t_i_o_n_d) Lack of skilled engineers\n* A_n_s_w_e_r: Scarcity of data for African languages)
        W_e_e_k 2: Easier Recall & Key Details**\n\q_u_e_s_t_i_o_n: What is ÌròyìnSpeech?\o_p_t_i_o_n_a: ÌròyìnSpeech is a multi-purpose ... Lack of funding for research\n * o_p_t_i_o_n_b) Limited access to internet connectivity\n * o_p_t_i_o_n_c) Scarcity of data for African languages\n * o_p_t_i_o_n_d) Lack of skilled engineers\n* A_n_s_w_e_r: Scarcity of data for African languages)
        **\n\q_u_e_s_t_i_o_n: What is ÌròyìnSpeech?\o_p_t_i_o_n_a: ÌròyìnSpeech is a multi-purpose ... Lack of funding for research\n * o_p_t_i_o_n_b) Limited access to internet connectivity\n * o_p_t_i_o_n_c) Scarcity of data for African languages\n * o_p_t_i_o_n_d) Lack of skilled engineers\n* A_n_s_w_e_r: Scarcity of data for African languages)
        Please let your response be in the exact same format as the above example, and when the chosen language is not English, ensure that there are no English text in your response except for the subseequent W_e_e_ks, q_u_e_s_t_i_o_ns, A_n_s_w_e_r and o_p_t_i_o_ns.
        """
     
        prompt.append(GENERATION_PROMPT)
        message = assistant.message(
            config.ASSISTANT_ID,
            config.SESSION_ID,
            input={'text': prompt},
            context={
                'metadata': {
                    'deployment': 'myDeployment'
                }
            }).get_result()

        response = json.dumps(message, indent=2)
            
        # Regex patterns to extract different sections
        week_pattern = r"\*\*W_e_e_k (\d+):.*?\*\*"
        question_pattern = r"\\q_u_e_s_t_i_o_n: (.+?)\n"
        answer_pattern = r"\\A_n_s_w_e_r: (.+?)\n"
        option_a_pattern = r"\\o_p_t_i_o_n_a: (.+?)\n"
        option_b_pattern = r"\\o_p_t_i_o_n_b: (.+?)\n"
        option_c_pattern = r"\\o_p_t_i_o_n_c: (.+?)\n"
        option_d_pattern = r"\\o_p_t_i_o_n_d: (.+?)\n"

        # Extract data
        weekly_data = re.findall(week_pattern, response)
        questions = re.findall(question_pattern, response)
        answers = re.findall(answer_pattern, response)
        options_a = re.findall(option_a_pattern, response)
        options_b = re.findall(option_b_pattern, response)
        options_c = re.findall(option_c_pattern, response)
        options_d = re.findall(option_d_pattern, response)
        email = [generate_str(email) for _ in range(len(answers))]
        subject = [generate_str(subject) for _ in range(len(answers))]
        weeks = generate_weekly_str(weekly_data)
        question_id = [generate_question_id() for _ in range(len(answers))]
        # Print results
        print(len(weeks))
        print(len(questions))
        print(len(options_a))
        print(len(options_b))
        print(len(options_c))
        print(len(options_d))
        print(len(answers))
            
        data = {
            'question_id': question_id,
            'email': email,
            'subject': subject,
            'questions': questions,
            'answers': answers,
            'option_a': options_a,
            'option_b': options_b,
            'option_c': options_c,
            'option_d': options_d
        }
        # print("Data: ", data)
        
        # # print("Dict: ", data)
        df = pd.DataFrame(data)
        
        engine = connect_with_db()
        push_quiz_db(df, engine)

        return {"message": response}
        # return {"message": "DONE"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/evaluate_performance")
def evaluate_performance(email: str, subject: str):
    try:
        engine = connect_with_db()
        # questions, answers, student_response = retrieve_student_responses_from_db(engine, email, subject)
        data = retrieve_student_responses_from_db(engine, email, subject)
        # print("data: ", data)
        model_feed = []
        output = []
        for d in data:
            question = d[0]
            answers = d[1]
            student_response = d[2]
            model_feed.append({"question": question, "correct answer": answers, "student response": student_response})
        print("model_feed: ", model_feed)
            
        GENERATION_PROMPT = f"""
        You are a teacher who gives students flash cards and then quizes to do. You can find the students responses to the quizes here: {model_feed}
        I want you to give performance feedback to the student based on their responses to the quizes
        An example of the format that I want is:
        O_v_e_r_a_l_l: It seems you're grasping so ..... \n Feedback_Question 1: While honesty is appreciated ...... \n Feedback_Question 2: You're on the right track! However ...
        """

        message = assistant.message(
            config.ASSISTANT_ID,
            config.SESSION_ID,
            input={'text': GENERATION_PROMPT},
            context={
                'metadata': {
                    'deployment': 'myDeployment'
                }
            }).get_result()

        response = json.dumps(message, indent=2)

        # Regex patterns to extract O_v_e_r_a_l_l and Feedback_Questions
        overall_pattern = r"O_v_e_r_a_l_l: (.+?)\n\n"
        feedback_question_pattern = r"Feedback_Question (\d+): (.+?)(?=\n\n|$)"

        # Extract the overall feedback
        overall_feedback = re.search(overall_pattern, response).group(1)
        output.append({"Overall": overall_feedback})
        # Extract the feedback questions
        feedback_questions = re.findall(feedback_question_pattern, response)

        
        # Print the extracted data
        # print(f"O_v_e_r_a_l_l: {overall_feedback}")
        for question_num, question_text in feedback_questions:
            # print(f"Feedback_Question {question_num}: {question_text}")
            output.append({f"Feedback_Question {question_num}": question_text})
        
        return {"cards": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
