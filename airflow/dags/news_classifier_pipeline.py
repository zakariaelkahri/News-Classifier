from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 12, 12),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_data():
    print("Loading data from Hugging Face...")
    # Logic to load data

def preprocess_data():
    print("Preprocessing data...")
    # Logic to preprocess

def generate_embeddings():
    print("Generating embeddings...")
    # Logic to generate embeddings

def store_in_chromadb():
    print("Storing in ChromaDB...")
    # Logic to store in ChromaDB

def train_model():
    print("Training model...")
    # Logic to train model

def evaluate_model():
    print("Evaluating model...")
    # Logic to evaluate

with DAG(
    'news_classifier_pipeline',
    default_args=default_args,
    description='A simple pipeline for news classification',
    schedule_interval= "0 16 * * 1",
) as dag:

    t1 = PythonOperator(
        task_id='load_data',
        python_callable=load_data,
    )

    t2 = PythonOperator(
        task_id='preprocess_data',
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id='generate_embeddings',
        python_callable=generate_embeddings,
    )

    t4 = PythonOperator(
        task_id='store_in_chromadb',
        python_callable=store_in_chromadb,
    )

    t5 = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    t6 = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )

    t1 >> t2 >> t3 >> t4 >> t5 >> t6
