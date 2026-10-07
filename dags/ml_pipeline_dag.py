from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from src.data_preprocessing import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from src.deploy import deploy_model

with DAG(
    dag_id="ml_pipeline_dag",
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:
    preprocess_task = PythonOperator(
        task_id="preprocess_data_task", python_callable=preprocess_data
    )
    train_task = PythonOperator(
        task_id="train_model_task", python_callable=train_model
    )
    evaluate_task = PythonOperator(
        task_id="evaluate_model_task", python_callable=evaluate_model
    )
    deploy_task = PythonOperator(
        task_id="deploy_model_task", python_callable=deploy_model
    )

    preprocess_task >> train_task >> evaluate_task >> deploy_task
