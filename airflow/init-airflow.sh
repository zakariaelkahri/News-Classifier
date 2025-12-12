#!/bin/bash
set -e

# Initialize the database
airflow db migrate

# Create an admin user if it doesn't exist
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin || true

# Start the scheduler in the background
airflow scheduler &

# Start the webserver
exec airflow webserver
