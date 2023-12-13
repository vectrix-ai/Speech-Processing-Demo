# Use an official Python runtime as the base image
FROM mcr.microsoft.com/devcontainers/python:0-3.11

WORKDIR /streamlit

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

COPY . .

# Install the dependencies
RUN pip install --no-cache-dir -r streamlit_requirements.txt

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501"]