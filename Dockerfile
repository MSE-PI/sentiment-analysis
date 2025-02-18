# Base image
FROM python:3.11

# Work directory
WORKDIR /app

# Copy requirements file
COPY ./requirements.txt .
COPY ./requirements-all.txt .

# Install dependencies
RUN pip install --requirement requirements.txt --requirement requirements-all.txt

# Copy sources
COPY src src

# Environment variables
ENV ENVIRONMENT=${ENVIRONMENT}
ENV LOG_LEVEL=${LOG_LEVEL}
ENV ENGINE_URL=${ENGINE_URL}
ENV MAX_TASKS=${MAX_TASKS}
ENV ENGINE_ANNOUNCE_RETRIES=${ENGINE_ANNOUNCE_RETRIES}
ENV ENGINE_ANNOUNCE_RETRY_DELAY=${ENGINE_ANNOUNCE_RETRY_DELAY}

# Exposed ports
EXPOSE 80

# Switch to src directory
WORKDIR "/app/src"

# download spacy model
RUN python3 -m spacy download en_core_web_sm

# download nltk data
RUN python -c "import nltk;nltk.download('averaged_perceptron_tagger')"
RUN python -c "import nltk;nltk.download('punkt')"
RUN python -c "import nltk;nltk.download('vader_lexicon')"
RUN python -c "import nltk;nltk.download('stopwords')"

# Command to run on start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
