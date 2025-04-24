FROM python:3.10-slim

# Install system dependencies including texlive for PDF generation
RUN apt-get update && apt-get install -y \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Set environment variables
ENV PORT=8501

# Expose the port the app runs on
EXPOSE 8501

# Command to run the application
CMD streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
