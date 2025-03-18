# Use official Python image as the base image
FROM python:3.10-slim

# Install system dependencies for running Apache, Streamlit, and TTS
RUN apt-get update && \
    apt-get install -y \
    apache2 \
    apache2-utils \
    ffmpeg \ 
    && apt-get clean

# Install the required Apache modules for proxy and WebSocket support
RUN apt-get update && \
    apt-get install -y \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    libxslt-dev \
    && apt-get clean

# Set up the work directory
WORKDIR /app

# Copy your requirements.txt into the Docker container
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Python code into the Docker container
COPY app.py /app
COPY logo /app/logo
COPY .streamlit /app/.streamlit

# Expose port for Streamlit
EXPOSE 2501

# Set up the Apache proxy configurations
RUN echo "ProxyPass /team1s25 http://localhost:2501/team1s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team1s25 http://localhost:2501/team1s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team1s25/(.*) ws://localhost:2501/team1s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf

# Enable Apache modules for proxy and WebSocket support
RUN a2enmod proxy proxy_http rewrite

# Start Apache and Streamlit using `sh` in the CMD
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.maxUploadSize=10 --server.port=2501 --server.baseUrlPath=/team1s25"]
