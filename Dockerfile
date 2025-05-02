#Use Python image 3.10 as the base builder
FROM python:3.10-slim AS builder

# Copy requirements.txt into image
COPY "requirements.txt" .

RUN python3 -m venv /env \
	&& /env/bin/pip install --upgrade pip \
	&& /env/bin/pip install -r "requirements.txt" --no-cache-dir -U --upgrade-strategy eager
# && echo "    - Installed necessary Python libraries."


# Use official Python image as the base image
FROM python:3.10-slim

#Copy from builder image into base image
COPY --from=builder /env /env

ENV PATH="/env/bin:$PATH"

WORKDIR /app

# Copy your Python code into the Docker container
COPY app.py /app
COPY logo /app/logo
COPY pdf /app/pdf
COPY .streamlit /app/.streamlit


# Install system dependencies for running Apache, Streamlit, and TTS
RUN apt-get update && \
    apt-get install -y \
    apache2 \
    apache2-utils \
    libapache2-mod-proxy-uwsgi \
    libxml2-dev \
    && apt-get upgrade -y \
    && apt-get clean 

# Expose port for Streamlit 
EXPOSE 2501

# Set up the Apache proxy configurations
RUN echo "ProxyPass /team1s25 http://localhost:2501/team1s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "ProxyPassReverse /team1s25 http://localhost:2501/team1s25" >> /etc/apache2/sites-available/000-default.conf && \
    echo "RewriteRule /team1s25/(.*) ws://localhost:2501/team1s25/$1 [P,L]" >> /etc/apache2/sites-available/000-default.conf \
    && a2enmod proxy proxy_http rewrite  # Enable Apache modules for proxy and WebSocket support


# Start Apache and Streamlit using `sh` in the CMD
CMD ["sh", "-c", "apache2ctl start & streamlit run app.py --server.maxUploadSize=10 --server.port=2501 --server.baseUrlPath=/team1s25"]
