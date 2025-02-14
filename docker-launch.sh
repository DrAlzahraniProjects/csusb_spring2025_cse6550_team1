#!/bin/bash

# Sets Container Name and Ports
CONT_NAME="spring2025team1-app"
PORT_NUM=2501
J_PORTNUM=2511

# Asks for API Key from User
echo "Please enter your API key from GROQ:"
read GROQ_API_KEY

# Building Docker Image
docker build -t $CONT_NAME .

# Running Docker Image
docker run -d -p $PORT_NUM:$PORT_NUM -p $J_PORTNUM:$J_PORTNUM --name $CONT_NAME -e GROQ_API_KEY="$GROQ_API_KEY" $CONT_NAME

# Output where the apps are running
echo "Streamlit is available at: http://localhost:$PORT_NUM/team1s25"
echo "Jupyter Notebook is available at: http://localhost:$J_PORTNUM/team1s25/jupyter"

