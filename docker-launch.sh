#!/bin/bash

#Sets Container Name
CONT_NAME="Spring2025Team1-app"
PORT_NUM= 2501
J_PORTNUM= 2511

#Asks for API Key from User
echo "Please enter your API key from GROQ:"
read GROQ_API_KEY


#Building Docker Image
docker build =t $CONT_NAME .

#Running Docker Image
docker run -d -p $PORT_NUM:$PORT_NUM --name $CONT_NAME -e GROQ_API_KEY="$GROQ_API_KEY" $CONT_NAME

echo " Your streamlit is here: http://localhost:2501"
echo " Your Jupyter is here: http://localhost:2511"

