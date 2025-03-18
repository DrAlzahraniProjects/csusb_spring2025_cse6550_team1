#!/bin/bash

# Sets Container Name and Ports
CONT_NAME="team1s25-app"
PORT_NUM=2501
J_PORTNUM=2511

# Asks for API Key from User
echo "Please enter your API key from GROQ:"
read GROQ_API_KEY

# Detect the host operating system
OS_TYPE=$(uname)

# Inform the user about the OS being used
echo "Detected OS: $OS_TYPE"

# Adjust Docker run command based on the OS
if [[ "$OS_TYPE" == "Linux" ]]; then
    # Linux: No special configuration needed
    DOCKER_RUN_CMD="docker run -d \
        -p $PORT_NUM:$PORT_NUM \
        -p $J_PORTNUM:$J_PORTNUM \
        --name $CONT_NAME \
        -e GROQ_API_KEY=\"$GROQ_API_KEY\" \
        $CONT_NAME"
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS: No special configuration needed
    DOCKER_RUN_CMD="docker run -d \
        -p $PORT_NUM:$PORT_NUM \
        -p $J_PORTNUM:$J_PORTNUM \
        --name $CONT_NAME \
        -e GROQ_API_KEY=\"$GROQ_API_KEY\" \
        $CONT_NAME"
elif [[ "$OS_TYPE" == "Linux" && -n "$WSL_DISTRO_NAME" ]]; then
    # Windows/WSL: No special configuration needed
    DOCKER_RUN_CMD="docker run -d \
        -p $PORT_NUM:$PORT_NUM \
        -p $J_PORTNUM:$J_PORTNUM \
        --name $CONT_NAME \
        -e GROQ_API_KEY=\"$GROQ_API_KEY\" \
        $CONT_NAME"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Building Docker Image
echo "Building Docker image..."
docker build -t $CONT_NAME .

# Running Docker Image
echo "Running Docker container..."
eval $DOCKER_RUN_CMD

# Output where the apps are running
echo "Streamlit is available at: http://localhost:$PORT_NUM/team1s25"
echo "Google Colab is available at: https://colab.research.google.com/drive/1AcIKcovL3VLEsC65BsshNjKJR_WPraxI?usp=sharing"
