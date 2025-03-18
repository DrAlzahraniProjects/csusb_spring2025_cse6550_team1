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

# PulseAudio configuration
if [[ "$OS_TYPE" == "Linux" ]]; then
    PULSE_SOCKET="/run/user/$(id -u)/pulse/native"
    PULSE_ENV="-v /run/user/$(id -u)/pulse:/run/user/$(id -u)/pulse -e PULSE_SERVER=unix:$PULSE_SOCKET"
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS configuration
    PULSE_ENV="-e PULSE_SERVER=host.docker.internal"
    echo "Starting PulseAudio on macOS..."
    pulseaudio --load="module-native-protocol-tcp" --exit-idle-time=-1 &
elif [[ "$OS_TYPE" == "Linux" && -n "$WSL_DISTRO_NAME" ]]; then
    # WSL configuration
    PULSE_SERVER_IP=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
    PULSE_ENV="-e PULSE_SERVER=$PULSE_SERVER_IP"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Building Docker Image
docker build -t $CONT_NAME .

# Running Docker Image with PulseAudio support
docker run -d \
    -p $PORT_NUM:$PORT_NUM \
    -p $J_PORTNUM:$J_PORTNUM \
    --name $CONT_NAME \
    $PULSE_ENV \
    -e GROQ_API_KEY="$GROQ_API_KEY" \
    $CONT_NAME

# Output where the apps are running
echo "Streamlit is available at: http://localhost:$PORT_NUM/team1s25"
echo "Google Colab is available at: https://colab.research.google.com/drive/1AcIKcovL3VLEsC65BsshNjKJR_WPraxI?usp=sharing"
