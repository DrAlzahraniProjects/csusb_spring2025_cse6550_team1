#!/bin/bash

# Define container name
CONT_NAME="team1s25-app"

# Stopping current container
echo "Stopping Current Docker container..."
docker stop $CONT_NAME

# Removing current container
echo "Removing Current Docker container..."
docker rm $CONT_NAME

# Remove the current Docker image
echo "Removing Currrent Docker image..."
docker rmi $CONT_NAME

echo "Docker Cleanup of current container is complete."