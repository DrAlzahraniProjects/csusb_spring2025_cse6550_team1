FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --upgrade pip && \
pip install \
jupyter \
ipykernel \
streamlit \
huggingface_hub

# Copy the Jupyter notebook into the container
COPY team1-demo-chatbot.ipynb /app/
COPY Demo_chatbot.py /app/

# Expose Jupyter and Flask ports
EXPOSE 8888 2500

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token='TEAM1'"]