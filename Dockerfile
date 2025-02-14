FROM python:3.10-slim
COPY . /app
WORKDIR /app # Set working directory
RUN pip install -r requirements.txt


# Install dependencies
RUN pip install -r requirments.txt


# Expose Jupyter and Flask ports
EXPOSE 2501 2511

# Command to run Jupyter Notebook
CMD ["bash", "-c", "streamlit run app.py --server.port=2501 --server.address=0.0.0.0 & jupyter notebook --ip=0.0.0.0 --port=2511 --no-browser --NotebookApp.token='' --allow-root"]