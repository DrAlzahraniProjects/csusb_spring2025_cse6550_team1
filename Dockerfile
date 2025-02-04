FROM python:3.10

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install flask jupyter

# Copy the Jupyter notebook into the container
COPY team1-hello-world.ipynb /app/

# Expose Jupyter and Flask ports
EXPOSE 8888 2500

# Command to run Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]