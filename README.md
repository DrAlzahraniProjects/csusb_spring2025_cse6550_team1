### CSUSB Podcast Bot

## Prerequisites

Before you begin, ensure you have the following:

1. **Git**: [Install Git](https://git-scm.com/) from the official website.
2. **Docker**: [Install Docker](https://www.docker.com) from the official website.
3. **Linux/MacOS**: No setup needed.
4. **Windows**: Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker's WSL integration by following [this guide](https://docs.docker.com/desktop/windows/wsl/).

---
### Step 1: Remove the existing code directory completely

Because the local repository can't been updated correctly with the script, you need to remove the directory currently on the system.

```bash
rm -rf csusb_spring2025_cse6550_team1 
```
### Step 2: Clone the Repository

Clone the GitHub repository to your local machine and navigate to that directory:

```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team1 && cd csusb_spring2025_cse6550_team1
```

### Step 3: Set & Run Build Script (enter your Groq API Key when prompted):

Run the setup script to build and start the Docker container:

```bash
chmod +x docker-launch.sh && ./docker-launch.sh
```

### Step 4: Access the Chatbot

For local:

- Once the container starts, Open your browser at http://localhost:2501/team1s25

---

### Hosted on CSE department web server

For Streamlit:

Open link at https://sec.cse.csusb.edu/team1s25

## Google Colab Notebook  

We are using a Google Colab notebook for easy access and execution.

[Open in Colab] https://colab.research.google.com/drive/1AcIKcovL3VLEsC65BsshNjKJR_WPraxI?usp=sharing
