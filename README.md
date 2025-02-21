# Team1 CSE 6550 Project

## Ensure that Git and Docker are installed
- Git : [Download](https://git-scm.com/downloads)
- Docker : [Download](https://www.docker.com/products/docker-desktop/)
- WSL : [Download](https://learn.microsoft.com/en-us/windows/wsl/install)

## Step 1: Clone the Repository
Clone the GitHub repository to your local machine:  
```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team1
```

## Step 2: Navigate to the Repository
Change to the cloned repository directory:  
```bash
cd csusb_spring2025_cse6550_team1
```

### Step 3: Set Build Script
```bash
chmod +x docker-launch.sh
```

### Step 4: Run Build Script
You will be prompted for an API please enter the key provided in Canvas
```bash
./docker-launch.sh
```

### Step 5: Visit Website For Streamlit Locally
[http://localhost:2501/team1s25](http://localhost:2501/team1s25)
#### Or online at: 
[https://sec.cse.csusb.edu/team1s25](https://sec.cse.csusb.edu/team1s25)

### Step 6: Visit Website For Notebook Locally
[http://localhost:2511/team1s25/jupyter](http://localhost:2511/team1s25/jupyter)
#### Or online at: 
[https://sec.cse.csusb.edu/team1s25/jupyter](https://sec.cse.csusb.edu/team1s25/jupyter)

### Step 7: If you need to stop the container from running and remove the container and image at the same time:
Enable execute permissions for the Docker cleanup script::
```
chmod +x docker-clean.sh
```
### Step 8: Run the script to stop and remove the Docker container and image at the same time:
```
./docker-clean.sh
```