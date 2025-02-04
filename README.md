### Step 1: Clone the Repository
Clone the GitHub repository to your local machine:  
```bash
git clone https://github.com/DrAlzahrani/csusb_spring2025_cse6550_team1.git
```

### Step 2: Navigate to the Repository
Change to the cloned repository directory:  
```bash
cd csusb_spring2025_cse6550_team1
```

### Step 3: Build Docker File
```bash
docker build -t team1-hello-world:latest .
```

### Step 4: Run Docker File
```bash
docker docker run -d -p 2500:2500 team1-hello-world
```