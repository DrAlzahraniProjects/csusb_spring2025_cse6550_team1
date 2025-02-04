### Step 1: Clone the Repository
Clone the GitHub repository to your local machine:  
```bash
git clone https://github.com/DrAlzahraniProjects/csusb_spring2025_cse6550_team1
```

### Step 2: Navigate to the Repository
Change to the cloned repository directory:  
```bash
cd csusb_spring2025_cse6550_team1
```

### Step 3: Build Docker File
```bash
docker build -t team1-hello-world .
```

### Step 4: Run Docker File
```bash
docker run -p 8888:8888 -p 2500:2500 team1-hello-world
```

### Step 5: Visit Website With Token
```bash
http://127.0.0.1:2500
```

### Step 6: Click on link to notebook in terminal and run.