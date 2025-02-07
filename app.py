from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    project_title = 'Team 1: CSUSB Study Podcast Assistant' # Project Title  
    return f'{project_title}\nHello World!' #Added Hello World base comment 

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 2500))
    app.run(debug=True, host='0.0.0.0', port = port)