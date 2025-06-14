from flask import Flask, render_template, request, jsonify
import requests
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/automate', methods=['POST'])
def automate_task():
    task = request.form.get('task')
    
    # Simulated "automation"
    result = run_automation(task)
    
    return jsonify({'status': 'success', 'output': result})

def run_automation(task_name):
    # Simulate REST call or logic
    simulated_api_response = {"message": f"Task '{task_name}' completed at {time.ctime()}"}
    return simulated_api_response

if __name__ == '__main__':
    app.run(debug=True)
