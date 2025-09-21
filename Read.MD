<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Medical Diagnosis System</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
            color: #1a1a1a;
        }
        code {
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            background-color: #f0f0f0;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        pre {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .container {
            background-color: #ffffff;
            padding: 25px;
            border: 1px solid #ddd;
            border-radius: 8px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            font-size: 0.8em;
            font-weight: bold;
            border-radius: 4px;
            color: #fff;
        }
        .badge-model {
            background-color: #007bff;
        }
        .badge-tech {
            background-color: #28a745;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1> Multi-Agent Medical Diagnosis System</h1>

        <img src="My First Board.jpg" alt="My Project Logo" width="200" height="200">

        <p>This project demonstrates a multi-agent AI system designed to analyze medical reports and provide a preliminary differential diagnosis. It leverages a team of specialized AI agents, each powered by a language model, to assess a patient's case from multiple clinical perspectives. The final output is a synthesized report from a multidisciplinary team, identifying the most likely health issues.</p>
        
        <p>
            <span class="badge badge-model">Model: Google Flan-T5</span>
            <span class="badge badge-tech">Framework: LangChain</span>
            <span class="badge badge-tech">Technology: Python</span>
        </p>

        <h2> Features</h2>
        <ul>
            <li><strong>Multi-Agent Architecture:</strong> Employs six specialist agents and one coordinating agent to simulate a multidisciplinary medical review.</li>
            <li><strong>Parallel Processing:</strong> Specialist agents run concurrently using threading for efficient analysis.</li>
            <li><strong>Specialized Roles:</strong> Each agent is prompted to act as a specific medical specialist (e.g., Cardiologist, Oncologist), focusing on relevant data.</li>
            <li><strong>Synthesized Output:</strong> A final "MultidisciplinaryTeam" agent consolidates all specialist findings into a concise list of the three most likely health issues with reasoning.</li>
            <li><strong>Extensible:</strong> The framework can be easily extended to include more specialized agents.</li>
        </ul>

        <h2> The AI Agents</h2>
        <p>A medical report is passed to all specialist agents, who analyze it in parallel. Their findings are then passed to a final team agent for synthesis.</p>

        <h3>Specialist Agents</h3>
        <ol>
            <li><strong>Cardiologist Agent:</strong> Focuses on heart-related issues like arrhythmias and ischemia.</li>
            <li><strong>Psychologist Agent:</strong> Identifies potential psychological conditions such as anxiety and stress disorders.</li>
            <li><strong>Pulmonologist Agent:</strong> Assesses respiratory causes for symptoms like asthma or COPD.</li>
            <li><strong>Neurologist Agent:</strong> Looks for signs of neurological disorders (e.g., headaches, neuropathy).</li>
            <li><strong>Gastroenterologist Agent:</strong> Focuses on digestive system issues like GERD or IBS.</li>
            <li><strong>Oncologist Agent:</strong> Scans for any signs that might indicate a malignancy.</li>
        </ol>

        <h3>Coordinating Agent</h3>
        <ul>
            <li><strong>MultidisciplinaryTeam Agent:</strong> Reviews all specialist reports, synthesizes the information, and determines the top three most probable health concerns.</li>
        </ul>

        <h2> Repository Structure</h2>
        <pre><code>.
├── Medical Reports/     # Contains synthetic medical report samples
├── results/             # Stores the final diagnosis outputs
├── main.py              # Main script to run the agent system
└── requirements.txt     # Project dependencies
</code></pre>

        <h2> Quickstart Guide</h2>
        
        <h3>1. Clone the Repository</h3>
        <pre><code>git clone https://github.com/your-username/Multi-Agent-Medical-Diagnosis.git
cd Multi-Agent-Medical-Diagnosis</code></pre>

        <h3>2. Set Up Virtual Environment and Install Dependencies</h3>
        <pre><code># Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install the required packages
pip install -r requirements.txt</code></pre>
        <p><strong>Note:</strong> Since this project uses a public Hugging Face model (<code>google/flan-t5-small</code>), no API keys are required.</p>

        <h3>3. Prepare a Medical Report</h3>
        <p>Place your medical report as a <code>.txt</code> file in the project's root directory or update the file path in <code>main.py</code>. Example reports are available in the <code>Medical Reports/</code> folder.</p>

        <h3>4. Run the System</h3>
        <pre><code>python main.py</code></pre>
        <p>The script will run all agents, print the individual and final reports to the console, and save the final diagnosis in the <code>results/</code> directory.</p>
    </div>
</body>
</html>