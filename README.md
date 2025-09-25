# Interview Prep App

A Streamlit-based app to practice job interviews with an AI interviewer. Paste or upload a Job Description, chat through mock questions, get feedback and interview preparation plan, and see live token/cost estimates.
- Streamlit app link: [ADD LINK](https://neshsab-ai-interview-app-streamlit-appapp-pyg4sc.streamlit.app/)
	- To use an app, you need OpenAI API key.
<br>

## Table of Contents
- [Introduction](#introduction)
- [How to Navigate](#how-to-navigate-this-repository)
- [Features](#features)
- [How to Run](#how-to-run)
- [Project Structure](#project-structure)
- [Further Improvements](#further-improvements)
- [Get Help](#get-help)
- [Contribution](#contribution)
<br>

## Introduction
This app helps users prepare for job interviews by simulating realistic interview scenarios with an AI interviewer. Users can upload or paste a job description, select the interviewer’s role and persona, as well as questions types, and receive tailored questions, feedback, and a preparation plan. The app is designed for privacy and flexibility, supporting various seniority levels and interview styles.
<br>


## How to Navigate This Repository
- `app/`: Main Streamlit UI and dependencies;
- `jailbreak_experiments/`: Recorded jailbreak experiments in csv file and script to automatically run multiple experiments.
- `generated_roadmaps/`: Few examples of generated images adherent to set requirements.
<br>


## Features
- **Practice Tab:** Chat with an AI interviewer, select persona (Friendly/Neutral/Strict), style (Concise/Detailed), and practice mode (Behavioral/Technical/Situational).
- **Job Description Ingestion:** Paste text or upload a PDF to tailor interview questions.
- **Feedback Tab:** Score your answers, preview improved responses, get next actions, and check summaries with each Interviewer.
- **Plan Tab:** Receive a personalized 7-day preparation plan based on your job description.
- **Voice and/or Speach Mode:** Optionally answer questions by voice and listen to AI responses.
- **Live Pricing:** See token usage and estimated cost by model.
- **Security:** Input validation, PII redaction, and prompt injection checks.
- **Supported Seniority Levels:** Junior, Mid, Senior, Team Lead.
- **Privacy:** Your data and API key remain secure in your browser session.
<br>

## How to Run Locally
1. Clone the repository:
	```bash
	git clone https://github.com/NeshSab/ai_interview_app_streamlit
	```
2. Navigate to the project directory:
	```bash
	cd ai_interview_app_streamlit
	```
3. (Optional) Create and activate a Python virtual environment:
	```bash
	python3 -m venv int_env
	source int_env/bin/activate
	```
4. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
5. Navigate to app folder and run the app:
	```bash
	streamlit run app/app.py
	```
6. Enter your OpenAI API key in the sidebar to start using the app.
<br>

## Project Structure
```
├── app.py                    # Streamlit UI layer
└──  core/
    ├── controller.py         # Session orchestration (text)
    ├── controller_roadmap.py # Session orchestration (image)
    ├── models.py             # Data classes
    ├── interfaces.py         # Protocols
    ├── prompts/			  # Prompt engineering  
    └── services/
    │	├── llm_openai.py     # OpenAI client wrapper
   	│	├── jd_analyzer.py    # JD ingestion and normalization
    │	├── answer_critic.py  # Scoring & improvement
    │	├── security.py       # Input guardrails
    │	├── speech.py         # TTS integration
    │	├── voice.py          # STT integration
    │	└── pricing.py        # Token math & model tests
    └── utils/				  # Utility functions
```
<br>

## Further Improvements
- Better token estimation for image generation, as well ass text-to-speach and speach-to-text operations;
- Pricing tab - split by tokens used by different models and calculate price accordingly; 
- Library of sample job descriptions for quick practice.
- Improved, more user friendly error handling for invalid inputs.
- Support additional seniority levels and interviewer personas.
- Enhance security with session budget limits and stricter prompt filtering.
- Explore image generation promting.
- Add wider selection of models.
- Revisit selected models for background operations (everything, except chatting)...
<br>

## Get Help
If you encounter any issues or have questions about this project, feel free to reach out. Here are the ways you can get help:
- Open an Issue: if you find a bug, experience problems, or have a feature request, please open an issue.
- Email Me: For personal or specific questions, you can reach me via email at: agneska.sablovskaja@gmail.com.
<br>

## Contribution
Contributions are welcome and appreciated! If you'd like to contribute to this project, here’s how you can get involved:
1. Reporting Bugs: if you find a bug, please open an issue and provide detailed information about the problem. Include steps to reproduce the bug, any relevant logs or error messages, and your environment details (OS, versions, etc.).
2. Suggesting Enhancements: if you have ideas for new features or improvements, feel free to open an issue to discuss it. Describe your suggestion in detail and provide context for why it would be useful.