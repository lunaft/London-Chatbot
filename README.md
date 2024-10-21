# London-Chatbot
ChatBot designed to answer tourists' frequently asked questions in London

This project implements uses the **SentenceTransformer** model to answer questions based on a set of frequently asked questions that tourists traveling to London have. These questions are stored in a JSON file. The chatbot relies on the semantic similarity of the questions to find the most appropriate answer.

## Features

- Answers based on semantic similarity using **SentenceTransformer**.
- Efficient loading of FAQs from a JSON file.
- Text preprocessing to improve answer accuracy.
- Logging of unrecognized questions to improve the model.
- Simple web interface using **Flask**.

## Requirements

Make sure you have Python 3.7 or higher and the following libraries installed:

- Flask
- numpy
- sentence-transformers
- torch

You can install the required dependencies by running:

````bash
pip install -r requirements.txt
````

## Usage

1. Set up your environment: Clone the repository and navigate to the project directory.

2. Run the server: Start the chatbot by executing the Python script (python chatbot.py) This will start the Flask server at http://localhost:5001.

3. Interacting with the ChatBot: Open your browser and go to http://localhost:5001 to access the chatbot interface. You can ask questions and receive answers based on the uploaded FAQs.
