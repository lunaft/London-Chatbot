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

## Code Structure

- Imports: Libraries such as Flask for web API creation, sentence_transformers for the language model and torch for CPU/GPU computation are used.

- ChatBot class:
  The ChatBot class encapsulates all chatbot logic, including model initialization, FAQs loading and path configuration in Flask.
  
  The model is initialized using SentenceTransformer, which allows sentence embeddings to be calculated. The default model is sentence-transformers/all-MiniLM-L6-v2, a pretrained model   optimized for semantic matching tasks.

- Loading FAQs:
  Questions and answers are loaded from a JSON file (faq.json). Each question is preprocessed to remove unnecessary characters and converted into an embedding using the model.
  
  The embeddings are generated in batches to improve efficiency, reducing processing time and optimizing memory usage.

- Text Preprocessing:
  Question text is normalized by converting it to lowercase, removing unwanted characters, and extracting keywords.
  
- Similarity Calculation:
  When a question is received, it is converted to an embedding and the similarity to the embeddings of the FAQs questions is calculated using cosine similarity. This allows to identify   the most similar question and the corresponding answer.

  If the similarity exceeds a threshold (similarity_threshold), the answer is returned; otherwise, it is recorded as an unrecognized question.

- Recording Unrecognized Questions:
  Questions that cannot be answered are stored along with their confidence score and timestamp, allowing for further analysis and model improvement.

## Machine Learning Model

sentence-transformers/all-MiniLM-L6-v2 is a Transformers-based language model that has been pre-trained on a wide variety of data. It is designed to generate embeddings that capture sentence semantics, facilitating text comparison in tasks such as question classification and information retrieval.

