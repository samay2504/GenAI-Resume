#Note that the pretrained model weights are uploaded on "jira ai" as a zip file for DistilBERT.


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import json
import chardet

# Function to detect file encoding using chardet
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_distilbert")
tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_distilbert")

# Function to predict AI or human-written resume and return probabilities
def detect_resume_type(resume_text):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
    ai_score = probabilities[0][1].item() * 100
    human_score = probabilities[0][0].item() * 100
    return human_score, ai_score

# Recursive function to extract text from JSON
def extract_text_from_json(data, text=""):
    if isinstance(data, dict):
        for key, value in data.items():
            text = extract_text_from_json(value, text)
    elif isinstance(data, list):
        for item in data:
            text = extract_text_from_json(item, text)
    elif isinstance(data, (str, int, float)):
        text += str(data) + "\n"
    return text

# Function to load resumes from a .json file and detect type
def detect_from_json(file_path):
    # Detect encoding of the file
    file_encoding = detect_encoding(file_path)
    #print(f"The detected encoding of the JSON file is: {file_encoding}")

    # Open the file with the detected encoding
    with open(file_path, 'r', encoding=file_encoding) as f:
        data = json.load(f)

    # Extract all text from JSON
    resume_text = extract_text_from_json(data)

    # Detect the type of resume and scores
    human_score, ai_score = detect_resume_type(resume_text)
    print(f"Resume Analysis:")
    print(f"  Human-Written Score: {human_score:.2f}%")
    print(f"  AI-Generated Score: {ai_score:.2f}%")

# Example usage
json_file_path = r"path to your resume"
detect_from_json(json_file_path)
