import os
import re
import json
from collections import Counter
import spacy
import PyPDF2
import requests
from bs4 import BeautifulSoup

# Initialize spaCy model for named entity recognition
nlp = spacy.load("en_core_web_sm")

# Input folder containing resumes
input_folder = "D:/Lusak.tech/Sample Resume"
output_folder = "D:/Lusak.tech/Dataset"
os.makedirs(output_folder, exist_ok=True)

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text += page.extract_text()
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
    return text

# Function to extract candidate name using spaCy
def extract_name(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    return None  # Return None if no name is found

# Function to sanitize file names
def sanitize_filename(name: str) -> str:
    """Sanitize filenames to remove invalid characters and limit length."""
    sanitized_name = re.sub(r'[/*?:"<>|]', '_', name)
    sanitized_name = sanitized_name.replace('\n', ' ').strip()
    return sanitized_name[:50] if len(sanitized_name) > 50 else sanitized_name

# Possible headings for each section
SECTION_HEADINGS = {
    "education": ["education", "academic background", "academic qualifications"],
    "work": ["work experience", "professional experience", "employment history", "work history"],
    "projects": ["projects", "selected projects", "key projects"],
    "skills": ["skills", "technical skills", "key competencies"],
    "leadership": ["leadership", "activities", "extracurricular activities"],
}

def find_section_heading(text, section_name):
    """Identify the most likely heading for a given section."""
    possible_headings = SECTION_HEADINGS.get(section_name, [])
    lines = text.split("\n")
    for line in lines:
        if any(heading.lower() in line.lower() for heading in possible_headings):
            return line
    return None

def extract_section_content(text, heading, stop_phrases):
    """Extract content for a section starting from a heading until a stop phrase."""
    if not heading:
        return ""
    start = text.find(heading)
    if start == -1:
        return ""
    section_text = text[start:]
    for stop_phrase in stop_phrases:
        stop = section_text.find(stop_phrase)
        if stop != -1:
            return section_text[:stop].strip()
    return section_text.strip()

def parse_resume_content(text):
    parsed_data = {
        "basics": {},
        "education": [],
        "work": [],
        "projects": [],
        "skills": {},
        "leadership": [],
    }

    # Extract basics (name, email, phone)
    parsed_data["basics"]["name"] = extract_name(text) or "Unknown"  # Handle missing name
    email_match = re.search(r'[\w.-]+@[\w.-]+', text)
    phone_match = re.search(r'(\+\d{1,2}\s?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    parsed_data["basics"]["email"] = email_match.group(0) if email_match else ""
    parsed_data["basics"]["phone"] = phone_match.group(0) if phone_match else ""

    # Iterate over sections and extract content dynamically
    for section, possible_headings in SECTION_HEADINGS.items():
        heading = find_section_heading(text, section)
        stop_phrases = [h for h in SECTION_HEADINGS.keys() if h != section]
        parsed_data[section] = extract_section_content(text, heading, stop_phrases)

    return parsed_data

# Function to process resumes in the input folder
def process_resumes():
    processed_names = Counter()

    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(file_path):
            continue

        print(f"Processing: {file_name}")

        # Extract text based on file type
        if file_name.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            print(f"Unsupported file format: {file_name}")
            continue

        if not text.strip():
            print(f"No text found in {file_name}")
            continue

        # Parse structured content
        parsed_data = parse_resume_content(text)

        # Handle name-based file naming or fallback to "unknown" names
        name = parsed_data["basics"].get("name", "Unknown")
        processed_names[name] += 1
        if name == "Unknown":
            unique_name = f"unknown {processed_names[name]}"
        else:
            unique_name = f"{name}_{processed_names[name]}" if processed_names[name] > 1 else name

        sanitized_name = sanitize_filename(unique_name)

        # Save parsed information to JSON
        output_file = os.path.join(output_folder, f"{sanitized_name}.json")
        with open(output_file, mode='w', encoding='utf-8') as json_file:
            json.dump(parsed_data, json_file, ensure_ascii=False, indent=4)

        print(f"Saved extracted data to ./{output_folder}/{sanitized_name}.json")
        print("--------------------------------------------------")

# Function to scrape resumes from a URL
def scrape_resumes_from_url(url):
    try:
        # Fetch the URL content
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all relevant resume links or content
        resume_content = soup.find_all('div', class_='Resume')  # Example class, adjust based on your target URL

        # Extract relevant data from the found content
        resumes_data = []
        for resume in resume_content:
            name = resume.find('h2').text.strip()  # Adjust this as needed
            work_experience = resume.find('ul', class_='work').text.strip()

            resumes_data.append({
                'name': name,
                'work_experience': work_experience
            })

        return resumes_data

    except requests.exceptions.RequestException as e:
        print(f"Error scraping URL: {e}")
        return []

if __name__ == "__main__":
    process_resumes()

    # Example of web scraping
    url = "https://example.com/resumes"  # Replace it
    scraped_data = scrape_resumes_from_url(url)
    print(scraped_data)
