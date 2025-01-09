import os
import re
import json
import spacy
import PyPDF2
import logging
from typing import Dict, Any, List
from pathlib import Path
from docx import Document
from datetime import datetime
from groq import Groq

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_lg")
groq_client = Groq(api_key="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")


class ResumeParser:
    def __init__(self, use_groq: bool = True):
        self.use_groq = use_groq
        self.section_patterns = {
            "education": [r"(?i)education(?:al)?(?: background)?", r"(?i)academic (?:background|history|qualification)",
                          r"(?i)qualification", r"(?i)degrees?"],
            "work": [r"(?i)work(?: experience| history)?", r"(?i)professional(?: experience| background)",
                     r"(?i)employment(?: history)?", r"(?i)career(?: history)?"],
            "projects": [r"(?i)projects?(?:\s+completed)?", r"(?i)case studies", r"(?i)portfolio",
                         r"(?i)implementations?"],
            "skills": [r"(?i)skills?(?: summary| set)?", r"(?i)technical(?: skills| proficiencies)?",
                       r"(?i)core competencies", r"(?i)expertise"],
            "languages": [r"(?i)languages?(?:\s+proficiency)?", r"(?i)linguistic abilities"],
            "certifications": [r"(?i)certifications?", r"(?i)professional certifications?", r"(?i)licenses?",
                               r"(?i)accreditations?"],
            "awards": [r"(?i)awards?(?:\s+and achievements)?", r"(?i)achievements?", r"(?i)honors?",
                       r"(?i)recognitions?"]
        }

        self.groq_prompts = {
            "section_identification": """
            Analyze the following resume text and identify the main sections.
            For each section, provide:
            1. The section name
            2. The starting line number
            3. The ending line number
            4. Confidence score (0-1)

            Resume text:
            {text}
            """,

            "entity_extraction": """
            Extract the following information from the resume text:
            - Full Name
            - Email Address
            - Phone Number
            - LinkedIn Profile
            - GitHub Profile
            - Location
            - Current Position

            Resume text:
            {text}
            """,

            "section_summary": """
            Summarize the following section from a resume:
            Section name: {section_name}

            Section content:
            {content}

            Provide a structured summary including key points and relevant details.
            """
        }

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = []
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    cleaned_text = re.sub(r'\s{2,}', '\n', page_text)
                    text.append(cleaned_text)
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""

    def extract_text_from_docx(self, file_path: str) -> str:
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text.strip())
            return '\n'.join(text)
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {str(e)}")
            return ""

    def get_groq_section_boundaries(self, text: str) -> Dict[str, Dict[str, Any]]:
        try:
            if not self.use_groq:
                raise Exception("Groq integration disabled")

            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": self.groq_prompts["section_identification"].format(text=text)}],
                model="mixtral-8x7b-32768",
                temperature=0.1,
                max_tokens=2000
            )

            sections = {}
            response_text = response.choices[0].message.content

            for line in response_text.split('\n'):
                if ':' in line:
                    section_name, details = line.split(':', 1)
                    section_name = section_name.strip().lower()
                    if section_name in self.section_patterns:
                        start, end, confidence = map(int, re.findall(r'\d+', details))
                        sections[section_name] = {
                            "start": start,
                            "end": end,
                            "confidence": confidence / 100
                        }

            return sections

        except Exception as e:
            logger.warning(f"Groq section identification failed: {str(e)}")
            return {}

    def extract_entities_with_groq(self, text: str) -> Dict[str, str]:
        try:
            if not self.use_groq:
                raise Exception("Groq integration disabled")

            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": self.groq_prompts["entity_extraction"].format(text=text)}],
                model="mixtral-8x7b-32768",
                temperature=0.1,
                max_tokens=1000
            )

            entities = {}
            response_text = response.choices[0].message.content

            for line in response_text.split('\n'):
                if ':' in line:
                    entity_type, value = line.split(':', 1)
                    entity_type = entity_type.strip().lower()
                    entities[entity_type] = value.strip()

            return entities

        except Exception as e:
            logger.warning(f"Groq entity extraction failed: {str(e)}")
            return {}

    def summarize_section_with_groq(self, section_name: str, content: str) -> str:
        try:
            if not self.use_groq:
                raise Exception("Groq integration disabled")

            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": self.groq_prompts["section_summary"].format(
                    section_name=section_name,
                    content=content
                )}],
                model="mixtral-8x7b-32768",
                temperature=0.3,
                max_tokens=1500
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning(f"Groq summarization failed: {str(e)}")
            return content

    def _extract_section_content(self, text: str, start: int, end: int) -> str:
        lines = text.split('\n')
        if start < 0:
            start = 0
        if end >= len(lines):
            end = len(lines)
        return '\n'.join(lines[start:end]).strip()

    def _fallback_entity_extraction(self, text: str) -> Dict[str, str]:
        entities = {
            "name": "",
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": "",
            "location": "",
            "current_position": ""
        }

        doc = nlp(text[:1000])
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        if person_names:
            entities["name"] = person_names[0]

        email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        if email_match:
            entities["email"] = email_match.group(0)

        phone_match = re.search(r'(?:\+\d{1,3}[-.]?)?\s*(?:\(?\d{3}\)?[-.]?)?\s*\d{3}[-.]?\d{4}', text)
        if phone_match:
            entities["phone"] = phone_match.group(0)

        linkedin_match = re.search(r'linkedin\.com/in/[a-zA-Z0-9-]+', text)
        if linkedin_match:
            entities["linkedin"] = linkedin_match.group(0)

        github_match = re.search(r'github\.com/[a-zA-Z0-9-]+', text)
        if github_match:
            entities["github"] = github_match.group(0)

        return entities

    def _fallback_section_identification(self, text: str) -> Dict[str, Dict[str, Any]]:
        sections = {}
        lines = text.split('\n')

        for idx, line in enumerate(lines):
            for section_name, patterns in self.section_patterns.items():
                for pattern in patterns:
                    if re.match(pattern, line.strip()):
                        sections[section_name] = {
                            "start": idx + 1,
                            "end": len(lines),
                            "confidence": 0.7
                        }
                        break

        return sections

    def extract_section_content(self, text: str, patterns: List[str]) -> str:
        text_lines = text.split('\n')
        content = []
        capturing = False

        for line in text_lines:
            if any(re.match(pattern, line.strip()) for pattern in patterns):
                capturing = True
                continue

            if capturing and any(re.match(pattern, line.strip())
                                 for patterns in self.section_patterns.values()
                                 for pattern in patterns):
                break

            if capturing and line.strip():
                content.append(line.strip())

        return '\n'.join(content)

    def structure_section_content(self, section_name: str, content: str) -> Any:
        if not content:
            return ""

        if section_name == "skills":
            skills = re.split(r'[,|•]', content)
            return [skill.strip() for skill in skills if skill.strip()]

        elif section_name == "education":
            education = []
            entries = content.split('\n\n')
            for entry in entries:
                edu_entry = {}
                degree_match = re.search(r'(?i)(B\.?S\.?|M\.?S\.?|Ph\.?D\.?|Bachelor|Master|Doctor|MBA)', entry)
                if degree_match:
                    edu_entry["degree"] = degree_match.group(0)
                year_match = re.search(r'20\d{2}|19\d{2}', entry)
                if year_match:
                    edu_entry["year"] = year_match.group(0)
                if edu_entry:
                    education.append(edu_entry)
            return education or content

        return content

    def parse_resume(self, file_path: str) -> Dict[str, Any]:
        text = ""
        file_type = Path(file_path).suffix.lower()

        if file_type == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_type in ['.docx', '.doc']:
            text = self.extract_text_from_docx(file_path)

        if not text:
            return {}

        sections = self.get_groq_section_boundaries(text)
        if not sections:
            sections = self._fallback_section_identification(text)

        entities = self.extract_entities_with_groq(text)
        if not entities:
            entities = self._fallback_entity_extraction(text)

        processed_sections = {}
        for section_name, boundaries in sections.items():
            if isinstance(boundaries, dict) and "start" in boundaries:
                content = self._extract_section_content(text, boundaries["start"], boundaries["end"])
            else:
                content = self.extract_section_content(text, self.section_patterns[section_name])

            if content:
                processed_content = self.summarize_section_with_groq(section_name, content)
                processed_sections[section_name] = self.structure_section_content(section_name, processed_content)

        return {
            "basics": entities,
            "sections": processed_sections,
            "metadata": {
                "parsed_date": datetime.now().isoformat(),
                "file_name": Path(file_path).name,
                "file_type": file_type,
                "parsing_method": "groq" if self.use_groq else "traditional"
            }
        }


def process_resume_directory(input_dir: str, output_dir: str, use_groq: bool = True):
    parser = ResumeParser(use_groq=use_groq)
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(file_path):
            continue

        if Path(file_path).suffix.lower() not in ['.pdf', '.docx', '.doc']:
            logger.warning(f"Skipping unsupported file: {file_name}")
            continue

        try:
            parsed_data = parser.parse_resume(file_path)

            if parsed_data:
                output_path = os.path.join(output_dir, f"{Path(file_name).stem}.json")
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(parsed_data, f, indent=4, ensure_ascii=False)

                results.append({
                    "file_name": file_name,
                    "status": "success",
                    "parsing_method": parsed_data["metadata"]["parsing_method"],
                    "output_path": output_path
                })
                logger.info(f"Successfully processed: {file_name}")
            else:
                results.append({
                    "file_name": file_name,
                    "status": "failed",
                    "error": "No data extracted"
                })
                logger.error(f"Failed to extract data from: {file_name}")

        except Exception as e:
            results.append({
                "file_name": file_name,
                "status": "error",
                "error": str(e)
            })
            logger.error(f"Error processing {file_name}: {str(e)}")

    report_path = os.path.join(output_dir, "processing_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    return results


if __name__ == "__main__":
    input_folder = r"D:\Lusak.tech\Sample Resume"
    output_folder = r"D:\Lusak.tech\try0"
    results = process_resume_directory(input_folder, output_folder, use_groq=True)