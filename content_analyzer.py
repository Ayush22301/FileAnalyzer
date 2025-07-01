import re
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from dateutil import parser
import PyPDF2
from docx import Document
import io
import pytesseract
from PIL import Image
import spacy
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass


class ContentAnalyzer:
    """Advanced content analyzer for extracting meaningful information from documents."""
    
    def __init__(self):
        """Initialize the content analyzer with NLP models."""
        self.nlp = None
        self.sentiment_analyzer = None
        self.classifier = None
        
        # Try to load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Try to load transformers models
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        except Exception as e:
            logger.warning(f"Transformers models not loaded: {e}")
        
        # Document type patterns
        self.document_patterns = {
            "bank_statement": {
                "keywords": ["bank statement", "account statement", "transaction history", "balance", "deposit", "withdrawal"],
                "patterns": [
                    r"account\s+number[:\s]*(\d+)",
                    r"balance[:\s]*\$?([\d,]+\.?\d*)",
                    r"statement\s+period[:\s]*([A-Za-z]+\s+\d{4})",
                    r"customer[:\s]*([A-Za-z\s]+)",
                ]
            },
            "invoice": {
                "keywords": ["invoice", "bill", "amount due", "payment terms", "tax", "total"],
                "patterns": [
                    r"invoice\s+number[:\s]*([A-Z0-9-]+)",
                    r"amount\s+due[:\s]*\$?([\d,]+\.?\d*)",
                    r"due\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                    r"customer[:\s]*([A-Za-z\s]+)",
                ]
            },
            "resume": {
                "keywords": ["resume", "cv", "experience", "education", "skills", "objective"],
                "patterns": [
                    r"name[:\s]*([A-Za-z\s]+)",
                    r"email[:\s]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
                    r"phone[:\s]*(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})",
                    r"experience[:\s]*(\d+)\s+years?",
                ]
            },
            "contract": {
                "keywords": ["contract", "agreement", "terms", "conditions", "parties", "effective date"],
                "patterns": [
                    r"contract\s+number[:\s]*([A-Z0-9-]+)",
                    r"effective\s+date[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                    r"parties[:\s]*([A-Za-z\s&]+)",
                    r"amount[:\s]*\$?([\d,]+\.?\d*)",
                ]
            },
            "medical_report": {
                "keywords": ["medical", "diagnosis", "patient", "doctor", "treatment", "prescription"],
                "patterns": [
                    r"patient\s+name[:\s]*([A-Za-z\s]+)",
                    r"diagnosis[:\s]*([A-Za-z\s]+)",
                    r"doctor[:\s]*([A-Za-z\s]+)",
                    r"date[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                ]
            },
            "receipt": {
                "keywords": ["receipt", "purchase", "total", "tax", "store", "date"],
                "patterns": [
                    r"receipt\s+number[:\s]*([A-Z0-9-]+)",
                    r"total[:\s]*\$?([\d,]+\.?\d*)",
                    r"store[:\s]*([A-Za-z\s]+)",
                    r"date[:\s]*(\d{1,2}/\d{1,2}/\d{4})",
                ]
            }
        }
    
    def analyze_content(self, content: bytes, content_type: str, filename: str) -> Dict[str, Any]:
        """
        Analyze document content and extract meaningful information.
        
        Args:
            content: File content in bytes
            content_type: MIME type of the file
            filename: Original filename
            
        Returns:
            dict: Content analysis results
        """
        try:
            # Extract text content based on file type
            text_content = self._extract_text(content, content_type)
            
            if not text_content:
                return {"content_analysis": {"error": "Could not extract text content"}}
            
            # Perform content analysis
            analysis = {
                "document_type": self._identify_document_type(text_content, filename),
                "key_entities": self._extract_entities(text_content),
                "key_dates": self._extract_dates(text_content),
                "key_numbers": self._extract_numbers(text_content),
                "summary": self._generate_summary(text_content),
                "sentiment": self._analyze_sentiment(text_content),
                "key_phrases": self._extract_key_phrases(text_content),
                "document_metadata": self._extract_document_metadata(text_content, filename)
            }
            
            # Add specific analysis based on document type
            doc_type = analysis["document_type"]["type"]
            if doc_type in self.document_patterns:
                analysis["specific_details"] = self._extract_specific_details(text_content, doc_type)
            
            return {"content_analysis": analysis}
            
        except Exception as e:
            logger.error(f"Error in content analysis: {e}")
            return {"content_analysis": {"error": f"Content analysis failed: {str(e)}"}}
    
    def _extract_text(self, content: bytes, content_type: str) -> str:
        """Extract text content from different file types."""
        try:
            if content_type == "application/pdf":
                return self._extract_pdf_text(content)
            elif content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                return self._extract_docx_text(content)
            elif content_type in ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"]:
                return self._extract_excel_text(content)
            elif content_type.startswith("image/"):
                return self._extract_image_text(content)
            elif content_type.startswith("text/"):
                return content.decode('utf-8', errors='ignore')
            else:
                # Try to decode as text
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """Extract text from PDF files."""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            return ""
    
    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from DOCX files."""
        try:
            doc = Document(io.BytesIO(content))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {e}")
            return ""
    
    def _extract_excel_text(self, content: bytes) -> str:
        """Extract text from Excel files."""
        try:
            df = pd.read_excel(io.BytesIO(content))
            return df.to_string()
        except Exception as e:
            logger.error(f"Error extracting Excel text: {e}")
            return ""
    
    def _extract_image_text(self, content: bytes) -> str:
        """Extract text from images using OCR."""
        try:
            image = Image.open(io.BytesIO(content))
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            logger.error(f"Error extracting image text: {e}")
            return ""
    
    def _identify_document_type(self, text: str, filename: str) -> Dict[str, Any]:
        """Identify the type of document based on content and filename."""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        scores = {}
        for doc_type, patterns in self.document_patterns.items():
            score = 0
            
            # Check keywords
            for keyword in patterns["keywords"]:
                if keyword in text_lower:
                    score += 2
            
            # Check filename patterns
            if doc_type.replace("_", " ") in filename_lower:
                score += 3
            
            # Check specific patterns
            for pattern in patterns["patterns"]:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            
            scores[doc_type] = score
        
        # Find the best match
        best_type = max(scores, key=scores.get) if scores else "unknown"
        confidence = scores[best_type] / 10 if scores else 0
        
        return {
            "type": best_type,
            "confidence": min(confidence, 1.0),
            "scores": scores
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text."""
        entities = []
        
        if self.nlp:
            doc = self.nlp(text[:10000])  # Limit text length for performance
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "description": spacy.explain(ent.label_)
                })
        
        return entities[:20]  # Limit to top 20 entities
    
    def _extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',
            r'\d{1,2}-\d{1,2}-\d{4}',
            r'\d{4}-\d{1,2}-\d{1,2}',
            r'[A-Za-z]+\s+\d{1,2},?\s+\d{4}',
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}'
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return list(set(dates))[:10]  # Remove duplicates and limit
    
    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract important numbers from text."""
        numbers = []
        
        # Currency amounts
        currency_pattern = r'\$?([\d,]+\.?\d*)'
        currency_matches = re.findall(currency_pattern, text)
        for match in currency_matches[:10]:
            numbers.append({
                "value": match,
                "type": "currency",
                "context": self._get_number_context(text, match)
            })
        
        # Percentages
        percent_pattern = r'(\d+\.?\d*)%'
        percent_matches = re.findall(percent_pattern, text)
        for match in percent_matches[:5]:
            numbers.append({
                "value": match,
                "type": "percentage",
                "context": self._get_number_context(text, match)
            })
        
        return numbers
    
    def _get_number_context(self, text: str, number: str) -> str:
        """Get context around a number in text."""
        try:
            index = text.find(number)
            if index != -1:
                start = max(0, index - 50)
                end = min(len(text), index + len(number) + 50)
                return text[start:end].strip()
        except:
            pass
        return ""
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the document content."""
        try:
            # Simple summary based on key sentences
            sentences = sent_tokenize(text[:5000])  # Limit for performance
            if len(sentences) <= 3:
                return text[:500]
            
            # Find sentences with key terms
            key_terms = ["total", "amount", "balance", "date", "name", "account", "invoice", "contract"]
            important_sentences = []
            
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in key_terms):
                    important_sentences.append(sentence)
            
            if important_sentences:
                return " ".join(important_sentences[:3])
            else:
                return " ".join(sentences[:2])
                
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:300] + "..." if len(text) > 300 else text
    
    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of the document."""
        try:
            if self.sentiment_analyzer:
                result = self.sentiment_analyzer(text[:1000])  # Limit for performance
                return {
                    "label": result[0]["label"],
                    "score": result[0]["score"]
                }
            else:
                return {"label": "neutral", "score": 0.5}
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"label": "neutral", "score": 0.5}
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text."""
        try:
            # Simple key phrase extraction
            sentences = sent_tokenize(text[:3000])
            key_phrases = []
            
            for sentence in sentences:
                # Look for sentences with numbers or important terms
                if re.search(r'\d+', sentence) or any(term in sentence.lower() for term in ["total", "amount", "date", "name"]):
                    key_phrases.append(sentence.strip())
            
            return key_phrases[:5]
        except Exception as e:
            logger.error(f"Error extracting key phrases: {e}")
            return []
    
    def _extract_document_metadata(self, text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from document."""
        metadata = {
            "filename": filename,
            "word_count": len(text.split()),
            "sentence_count": len(sent_tokenize(text)),
            "has_numbers": bool(re.search(r'\d+', text)),
            "has_dates": bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{4}', text)),
            "has_currency": bool(re.search(r'\$[\d,]+\.?\d*', text)),
            "has_email": bool(re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)),
            "has_phone": bool(re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', text))
        }
        
        return metadata
    
    def _extract_specific_details(self, text: str, doc_type: str) -> Dict[str, Any]:
        """Extract specific details based on document type."""
        details = {}
        
        if doc_type in self.document_patterns:
            patterns = self.document_patterns[doc_type]["patterns"]
            
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Extract the pattern name from the regex
                    pattern_name = re.search(r'\(([^)]+)\)', pattern)
                    if pattern_name:
                        key = pattern_name.group(1)
                        details[key] = matches[0] if matches else None
        
        return details 