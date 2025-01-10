import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Union, Dict
import logging
import re
from datetime import datetime

class MedicalTextPreprocessor:
    """
    Handles preprocessing specific to medical text data.
    """
    
    # Common medical abbreviations and their expansions
    MEDICAL_ABBREVIATIONS = {
        'pt': 'patient',
        'dx': 'diagnosis',
        'hx': 'history',
        'tx': 'treatment',
        'sx': 'symptoms',
        'rx': 'prescription',
        'abd': 'abdominal',
        'temp': 'temperature',
        'hr': 'heart rate',
        'bp': 'blood pressure',
        'lab': 'laboratory',
        'w/': 'with',
        'w/o': 'without',
        'yo': 'years old',
        'yo f': 'year old female',
        'yo m': 'year old male',
    }
    
    # Regular expressions for common medical patterns
    MEASUREMENT_PATTERNS = {
        'blood_pressure': r'\b(\d{2,3})/(\d{2,3})\b',
        'temperature': r'\b\d{2}\.?\d*[°]?[CF]\b',
        'weight': r'\b\d+\.?\d*\s*(?:kg|lbs?)\b',
        'height': r'\b\d+\.?\d*\s*(?:cm|m|ft|in)\b',
        'lab_values': r'\b\d+\.?\d*\s*(?:mg/dL|mmol/L|mEq/L|µg|ng)\b'
    }

    def __init__(self):
        """Initialize the preprocessor with compiled regex patterns."""
        self.vital_signs_pattern = re.compile(
            r'\b(temp|bp|hr|rr|spo2|pulse|height|weight)[\s:].*?(?=\b(?:temp|bp|hr|rr|spo2|pulse|height|weight)\b|$)',
            re.IGNORECASE
        )

    def standardize_measurements(self, text: str) -> str:
        """
        Standardize various medical measurements in the text.
        """
        # Standardize blood pressure
        text = re.sub(
            self.MEASUREMENT_PATTERNS['blood_pressure'],
            lambda x: f"blood pressure {x.group(1)}/{x.group(2)} mmHg",
            text
        )
        
        # Standardize temperature
        def convert_temp(match):
            temp = match.group()
            if 'F' in temp:
                value = float(re.findall(r'\d+\.?\d*', temp)[0])
                celsius = round((value - 32) * 5/9, 1)
                return f"{celsius}°C"
            return temp
        text = re.sub(self.MEASUREMENT_PATTERNS['temperature'], convert_temp, text)
        
        return text

    def expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations to their full form.
        """
        words = text.split()
        expanded_words = []
        
        i = 0
        while i < len(words):
            word = words[i].lower()
            two_word_abbrev = f"{word} {words[i+1].lower()}" if i < len(words)-1 else ""
            
            if two_word_abbrev in self.MEDICAL_ABBREVIATIONS:
                expanded_words.append(self.MEDICAL_ABBREVIATIONS[two_word_abbrev])
                i += 2
            elif word in self.MEDICAL_ABBREVIATIONS:
                expanded_words.append(self.MEDICAL_ABBREVIATIONS[word])
                i += 1
            else:
                expanded_words.append(words[i])
                i += 1
        
        return ' '.join(expanded_words)

    def structure_vital_signs(self, text: str) -> str:
        """
        Structure vital signs into a consistent format.
        """
        def format_vital(match):
            vital = match.group().strip()
            vital_type = vital.split()[0].lower()
            
            try:
                value = re.search(r'[\d./]+', vital).group()
                return f"{vital_type}: {value}"
            except:
                return vital
                
        return self.vital_signs_pattern.sub(format_vital, text)

    def clean_text(self, text: str) -> str:
        """
        Clean and standardize medical text.
        """
        # Remove multiple spaces and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize dates
        text = re.sub(
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            lambda x: datetime.strptime(x.group(), '%m/%d/%Y' if len(x.group().split('/')[-1]) == 4 else '%m/%d/%y')
            .strftime('%Y-%m-%d'),
            text
        )
        
        # Remove redundant punctuation
        text = re.sub(r'[.,;]+(?=[.,;])', '', text)
        
        return text.strip()

    def preprocess(self, text: str) -> str:
        """
        Apply all preprocessing steps to the text.
        """
        text = self.clean_text(text)
        text = self.standardize_measurements(text)
        text = self.expand_abbreviations(text)
        text = self.structure_vital_signs(text)
        return text
