import re
import pandas as pd
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict
import unicodedata
import json
from pathlib import Path

class MedicalTextUtils:
    """Utilities for medical text processing and transformation."""
    
    # Common medical units and their standardized forms
    UNIT_MAPPINGS = {
        'mg/dl': 'mg/dL',
        'mg/dL': 'mg/dL',
        'mmol/l': 'mmol/L',
        'mmol/L': 'mmol/L',
        'ml': 'mL',
        'mcg': 'μg',
        'ng/ml': 'ng/mL',
        'u/l': 'U/L',
        'g/dl': 'g/dL',
    }
    
    # Common lab test reference ranges
    LAB_RANGES = {
        'glucose': {'unit': 'mg/dL', 'low': 70, 'high': 100},
        'hemoglobin': {'unit': 'g/dL', 'low': 12, 'high': 16},
        'wbc': {'unit': '×10⁹/L', 'low': 4.0, 'high': 11.0},
        'platelet': {'unit': '×10⁹/L', 'low': 150, 'high': 450},
        'creatinine': {'unit': 'mg/dL', 'low': 0.6, 'high': 1.2},
    }

    @staticmethod
    def standardize_units(text: str) -> str:
        """
        Standardize medical measurement units in text.
        
        Args:
            text: Input text containing medical measurements
            
        Returns:
            Text with standardized units
        """
        for old_unit, new_unit in MedicalTextUtils.UNIT_MAPPINGS.items():
            # Match units with various spacing patterns
            pattern = rf'\b{old_unit}\b|\b{old_unit.lower()}\b|\b{old_unit.upper()}\b'
            text = re.sub(pattern, new_unit, text)
        return text
    
    @staticmethod
    def extract_measurements(text: str) -> List[Dict[str, Union[str, float]]]:
        """
        Extract numerical measurements with units from text.
        
        Args:
            text: Input text containing measurements
            
        Returns:
            List of dictionaries containing value and unit
        """
        # Pattern to match number followed by unit
        pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+/?[a-zA-Z]*)'
        matches = re.finditer(pattern, text)
        
        measurements = []
        for match in matches:
            value, unit = match.groups()
            measurements.append({
                'value': float(value),
                'unit': unit,
                'original_text': match.group()
            })
        return measurements

    @staticmethod
    def normalize_lab_values(value: float, test_name: str) -> Dict[str, Union[float, str]]:
        """
        Normalize lab values relative to reference range.
        
        Args:
            value: Lab value
            test_name: Name of the lab test
            
        Returns:
            Dictionary with normalized value and status
        """
        if test_name not in MedicalTextUtils.LAB_RANGES:
            return {'normalized_value': value, 'status': 'unknown'}
            
        ranges = MedicalTextUtils.LAB_RANGES[test_name]
        normalized = (value - ranges['low']) / (ranges['high'] - ranges['low'])
        
        status = 'normal'
        if value < ranges['low']:
            status = 'low'
        elif value > ranges['high']:
            status = 'high'
            
        return {
            'normalized_value': normalized,
            'status': status,
            'reference_range': ranges
        }

class TimeSeriesUtils:
    """Utilities for handling temporal medical data."""
    
    @staticmethod
    def create_timeline(events: List[Dict[str, Union[str, datetime]]]) -> pd.DataFrame:
        """
        Create a timeline from medical events.
        
        Args:
            events: List of event dictionaries with timestamps
            
        Returns:
            DataFrame with organized timeline
        """
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Calculate time differences between events
        df['time_since_previous'] = df['timestamp'].diff()
        return df
    
    @staticmethod
    def aggregate_events(timeline_df: pd.DataFrame, 
                        window: str = '1D') -> pd.DataFrame:
        """
        Aggregate events over specified time windows.
        
        Args:
            timeline_df: Timeline DataFrame
            window: Time window for aggregation
            
        Returns:
            Aggregated DataFrame
        """
        return timeline_df.set_index('timestamp').resample(window).agg({
            'event_type': 'count',
            'event_description': lambda x: list(x)
        }).reset_index()

class TextNormalizationUtils:
    """Utilities for text normalization and cleaning."""
    
    @staticmethod
    def remove_phi(text: str) -> str:
        """
        Remove potential PHI (Personal Health Information).
        
        Args:
            text: Input text
            
        Returns:
            Text with potential PHI removed
        """
        # Remove dates
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '[DATE]', text)
        
        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+\.\S+', '[EMAIL]', text)
        
        # Remove SSN-like numbers
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
    
    @staticmethod
    def standardize_format(text: str) -> str:
        """
        Standardize text format and spacing.
        
        Args:
            text: Input text
            
        Returns:
            Standardized text
        """
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Standardize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Standardize newlines
        text = re.sub(r'[\r\n]+', '\n', text)
        
        return text.strip()

class DemographicUtils:
    """Utilities for handling demographic information."""
    
    @staticmethod
    def calculate_age(birth_date: Union[str, datetime],
                     reference_date: Optional[datetime] = None) -> int:
        """
        Calculate age from birth date.
        
        Args:
            birth_date: Date of birth
            reference_date: Date to calculate age at (defaults to current date)
            
        Returns:
            Age in years
        """
        if isinstance(birth_date, str):
            birth_date = pd.to_datetime(birth_date)
        
        reference_date = reference_date or datetime.now()
        return (reference_date - birth_date).days // 365
    
    @staticmethod
    def parse_demographics(text: str) -> Dict[str, str]:
        """
        Extract demographic information from text.
        
        Args:
            text: Input text containing demographic information
            
        Returns:
            Dictionary of demographic information
        """
        demographics = {}
        
        # Extract age
        age_match = re.search(r'\b(\d{1,3})\s*(?:year|yr|y)[s\s]*old\b', text, re.IGNORECASE)
        if age_match:
            demographics['age'] = int(age_match.group(1))
        
        # Extract gender
        gender_match = re.search(r'\b(male|female|m\b|f\b)', text, re.IGNORECASE)
        if gender_match:
            gender = gender_match.group(1).lower()
            demographics['gender'] = 'male' if gender in ['male', 'm'] else 'female'
        
        return demographics

class DiagnosisUtils:
    """Utilities for handling medical diagnoses."""
    
    @staticmethod
    def extract_icd_codes(text: str) -> List[str]:
        """
        Extract ICD-10 codes from text.
        
        Args:
            text: Input text containing ICD codes
            
        Returns:
            List of ICD codes
        """
        # Match ICD-10 code pattern
        pattern = r'\b[A-Z]\d{2}(?:\.[A-Z\d]{1,4})?\b'
        return re.findall(pattern, text)
    
    @staticmethod
    def categorize_diagnosis(diagnosis: str) -> str:
        """
        Categorize diagnosis into major disease categories.
        
        Args:
            diagnosis: Diagnosis text
            
        Returns:
            Disease category
        """
        categories = {
            'cardiac': r'\b(heart|cardiac|coronary|myocardial|arrhythmia)\b',
            'respiratory': r'\b(lung|respiratory|pneumonia|copd|asthma)\b',
            'neurological': r'\b(brain|neural|stroke|seizure|alzheimer)\b',
            'gastrointestinal': r'\b(stomach|intestinal|liver|pancreas|gastric)\b',
            'endocrine': r'\b(diabetes|thyroid|hormonal|endocrine)\b'
        }
        
        for category, pattern in categories.items():
            if re.search(pattern, diagnosis, re.IGNORECASE):
                return category
                
        return 'other'

def main():
    """Example usage of utility functions"""
    
    # Example medical text
    sample_text = """
    Patient is a 45-year-old female with glucose 120 mg/dl and BP 140/90.
    Lab results show WBC 12.5 x10^9/l and Hgb 11.2 g/dl.
    Diagnosed with Type 2 Diabetes (E11.9) on 01/15/2024.
    Contact: 123-456-7890, email@example.com
    """
    
    # Initialize utilities
    medical_utils = MedicalTextUtils()
    text_utils = TextNormalizationUtils()
    diagnosis_utils = DiagnosisUtils()
    
    # Process text
    processed_text = text_utils.remove_phi(sample_text)
    processed_text = medical_utils.standardize_units(processed_text)
    
    print("Processed Text:")
    print(processed_text)
    
    # Extract measurements
    measurements = medical_utils.extract_measurements(processed_text)
    print("\nExtracted Measurements:")
    print(measurements)
    
    # Extract demographics
    demographics = DemographicUtils.parse_demographics(processed_text)
    print("\nDemographics:")
    print(demographics)
    
    # Extract ICD codes
    icd_codes = diagnosis_utils.extract_icd_codes(processed_text)
    print("\nICD Codes:")
    print(icd_codes)

if __name__ == "__main__":
    main()
