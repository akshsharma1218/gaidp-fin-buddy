import pdfplumber
import re
import json
from typing import List, Dict, Optional, Union
import google.generativeai as genai
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
from fuzzywuzzy import fuzz
import time
import os
from keys import GOOGLE_API_KEY  # Store your API key here

# Configuration
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
logging.getLogger('pdfminer').setLevel(logging.ERROR)

nltk.download('punkt')
nltk.download('stopwords')

# Ensure the Logs directory and log file exist
log_dir = 'Logs'
log_file = os.path.join(log_dir, 'process.log')
os.makedirs(log_dir, exist_ok=True)
if not os.path.exists(log_file):
    with open(log_file, 'w', encoding='utf-8') as f:  # Use UTF-8 encoding
        pass  # Create an empty log file

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=os.path.join('Logs', 'process.log'),
    filemode='w',
    encoding='utf-8'  # Use UTF-8 encoding
)

# Add this line to remove ANSI escape sequences from logs
logging.getLogger().handlers[0].addFilter(lambda record: setattr(record, 'msg', re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', record.msg)) or True)

# Initialize Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-pro')

class Processor:
    def __init__(self):
        self.batch_size = 20
        self.max_retries = 3
        self.retry_delay = 60
        self.json_encoder = json.JSONEncoder(default=self._json_serializer)
        self.index_pattern = re.compile(r"(?i)(SCHEDULE|SECTION)\s+([A-Z0-9—-]+)\s*\.{3,}\s*(\d+)")
        self.gemini_prompt = """
        [IMPORTANT INSTRUCTIONS]
        1. Return the COMPLETE JSON response without truncation in the specified format ({format}).
        2. Ensure all rules are fully specified, specially those derived from tables.
        3. If the input contains a table with fields such as "Field No.", "Field Name", "Description", or "Allowable Values", there must be a rule for each field from the table.
        4. Include all allowed values, conditions, and constraints directly in the response.

        Extract ALL data profiling rules for {schedule_name} from the text below. These rules will be used for adaptive risk scoring of transactional data and providing remediation. The structure of the input dataset will closely resemble the structure of tables in the text, if tables are present.

        {text}

        Return the JSON response in the following structure, ensuring completeness for each rule:

        - **field**: The field name (exactly as it appears in tables, if present).
        - **required**: True or False (default is False).
        - **description**: A detailed description including:
            - Expected data type and format specifications.
            - All valid values or value ranges.
            - References to any related fields or tables.
            - Full conditional logic, if applicable.
            - Any business rules or validation requirements.
        - **constraint_type**: One of the following:
            - "format": Include the complete format specification (e.g., regex).
            - "value": Include all allowed values.
            - "range": Include complete min/max values.
            - "reference": Include full reference details.
            - "conditional": Include complete conditional logic.
        - **allowed_values**:
            - For "value": Provide a complete list of allowed values.
            - For "format": Provide the full format specification (e.g., regex).
            - For "range": Provide the complete min/max values (use null if unbounded).
            - For "reference": Provide full reference details, including:
                - Referenced field/table name.
                - Exact matching requirements.
                - Complete relationship description.
            - For "conditional": Provide complete conditional logic, including:
                - All fields involved.
                - Full logical expression.
                - Expected outcomes.
        - **severity**: One of "critical", "high", "medium", or "low".
        - **dependencies**: A list of field names this rule depends on.
        - **source**: A one-line summary of the source text for traceability.

        Ensure the response is well-structured, comprehensive, and adheres to the above format.
        """

    def _call_gemini_with_retry(self, prompt: str) -> str:
        """Call Gemini API with retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.1
                    }
                )
                return response.text
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    logging.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds... Error: {str(e)}")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"Failed after {self.max_retries} attempts. Last error: {str(e)}")
        return ""

    def _save_prompt_response(self, prompt: str, response: str, prefix: str = "") -> None:
        """Save prompt and response to files"""
        timestamp = int(time.time())
        os.makedirs("Logs/gemini_logs", exist_ok=True)
        
        prompt_file = f"Logs/gemini_logs/{prefix}prompt_{timestamp}.txt"
        response_file = f"Logs/gemini_logs/{prefix}response_{timestamp}.txt"
        
        with open(prompt_file, 'w', encoding='utf-8') as f:  # Use UTF-8 encoding
            f.write(prompt)
        
        with open(response_file, 'w', encoding='utf-8') as f:  # Use UTF-8 encoding
            f.write(response)


    def _generate_rules_with_gemini(self, text: str, schedule_name: str) -> List[Dict]:
        """Generate rules using Gemini with retry logic and proper logging"""
        try:
            prompt = self.gemini_prompt.format(
                format = r'{"rules":[{"field":"field_name","required":"true",....},]}',
                schedule_name=schedule_name,
                text=text  # Limit text size
            )
            
            self._save_prompt_response(prompt, "", "rule_gen_")
            response_text = self._call_gemini_with_retry(prompt)
            time.sleep(50)
            self._save_prompt_response(prompt, response_text, "rule_gen_")
            
            json_str = self._extract_json(response_text)
            if not json_str:
                raise ValueError("No valid JSON found in response")
            rules = json.loads(json_str)
            if isinstance(rules, dict) and 'rules' in rules:
                rules = rules['rules']
            elif not isinstance(rules, list):
                raise ValueError("Expected list of rules or object with 'rules' key")
                
            # Validate each rule has required fields
            for rule in rules:
                if not all(key in rule for key in ['field', 'constraint_type']):
                    raise ValueError("Invalid rule structure in response")
                    
            return rules
            
        except Exception as e:
            logging.error(f"Rule extraction failed: {e}")
            return []

    def process_schedule(self, pdf_path: str, target_schedule: str) -> Dict:
        """Extract rules from PDF schedule with retry logic"""
        pdf_path = pdf_path
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page_range = self._get_schedule_pages(pdf, target_schedule)
                if not page_range:
                    raise ValueError(f"Schedule {target_schedule} not found in {pdf_path}")
                schedule_text = self._extract_pages_text(pdf, page_range)

                rules = self._generate_rules_with_gemini(schedule_text, target_schedule)
                return {
                    "schedule": target_schedule,
                    "pages": f"{page_range[0] + 1}-{page_range[1] + 1}",
                    "rules": rules,
                }
        except Exception as e:
            logging.exception(f"Error processing schedule: {e}")
            return {}

    def _extract_json(self, text: str) -> str:
        """Extract complete JSON from Gemini response"""
        # Try to find JSON between markers
        json_markers = ['```json', '```']
        for marker in json_markers:
            if marker in text:
                parts = text.split(marker)
                if len(parts) > 1:
                    return parts[1].split('```')[0].strip()
        
        # Fallback: find first { and last }
        try:
            first_brace = text.find('{')
            last_brace = text.rfind('}')
            if first_brace != -1 and last_brace != -1:
                return text[first_brace:last_brace+1]
        except:
            pass
            
        return ""

    def analyze_transactions(self, transactions: List[Dict], rules: List[Dict]) -> pd.DataFrame:
        """Process transactions with NA handling"""
        if not transactions or not rules:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(transactions).convert_dtypes()
            if df.empty:
                return df
                
            # Initialize with NA-safe defaults
            df = df.assign(
                risk_score=0,
                remediation='',
                violations=None,
                anomalies=None,
                warning_flags=0
            )
            
            for batch_start in range(0, len(df), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(df))
                batch = df.iloc[batch_start:batch_end]
                
                local_results = self._process_batch_locally(batch, rules)
                
                try:
                    prompt = self._create_batch_prompt(local_results, rules)
                    batch_response = self._call_gemini_with_retry(
                        prompt
                    )
                    self._save_prompt_response(prompt, batch_response, f"batch_{batch_start}")
                    # print("local_results ", local_results)
                    final_results = self._validate_batch_with_gemini(
                        local_results, 
                        rules,
                        batch_response
                    )
                    # print("final_results", final_results)
                except Exception as e:
                    logging.error(f"Batch validation failed: {e}")
                    final_results = local_results
                
                for idx, result in final_results.items():
                    # Convert NA values to None before JSON serialization
                    violations = result['violations'] if 'violations' in result else None
                    anomalies = result['anomalies'] if 'anomalies' in result else None
                    
                    df.at[idx, 'risk_score'] = result.get('risk_score', 0)
                    df.at[idx, 'remediation'] = result.get('remediation', '')
                    df.at[idx, 'violations'] = json.dumps(
                        violations, 
                        default=self._json_serializer
                    ) if violations else None
                    df.at[idx, 'anomalies'] = json.dumps(
                        anomalies,
                        default=self._json_serializer
                    ) if anomalies else None
                    df.at[idx, 'warning_flags'] = result.get('warning_flags', 0)
                    print(df.head())
            return df
            
        except Exception as e:
            logging.exception(f"Transaction analysis failed: {e}")
            return pd.DataFrame()

    def _json_serializer(self, obj):
        if pd.isna(obj):
            return None
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} not serializable")

    def _validate_batch_with_gemini(self, local_results: Dict[int, Dict], rules: List[Dict], response_text: str) -> Dict[int, Dict]:
        """Process Gemini batch response with NA handling"""
        complex_rules = [r for r in rules if r.get('constraint_type') in ['conditional', 'reference']]
        if not complex_rules:
            return local_results
            
        try:
            gemini_results = self._parse_gemini_response(response_text)
            
            for idx, result in local_results.items():
                if str(idx) in gemini_results:
                    gemini_data = gemini_results[str(idx)]
                    
                    # Ensure all values are JSON-serializable
                    violations = gemini_data.get('violations', [])
                    if violations:
                        result['violations'].extend([
                            {k: (None if pd.isna(v) else v) 
                            for k, v in violation.items()}
                            for violation in violations
                        ])
                        
                    result['risk_score'] = min(100, result['risk_score'] + gemini_data.get('risk_score_adjustment', 0))
                    result['remediation'] = gemini_data.get('remediation', '')
                    result['warning_flags'] = len(result.get('missing_fields', [])) + \
                                            len(result.get('violations', [])) + \
                                            len(result.get('anomalies', []))
                    
            return local_results
            
        except Exception as e:
            logging.error(f"Failed to process Gemini response: {e}")
            return local_results

    def _process_batch_locally(self, batch: pd.DataFrame, rules: List[Dict]) -> Dict[int, Dict]:
        """Process a batch of transactions without API calls"""
        results = {}
        field_map = self._create_field_map(batch.columns, rules)
        for idx, row in batch.iterrows():
            transaction = self._prepare_transaction(row, field_map)
            required_fields = {r['field'].lower() for r in rules if r.get('required', False)}
            
            missing_fields = [
                f for f in required_fields 
                if f not in transaction or pd.isna(transaction.get(f))
            ]
            
            violations = self._check_rules_locally(transaction, rules, missing_fields)
            anomalies = self._detect_anomalies(transaction, rules)
            
            results[idx] = {
                'transaction': transaction,
                'missing_fields': missing_fields,
                'violations': violations,
                'anomalies': anomalies,
                'risk_score': self._calculate_risk_score(missing_fields, violations, anomalies),
                'remediation': '',
                'warning_flags': len(missing_fields) + len(violations) + len(anomalies)
            }
            
        return results

    def _create_batch_prompt(self, local_results: Dict[int, Dict], complex_rules: List[Dict]) -> str:
        """Create prompt for batch validation"""
        sample_transactions = {k: v for k, v in list(local_results.items())[:3]}  # Include first 3 as examples
        
        return f"""
        Analyze these transactions against complex rules and:
        1. Validate all conditional/reference rules
        2. Identify any additional violations
        3. Provide comprehensive remediation advice

        COMPLEX RULES:
        {json.dumps(complex_rules, indent=2)}

        SAMPLE TRANSACTIONS (showing structure):
        {json.dumps(sample_transactions, indent=2)}

        FULL BATCH DATA (validate all transactions):
        {json.dumps(local_results, indent=2)}

        Return JSON with:
        - transaction_id: Original index
        - violations: Additional violations found (empty if none)
        - risk_score_adjustment: Points to add to local score (0 if none)
        - remediation: Comprehensive advice

        Structure your response as:
        {{
            "results": [
                {{
                    "transaction_id": 0,
                    "violations": [...],
                    "risk_score_adjustment": 5,
                    "remediation": "..."
                }},
                ...
            ]
        }}
        """

    def _parse_gemini_response(self, response_text: str) -> Dict[str, Dict]:
        """Parse Gemini's batch response"""
        try:
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                json_str = response_text.split('```')[1]
            else:
                json_str = response_text
                
            data = json.loads(json_str)
            return {str(item['transaction_id']): item for item in data.get('results', [])}
        except Exception as e:
            logging.error(f"Failed to parse Gemini response: {e}")
            return {}

    def _create_field_map(self, columns: List[str], rules: List[Dict]) -> Dict[str, str]:
        """Map rule fields to transaction columns"""
        field_map = {}
        rule_fields = {r['field'].lower() for r in rules if 'field' in r}
        
        for rule_field in rule_fields:
            best_match, best_score = None, 0
            for col in columns:
                trans_field = str(col).lower()
                score = max(
                    fuzz.token_set_ratio(rule_field, trans_field),
                    fuzz.partial_ratio(rule_field, trans_field)
                )
                
                if score > best_score and score > 70:
                    best_score = score
                    best_match = col
                    
            if best_match:
                field_map[rule_field] = best_match
                
        return field_map

    def _prepare_transaction(self, row: pd.Series, field_map: Dict[str, str]) -> Dict:
        """Prepare transaction data with type conversion"""
        transaction = {}
        for rule_f, trans_f in field_map.items():
            if trans_f in row:
                val = row[trans_f]
                if pd.isna(val):
                    transaction[rule_f] = None
                elif isinstance(val, str):
                    transaction[rule_f] = val.strip() or None
                else:
                    transaction[rule_f] = val
        return transaction

    def _check_rules_locally(self, transaction: Dict, rules: List[Dict], missing_fields: List) -> List[Dict]:
        """Local validation without API calls"""
        violations = []
        for rule in rules:
            if rule.get('constraint_type') in ['conditional', 'reference']:
                continue  # Skip complex rules for now
                
            field = rule.get('field', '').lower()

            if field in missing_fields:
                violations.append({
                    'field': field,
                    'value': None,
                    'violation': f"Missing required field: {field}",
                    'severity': rule.get('severity', 'critical')
                })
                continue
            elif field not in transaction:
                continue
            value = transaction[field]
                
            if pd.isna(value):
                continue
                
            if rule.get('constraint_type') == 'value':
                allowed = rule.get('allowed_values', [])
                if allowed and str(value) not in map(str, allowed):
                    violations.append({
                        'field': field,
                        'value': value,
                        'violation': f"Value not in allowed values: {allowed}",
                        'severity': rule.get('severity', 'medium')
                    })
                    
            elif rule.get('constraint_type') == 'format':
                pattern = rule.get('allowed_values')
                if pattern and not re.match(pattern, str(value)):
                    violations.append({
                        'field': field,
                        'value': value,
                        'violation': f"Format mismatch (expected: {pattern})",
                        'severity': rule.get('severity', 'medium')
                    })
                    
            elif rule.get('constraint_type') == 'range':
                try:
                    num = float(value)
                    min_val = rule.get('allowed_values', {}).get('min')
                    max_val = rule.get('allowed_values', {}).get('max')
                    
                    if (min_val is not None and num < min_val) or (max_val is not None and num > max_val):
                        violations.append({
                            'field': field,
                            'value': value,
                            'violation': f"Value out of range ({min_val}-{max_val})",
                            'severity': rule.get('severity', 'high')
                        })
                except (ValueError, TypeError):
                    violations.append({
                        'field': field,
                        'value': value,
                        'violation': "Invalid numeric value",
                        'severity': rule.get('severity', 'high')
                    })
                    
        return violations

    def _calculate_risk_score(self, missing_fields: List[str], violations: List[Dict], anomalies: List[Dict]) -> int:
        """Calculate risk score locally with fallback for NaN"""
        try:
            score = len(missing_fields) * 3  # Missing fields are critical
            
            severity_weights = {
                'critical': 3,
                'high': 2,
                'medium': 1,
                'low': 0.5
            }
            
            for violation in violations:
                score += severity_weights.get(violation.get('severity', 'medium'), 1)
                
            score += len(anomalies) * 2  # Anomalies are weighted higher
            
            # Ensure score is within valid bounds
            return min(100, max(0, int(score)))
        except Exception as e:
            logging.error(f"Error calculating risk score: {e}")
            return 0  # Default to 0 in case of error

    def _detect_anomalies(self, transaction: Dict, rules: List[Dict]) -> List[Dict]:
        """Detect numerical anomalies without API calls"""
        anomalies = []
        numerical_fields = [
            r['field'] for r in rules 
            if r.get('field') in transaction and 
            isinstance(transaction[r['field']], (int, float, np.number))
        ]
        
        if not numerical_fields:
            return anomalies
            
        try:
            values = np.array([transaction[f] for f in numerical_fields]).reshape(-1, 1)
            scaler = StandardScaler()
            scaled = scaler.fit_transform(values)
            
            dbscan = DBSCAN(eps=1.5, min_samples=1).fit(scaled)
            
            for i, label in enumerate(dbscan.labels_):
                if label == -1:
                    anomalies.append({
                        'field': numerical_fields[i],
                        'value': transaction[numerical_fields[i]],
                        'violation': "Numerical anomaly detected",
                        'severity': 'high'
                    })
                    
        except Exception as e:
            logging.error(f"Anomaly detection failed: {e}")
            
        return anomalies

    def _get_schedule_pages(self, pdf, target_schedule: str) -> Optional[tuple]:
        """Original method unchanged"""
        for page in pdf.pages[:5]:
            text = page.extract_text()
            if not text:
                continue
            schedule_map = {}
            for match in self.index_pattern.finditer(text):
                name, page_num = match.group(2), int(match.group(3))
                schedule_map[name] = page_num - 1
            if target_schedule in schedule_map:
                start_page = schedule_map[target_schedule]
                next_page = min(
                    [p for s, p in schedule_map.items() if p > start_page and s != target_schedule],
                    default=start_page + 5,
                )
                return (start_page, next_page - 1)
        return None

    def _extract_pages_text(self, pdf, page_range: tuple) -> str:
        """Original method unchanged"""
        start, end = page_range
        text_blocks = []
        for page in pdf.pages[start:end + 1]:
            text = page.extract_text()
            if text:
                clean_text = re.sub(r"Page \d+ of \d+", "", text)
                clean_text = re.sub(r"FR 2052a\s*\d+", "", clean_text)
                text_blocks.append(clean_text.strip())
        return "\n".join(text_blocks)

def save_json(data: Union[Dict, List], filename: str) -> None:
    """Helper to save JSON data"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    logging.info(f"Saved data to {filename}")

def load_json(filename: str) -> Union[Dict, List]:
    """Helper to load JSON data"""
    with open(filename, 'r') as f:
        data = json.load(f)
    logging.info(f"Loaded data from {filename}")
    return data

if __name__ == "__main__":
    processor = Processor()
    
    # Step 1: Extract rules with retry logic
    logging.info("Extracting rules from PDF...")
    
    results = processor.process_schedule("guidelines_tran.pdf", "B—Securities")
    if results and results['rules']:
        save_json(results['rules'], "extracted_rules.json")
        logging.info(f"Extracted {len(results['rules'])} rules from f{results["pages"]}")
    else:
        logging.error("Failed to extract rules")
        exit(1)
    # results = {}
    # results["rules"] = load_json("extracted_rules.json")
    # Step 2: Load transactions
    logging.info("Loading transactions...")
    try:
        transactions = pd.read_csv("transaction.csv").head(500).to_dict(orient="records")
        logging.info(f"Loaded {len(transactions)} transactions")
    except Exception as e:
        logging.error(f"Failed to load transactions: {e}")
        exit(1)
    
    # Step 3: Process transactions with retry logic
    logging.info("Analyzing transactions...")
    analysis_df = processor.analyze_transactions(transactions, results["rules"])
    
    # Step 4: Save results
    logging.info("Saving results...")
    analysis_df.to_excel("analysis_report.xlsx", index=False)
    analysis_df.to_json("analysis_report.json", orient="records", indent=4)
    
    print("\nAnalysis Complete")
    print(f"Transactions processed: {len(transactions)}")
    print(f"Average risk score: {analysis_df['risk_score'].mean():.1f}")
    print(f"Violations found: {analysis_df['violations'].notna().sum()}")
    print(f"Reports saved to analysis_report.xlsx/json")