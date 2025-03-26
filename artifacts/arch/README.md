# GAIDP Financial Buddy - Solution Architecture

## Overview
The model is designed to assist businesses in analyzing financial transactions and ensuring compliance with regulatory guidelines. It leverages advanced AI models to extract rules from regulatory documents and applies these rules to transactional data for risk scoring and remediation.

This document provides an overview of the architecture, explaining both technical and business aspects.

---

## Business Perspective

### Key Features
1. **Automated Rule Extraction**: Extracts data profiling rules from regulatory documents (e.g., PDFs) using AI.
2. **Transaction Analysis**: Analyzes transactional data against extracted rules to identify violations, anomalies, and risks.
3. **Risk Scoring**: Assigns risk scores to transactions based on compliance with rules.
4. **Remediation Suggestions**: Provides actionable insights to address identified issues.
5. **Report Generation**: Generates detailed reports in JSON and Excel formats for further analysis.

### Business Benefits
- **Compliance Assurance**: Ensures adherence to regulatory guidelines, reducing the risk of penalties.
- **Operational Efficiency**: Automates manual processes, saving time and resources.
- **Actionable Insights**: Provides clear recommendations for addressing compliance issues.
- **Scalability**: Handles large volumes of transactions efficiently.

---

## Technical Perspective

### High-Level Architecture
The solution consists of the following components:

1. **Frontend**: A web-based interface built with Flask for uploading files, viewing reports, and downloading results.
2. **Backend**: A Python-based backend that processes uploaded files, extracts rules, and analyzes transactions.
3. **AI Integration**: Utilizes Google's Gemini AI model for rule extraction and validation.
4. **Data Processing**: Uses libraries like `pandas` for data manipulation and `pdfplumber` for PDF parsing.
5. **Storage**: Stores intermediate and final results in JSON and Excel formats.

## Workflow

1. **File Upload**:
   - Users upload a regulatory PDF and a transaction CSV file via the web interface.
   - A schedule name is provided to identify the relevant section in the PDF.

2. **Rule Extraction**:
   - The backend uses `pdfplumber` to extract text from the specified schedule in the PDF.
   - Google's Gemini AI model processes the text to generate data profiling rules.

3. **Transaction Analysis**:
   - The uploaded transaction data is validated against the extracted rules.
   - Violations, anomalies, and missing fields are identified.
   - Risk scores are calculated for each transaction.

4. **Report Generation**:
   - Results are saved in JSON and Excel formats.
   - Users can download the reports or view them in the web interface.

---

## Technical Components

### Backend (`main.py` and `process.py`)
- **`main.py`**:
  - Handles HTTP requests for file uploads, report generation, and downloads.
  - Manages logging and error handling.
- **`process.py`**:
  - Implements the core logic for rule extraction and transaction analysis.
  - Integrates with the Gemini AI model for advanced processing.

### AI Integration
- **Gemini AI**:
  - Extracts complex rules from regulatory documents.
  - Validates transactions against conditional and reference-based rules.

### Data Processing
- **`pandas`**:
  - Handles data manipulation and analysis.
- **`pdfplumber`**:
  - Extracts text from PDF documents.

---

## Example Use Case

1. A business uploads a regulatory document (e.g., "Guidelines for Securities") and a CSV file containing transaction data.
2. The system extracts rules from the "Securities" schedule in the document.
3. The uploaded transactions are analyzed against these rules.
4. The system identifies violations (e.g., missing fields, invalid values) and assigns risk scores.
5. A detailed report is generated, highlighting issues and providing remediation suggestions.

---

## Conclusion
The model bridges the gap between regulatory compliance and operational efficiency. By automating rule extraction and transaction analysis, it empowers businesses to stay compliant while focusing on their core operations.