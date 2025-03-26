# 🚀 GAIDP Fin Buddy

## 📌 Table of Contents
- [Introduction](#introduction)
- [Demo](#demo)
- [Inspiration](#inspiration)
- [What It Does](#what-it-does)
- [How We Built It](#how-we-built-it)
- [Challenges We Faced](#challenges-we-faced)
- [How to Run](#how-to-run)
- [Tech Stack](#tech-stack)
- [Architecture Overview](#architecture-overview)
- [Example Use Case](#example-use-case)
- [Additional Details](#additional-details)
- [Team](#team)

---

## 🎯 Introduction
GAIDP Fin Buddy is a financial assistant application designed to help businesses and individuals stay compliant with regulatory guidelines. It enables users to create data profiling rules from regulatory documents and apply these rules to transactional data for compliance analysis, risk scoring, and remediation.

## 🎥 Demo
🔗 [Live Demo](http://localhost:5000/)  
📹 [Video Demo](.artifacts/demo/fin-buddy-demo - Made with Clipchamp.mp4)  

## 💡 Inspiration
The inspiration for GAIDP Fin Buddy came from the need to help businesses and individuals stay compliant with regulatory guidelines. Understanding and mitigating risks associated with regulatory compliance can be challenging, and this application aims to simplify the process by providing actionable insights and automated analysis.

## ⚙️ What It Does
- **Creates Profile Rules**: Extracts data profiling rules from user-provided regulatory documents (e.g., PDFs).
- **Analyzes Transactions**: Validates transactional data against the extracted rules to identify compliance violations, anomalies, and risks.
- **Generates Reports**: Produces detailed reports with actionable insights to help users address compliance issues.
- **Ensures Regulatory Compliance**: Helps businesses and individuals adhere to regulatory guidelines efficiently.

## 🛠️ How We Built It
- **AI Integration**: Leveraged Google Generative AI for extracting complex rules from regulatory documents.
- **Backend Logic**: Developed custom logic in Python to process and validate transactional data.
- **Server**: Deployed the application on a FastAPI server for robust and scalable API handling.
- **Data Processing**: Used libraries like `pandas` and `pdfplumber` for data manipulation and PDF parsing.

## 🚧 Challenges We Faced
- **Limited Free API Usage**: The free tier of Google Generative AI APIs restricted the number of requests we could make, requiring careful optimization.
- **Computational Resources**: Lack of sufficient computational resources posed challenges in processing large datasets efficiently.
- **Dataset Creation**: Had to create our own datasets for testing and validation due to the unavailability of suitable public datasets.
- **Process Optimization**: Integrated multiple APIs and optimized the workflow to ensure smooth and efficient processing.

## 🏃 How to Run
1. Clone the repository  
   ```sh
   git clone https://github.com/gaidp-fin-buddy.git
   ```
2. Install dependencies  
   ```sh
   pip install -r requirements.txt
   ```
3. Add your Gemini API keys (optional for faster processing):  
   - Open the `keys.py` file in the `code` directory.
   - Add your Gemini API key in the following format:  
     ```python
     GOOGLE_API_KEY = "your_api_key_here"
     ```
4. Run the project  
   ```sh
   py .\code\src\main.py
   ```

## 🏗️ Tech Stack
- 🔹 Frontend: JS/HTML
- 🔹 Backend: FastAPI
- 🔹 Engine: Python/Gemini

## 🏛️ Architecture Overview

### High-Level Architecture
The solution consists of the following components:
1. **Frontend**: A web-based interface built with Flask for uploading files, viewing reports, and downloading results.
2. **Backend**: A Python-based backend that processes uploaded files, extracts rules, and analyzes transactions.
3. **AI Integration**: Utilizes Google's Gemini AI model for rule extraction and validation.
4. **Data Processing**: Uses libraries like `pandas` for data manipulation and `pdfplumber` for PDF parsing.
5. **Storage**: Stores intermediate and final results in JSON and Excel formats.

### Workflow
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

## 📝 Example Use Case
1. A business uploads a regulatory document (e.g., "Guidelines for Securities") and a CSV file containing transaction data.
2. The system extracts rules from the "Securities" schedule in the document.
3. The uploaded transactions are analyzed against these rules.
4. The system identifies violations (e.g., missing fields, invalid values) and assigns risk scores.
5. A detailed report is generated, highlighting issues and providing remediation suggestions.

---

## 📝 Additional Details
- **Tested Schedule**: The application has been tested with the "B—Securities" schedule but is compatible with other schedules as well. Ensure the schedule name matches one from the provided guideline PDF.
- **Input Requirements**:
  - **Guideline PDF**: Contains the regulatory schedules and rules.
  - **Transaction Data**: A CSV file with transaction details.
  - **Schedule Name**: The name of the schedule to be analyzed (must match a schedule in the PDF).
- **Processing Time**: The application uses free API keys, which may result in slower processing. For faster results, use your own Gemini API keys as described above.
- **Output**: A downloadable report containing remediation risks, violations, anomalies, and other insights.

- You can use sample input files from `test` directory

## 👥 Team
- **Sirisha** - #Manager
- **Aksh** - [GitHub](https://github.com/akshsharma4072) 
- **Priyanka**
- **Vaishnavi**
