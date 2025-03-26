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
GAIDP Fin Buddy is a financial assistant application designed to help users manage their personal finances effectively. It addresses the problem of financial illiteracy and provides tools for budgeting, expense tracking, and financial goal setting.

## 🎥 Demo
🔗 [Live Demo](https://example.com)  
📹 [Video Demo](https://example.com/video-demo)  
🖼️ Screenshots:

![Screenshot 1](https://example.com/screenshot1.png)

## 💡 Inspiration
The inspiration for GAIDP Fin Buddy came from the need to simplify personal finance management for individuals who struggle with budgeting and financial planning. We aim to empower users to make informed financial decisions.

## ⚙️ What It Does
- Generates insightful reports and analytics
- **Analyzes financial transactions based on regulatory schedules**  
  - Accepts a guideline PDF, transaction data (CSV format), and a schedule name as input.
  - Extracts data profile rules from the specified schedule in the PDF.
  - Executes these rules on the transaction data to identify remediation risks, violations, anomalies, etc.
  - Generates a detailed report that can be downloaded from the UI.

## 🛠️ How We Built It
- Frontend: HTML/CSS/JS for a responsive and user-friendly interface
- Backend: FastAPI for robust and scalable APIs
- Database: File System for secure and efficient data storage
- Other: OpenAI API and unsuprevised ML model

## 🚧 Challenges We Faced
- Integrating real-time bank account data securely
- Ensuring data privacy and compliance with financial regulations
- Designing an intuitive user interface for non-technical users

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
