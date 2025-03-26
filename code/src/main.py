from flask import Flask, request, render_template, send_file
import os
import pandas as pd
import logging
from process import Processor, save_json, load_json
import re

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

app = Flask(__name__)
UPLOAD_FOLDER = os.path.abspath('uploads')  # Convert to absolute path
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

processor = Processor()

@app.route('/')
def index():
    logging.info("Accessed index page.")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    logging.info("Upload endpoint hit.")
    if 'pdf_file' not in request.files or 'transaction_file' not in request.files or not request.form.get('schedule_name'):
        logging.error("Missing required files or schedule name.")
        return "Error: Please upload a PDF file, a transaction CSV file, and provide a schedule name.", 400

    pdf_file = request.files['pdf_file']
    transaction_file = request.files['transaction_file']
    schedule_name = request.form['schedule_name']

    if pdf_file.filename == '' or transaction_file.filename == '':
        logging.error("No file selected.")
        return "Error: No file selected.", 400

    pdf_path = os.path.join(UPLOAD_FOLDER, pdf_file.filename)
    transaction_path = os.path.join(UPLOAD_FOLDER, transaction_file.filename)
    pdf_file.save(pdf_path)
    transaction_file.save(transaction_path)

    try:
        logging.info(f"Processing schedule: {schedule_name}")
        results = processor.process_schedule(pdf_path, schedule_name)
        if results and results['rules']:
            save_json(results['rules'], os.path.join(UPLOAD_FOLDER, 'extracted_rules.json'))
            logging.info(f"Extracted {len(results['rules'])} rules.")
        else:
            logging.error("Failed to extract rules or schedule not found.")
            return "Error: Failed to extract rules or schedule not found.", 400

        logging.info("Loading transactions.")
        transactions = pd.read_csv(transaction_path).to_dict(orient="records")
        logging.info(f"Loaded {len(transactions)} transactions.")

        logging.info("Analyzing transactions.")
        analysis_df = processor.analyze_transactions(transactions, results['rules'])

        # Save analysis_df to JSON for the unified table
        analysis_path = os.path.join(UPLOAD_FOLDER, 'analysis_report.json')
        analysis_df.to_json(analysis_path, orient="records", indent=4)

        # Save analysis_df to Excel
        excel_path = os.path.join(UPLOAD_FOLDER, 'analysis_report.xlsx')
        analysis_df.to_excel(excel_path, index=False)
        logging.info("Analysis complete. Results saved.")

        return render_template(
            'report.html',
            transactions=analysis_df.to_dict(orient="records"),  # Pass analysis_df as transactions
            rules=results['rules']  # Pass extracted rules to the template
        )
    except Exception as e:
        logging.exception(f"Error during processing: {e}")
        return f"Error: {str(e)}", 500

@app.route('/report')
def report():
    try:
        # Load extracted rules and analysis report from the uploads directory
        rules_path = os.path.join(UPLOAD_FOLDER, 'extracted_rules.json')
        analysis_path = os.path.join(UPLOAD_FOLDER, 'analysis_report.json')

        if not os.path.exists(rules_path) or not os.path.exists(analysis_path):
            logging.error("Required JSON files not found in uploads directory.")
            return "Error: Required files not found. Please upload files first.", 400

        logging.info("Loading extracted rules and analysis report.")
        rules = load_json(rules_path)
        transaction_data = pd.read_json(analysis_path).to_dict(orient="records")

        return render_template(
            'report.html',
            transactions=transaction_data,
            rules=rules  # Pass extracted rules to the template
        )
    except Exception as e:
        logging.exception(f"Error loading report: {e}")
        return f"Error: {str(e)}", 500

@app.route('/download/<filename>')
def download(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(file_path):
        logging.info(f"File downloaded: {filename}")
        return send_file(file_path, as_attachment=True)
    logging.error(f"File not found: {filename}")
    return "Error: File not found.", 404

if __name__ == '__main__':
    logging.info("Starting Flask server.")
    app.run(debug=True)
