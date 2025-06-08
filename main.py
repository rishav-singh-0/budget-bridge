
# def main():
#     print("Hello from cashew-banker!")


# if __name__ == "__main__":
#     main()


# CashewBanker: Convert Bank Statements to Cashew Format

import streamlit as st
import pandas as pd
import json
import os
import google.generativeai as genai
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini API
def initialize_gemini(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.0-flash')

# Process file content
def process_file_content(content):
    return f"""
Convert this bank transaction data into Cashew budget app format with these columns:
Date, Amount, Category, Title, Note, Account

Follow these rules:
1. Format dates as YYYY-MM-DD
2. Make expenses negative and deposits positive
3. Infer categories from transaction descriptions
4. For account, use 'Bank' unless specified
5. Keep titles concise (max 3-4 words)
6. Amount should be numerical values only
7. For missing values, use empty strings

BANK STATEMENT DATA:
{content}

Output ONLY valid JSON in this format:
[
  {{"Date": "2025-01-14", "Amount": -50.0, "Category": "Groceries", "Title": "Fruits Vegetables", "Note": "Paid with cash", "Account": "Wallet"}},
  {{"Date": "2025-01-14", "Amount": 250.0, "Category": "Income", "Title": "Monthly Income", "Note": "", "Account": "Wallet"}}
]
"""

# Convert to Cashew format
def convert_to_cashew(content, model):
    prompt = process_file_content(content)
    response = model.generate_content(prompt)
    
    try:
        # Clean Gemini response
        json_str = response.text.replace('```json', '').replace('```', '').strip()
        transactions = json.loads(json_str)
        return pd.DataFrame(transactions)
    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        st.text(f"Gemini response: {response.text}")
        return pd.DataFrame()

# Streamlit app
st.title("Bank Statement to Cashew Converter")
st.markdown("Convert PDF/Excel/CSV bank statements to Cashew format")

# File upload
uploaded_file = st.file_uploader("Upload Bank Statement", type=["pdf", "csv", "xlsx", "xls"])
if not uploaded_file:
    st.info("ℹ️ Upload a bank statement file (PDF, CSV, Excel)")
    st.stop()

try:
    # Process different file types
    content = ""
    if uploaded_file.type == "application/pdf":
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            content = "\n".join(page.extract_text() for page in pdf.pages)
    
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        content = df.to_string()
    
    elif "spreadsheet" in uploaded_file.type:
        df = pd.read_excel(uploaded_file)
        content = df.to_string()
    
    # Initialize Gemini and convert
    model = initialize_gemini(GEMINI_API_KEY)
    cashew_df = convert_to_cashew(content, model)
    
    # Display results
    if not cashew_df.empty:
        st.subheader("Converted Transactions")
        st.dataframe(cashew_df)
        
        # Download functionality
        csv = cashew_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="cashew_transactions.csv",
            mime="text/csv"
        )
        
except Exception as e:
    st.error(f"Error processing file: {str(e)}")
