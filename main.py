import streamlit as st
import pandas as pd
import json
import google.generativeai as genai
from io import BytesIO
import re
import traceback
import os
from dotenv import load_dotenv
from datetime import datetime
from fuzzywuzzy import fuzz, process

# Load environment variables
load_dotenv()

# Initialize Gemini API
def initialize_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in .env file")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-pro')

# Process file content with reference knowledge
def process_file_content(content, reference_data=None):
    prompt = f"""
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
8. Mark category as "Uncategorized" if uncertain
9. Output ONLY JSON format - no additional text
"""
    # Add reference knowledge if available
    if reference_data is not None:
        prompt += f"""
Use the following existing transaction categories and mappings as reference:
{reference_data}
"""
    
    prompt += f"""
BANK STATEMENT DATA:
{content}

Output ONLY valid JSON in this format:
[
  {{"Date": "2025-01-14", "Amount": -50.0, "Category": "Groceries", "Title": "Fruits Vegetables", "Note": "Paid with cash", "Account": "Wallet"}},
  {{"Date": "2025-01-14", "Amount": 250.0, "Category": "Income", "Title": "Monthly Income", "Note": "", "Account": "Wallet"}}
]
"""
    return prompt

# Improved JSON extraction with multiple fallbacks
def extract_json(response):
    try:
        # Clean the response
        cleaned = re.sub(r'[\s\n]*```(json)?[\s\n]*', '', response).strip()
        
        # First try: Direct JSON parse
        try:
            return json.loads(cleaned)
        except:
            pass
            
        # Second try: Find JSON array pattern
        array_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except:
                pass
                
        # Third try: Find all JSON objects and combine
        obj_matches = re.findall(r'\{\s*".*?"\s*:\s*".*?"\s*(?:,\s*".*?"\s*:\s*".*?"\s*)*\}', response)
        if obj_matches:
            try:
                transactions = [json.loads(match) for match in obj_matches]
                return transactions
            except:
                pass
                
        # Final try: Manual extraction of key-value pairs
        transactions = []
        lines = response.split('\n')
        current = {}
        for line in lines:
            if line.strip() == '[' or line.strip() == ']' or not line.strip():
                continue
                
            # Try to parse as JSON object
            if line.strip().startswith('{') and line.strip().endswith('}'):
                try:
                    transactions.append(json.loads(line.strip()))
                except:
                    pass
                continue
                
            # Manual key-value extraction
            match = re.search(r'"([^"]+)":\s*"([^"]*)"', line)
            if match:
                key, value = match.groups()
                current[key] = value
            elif '}' in line and current:
                transactions.append(current)
                current = {}
                
        if transactions:
            return transactions
            
    except Exception as e:
        st.error(f"JSON extraction error: {str(e)}")
    
    return None

# Convert Gemini response to DataFrame
def parse_gemini_response(response):
    try:
        transactions = extract_json(response)
        
        if not transactions:
            st.error("No valid transactions found in response")
            return pd.DataFrame()
            
        # Convert to DataFrame with proper types
        df = pd.DataFrame(transactions)
        
        # Validate required columns
        required_columns = ['Date', 'Amount', 'Title']
        for col in required_columns:
            if col not in df.columns:
                st.error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        # Type conversion
        if 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        
        # Fill missing values
        if 'Category' not in df.columns:
            df['Category'] = 'Uncategorized'
        if 'Note' not in df.columns:
            df['Note'] = ''
        if 'Account' not in df.columns:
            df['Account'] = 'Bank'
            
        return df
        
    except Exception as e:
        st.error(f"Error parsing response: {str(e)}")
        return pd.DataFrame()

# Create reference knowledge from uploaded CSV
def create_reference_knowledge(reference_df):
    # Extract unique category mappings
    if 'Category' not in reference_df.columns:
        return None
        
    knowledge = []
    
    # Get most common titles per category
    category_mappings = reference_df.groupby('Category')['Title'].apply(
        lambda x: x.value_counts().index[0] if not x.empty else None
    ).dropna().to_dict()
    
    # Format for prompt
    knowledge.append("Existing Category Mappings:")
    for category, title in category_mappings.items():
        knowledge.append(f"- {category}: Transactions like '{title}'")
    
    # Get unique categories
    unique_categories = reference_df['Category'].unique()
    knowledge.append("\nAvailable Categories:")
    knowledge.append(", ".join(unique_categories))
    
    # Get subcategories if available
    if 'Subcategory' in reference_df.columns:
        subcategory_mappings = reference_df.groupby(['Category', 'Subcategory'])['Title'].first().dropna()
        knowledge.append("\nSubcategory Mappings:")
        for (category, subcategory), title in subcategory_mappings.items():
            knowledge.append(f"- {category} > {subcategory}: Example '{title}'")
    
    return "\n".join(knowledge)

# Fuzzy match titles to reference categories
def fuzzy_match_to_reference(transaction_title, reference_df, threshold=80):
    if reference_df is None or 'Title' not in reference_df.columns or 'Category' not in reference_df.columns:
        return None
    
    # Find best match in reference titles
    best_match, score = process.extractOne(
        transaction_title, 
        reference_df['Title'].dropna().unique(),
        scorer=fuzz.token_set_ratio
    )
    
    if score >= threshold:
        # Return the most common category for this title
        matches = reference_df[reference_df['Title'] == best_match]
        if not matches.empty:
            return matches.iloc[0]['Category']
    
    return None

# Process uploaded files
def process_uploaded_files(uploaded_files, model, reference_df=None):
    all_dfs = []
    reference_knowledge = create_reference_knowledge(reference_df) if reference_df is not None else None
    
    for uploaded_file in uploaded_files:
        try:
            content = ""
            file_type = uploaded_file.type
            
            if file_type == "application/pdf":
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    content = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            
            elif file_type == "text/csv":
                content = uploaded_file.read().decode("utf-8")
                uploaded_file.seek(0)  # Reset file pointer
            
            elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
                df = pd.read_excel(uploaded_file)
                content = df.to_string()
            
            if not content:
                st.warning(f"No content extracted from {uploaded_file.name}")
                continue
                
            # Limit content length to avoid token limits
            if len(content) > 30000:
                content = content[:30000]
                st.warning(f"Content truncated for {uploaded_file.name} due to length")
                
            prompt = process_file_content(content, reference_knowledge)
            
            # Display processing status
            with st.spinner(f"Processing {uploaded_file.name}..."):
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=2000
                    ),
                    safety_settings={
                        'HARASSMENT': 'block_none',
                        'HATE_SPEECH': 'block_none',
                        'SEXUAL': 'block_none',
                        'DANGEROUS': 'block_none'
                    }
                )
                
                # Display raw response for debugging
                with st.expander(f"View Raw Response for {uploaded_file.name}"):
                    st.code(response.text)
                
                df = parse_gemini_response(response.text)
                
                if not df.empty:
                    # Apply fuzzy matching to reference data
                    if reference_df is not None:
                        for idx, row in df.iterrows():
                            if row['Category'] == 'Uncategorized':
                                matched_category = fuzzy_match_to_reference(row['Title'], reference_df)
                                if matched_category:
                                    df.at[idx, 'Category'] = matched_category
                    
                    df['Source File'] = uploaded_file.name
                    all_dfs.append(df)
                else:
                    st.warning(f"No transactions extracted from {uploaded_file.name}")
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            st.error(traceback.format_exc())
    
    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

# Streamlit app
st.title("🏦 Bank Statement to Cashew Converter")
st.markdown("Convert bank statements to Cashew format using Gemini AI")

# Initialize Gemini model
model = initialize_gemini()

# Reference file upload
st.sidebar.header("Reference Data")
reference_file = st.sidebar.file_uploader(
    "Upload Cashew Reference CSV (Optional)", 
    type=["csv"],
    help="Upload a CSV export from Cashew to improve categorization accuracy"
)
reference_df = None
if reference_file:
    try:
        reference_df = pd.read_csv(reference_file)
        st.sidebar.success(f"Reference file loaded with {len(reference_df)} transactions")
        st.sidebar.dataframe(reference_df.head(3))
    except Exception as e:
        st.sidebar.error(f"Error loading reference file: {str(e)}")

# Main file upload - multiple files
st.header("Bank Statements")
uploaded_files = st.file_uploader(
    "Upload Bank Statements", 
    type=["pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("ℹ️ Upload one or more bank statement files (PDF, CSV, Excel)")
    st.stop()

# Process files
cashew_df = process_uploaded_files(uploaded_files, model, reference_df)

if cashew_df.empty:
    st.warning("No transactions found in any files")
    st.stop()

# Display raw data
st.subheader("Converted Transactions")
st.dataframe(cashew_df)

# Identify uncategorized transactions
if 'Category' in cashew_df.columns:
    uncategorized_df = cashew_df[
        (cashew_df['Category'].isna()) | 
        (cashew_df['Category'].str.lower().str.contains('uncategorized', na=True))
    ]
else:
    uncategorized_df = pd.DataFrame()

if not uncategorized_df.empty:
    st.subheader("Uncategorized Transactions")
    st.info("Please categorize the following transactions:")
    
    # Create editor with unique titles
    unique_titles = uncategorized_df['Title'].unique()
    title_mapping = {}
    
    with st.form("categorization_form"):
        for title in unique_titles:
            st.markdown(f"**{title}**")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                new_category = st.text_input(
                    f"Category for '{title}'",
                    key=f"cat_{title}",
                    placeholder="Enter category"
                )
            
            with col2:
                apply_to_all = st.checkbox(
                    "Apply to all similar",
                    key=f"apply_{title}",
                    value=True
                )
            
            title_mapping[title] = {
                'category': new_category,
                'apply_to_all': apply_to_all
            }
        
        submitted = st.form_submit_button("Apply Categorization")
    
    if submitted:
        # Apply categorization
        for title, mapping in title_mapping.items():
            if mapping['category']:
                if mapping['apply_to_all']:
                    # Apply to all transactions with this title
                    cashew_df.loc[cashew_df['Title'] == title, 'Category'] = mapping['category']
                else:
                    # Apply only to uncategorized transactions with this title
                    mask = (cashew_df['Title'] == title) & (
                        (cashew_df['Category'].isna()) | 
                        (cashew_df['Category'].str.lower().str.contains('uncategorized', na=True))
                    )
                    cashew_df.loc[mask, 'Category'] = mapping['category']
        
        st.success("Categorization applied!")
        st.experimental_rerun()

# Download functionality
st.subheader("Final Data")
st.dataframe(cashew_df)

csv = cashew_df.to_csv(index=False).encode('utf-8')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"cashew_transactions_{timestamp}.csv"

st.download_button(
    label="Download as CSV",
    data=csv,
    file_name=filename,
    mime="text/csv"
)
