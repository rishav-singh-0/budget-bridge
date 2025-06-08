# Budget Bridge

BudgetBridge is a lightweight, Streamlit‑powered utility that transforms raw bank statements (PDF, Excel or CSV) into ready‑to‑import budget entries. Leveraging Google’s Gemini text‑generation API, it:
- Parses diverse bank formats into a unified table
- Auto‑categorizes each transaction based on its description
- Normalizes dates, amounts (withdrawals as negatives, deposits as positives), and metadata (title, note, account)
- Displays an interactive preview within your browser
- Exports a clean CSV file compatible with your favorite budgeting tool

With BudgetBridge, you can go from messy statements to organized financial data in seconds—no manual copy‑pasting, no error‑prone spreadsheets, and zero friction between your bank and your budget.

## Running

> Note: Make sure to add your `GEMINI_API_KEY` to `.env` file.

### Using `uv`
```bash
uv run main.py
```
