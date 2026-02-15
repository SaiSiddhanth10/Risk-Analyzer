# Financial Risk Assessment Tool

A Python-based tool that automatically analyzes financial reports (Excel & PDF) and calculates key risk metrics for any company.

This project extracts important financial values such as profit, assets, equity, deposits, loans, and cash, then computes financial ratios and generates a risk assessment report with charts.

---

# Features

* Upload financial reports in Excel or PDF

* Automatic data extraction using fuzzy keyword matching

* Detect currency and scaling (Lakhs, Crores, Millions, etc.)

* Calculate important financial ratios:

  * Return on Assets (ROA)
  * Debt-to-Equity Ratio
  * Loan-to-Deposit Ratio
  * Liquidity Ratio
  * Equity Ratio

* Generate Composite Risk Score and Risk Level (Low / Medium / High)

* Create charts for:

  * Profit Trend
  * Debt-to-Equity Trend

---

# How It Works

1. User enters company name.
2. Upload financial reports.
3. Tool scans Excel/PDF tables.
4. Extracts financial values using fuzzy matching.
5. Calculates financial ratios.
6. Generates a risk report and charts.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* OpenPyXL
* pdfplumber
* Google Colab (for file upload and execution)

---

# Example Output

The tool generates a report including:

* Profit
* Total Assets
* Return on Assets (ROA)
* Debt-to-Equity Ratio
* Liquidity Ratio
* Risk Level

Charts are also created showing trends over time.

---

# Installation

Install dependencies:

pip install openpyxl pdfplumber matplotlib pandas numpy

Run the script in Google Colab or locally, upload financial reports, and view results.

---

# Future Improvements

* Add more financial ratios
* Support multiple companies comparison
* Build a web dashboard
* Add machine learning risk prediction

---

# Author

Sai
