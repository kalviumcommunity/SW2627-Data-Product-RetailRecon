
# Stock Sync

## Project Description

Stock Sync is an inventory analytics project that helps inventory managers identify stock shortages, excess inventory, sales trends, and return patterns across regional stores using an interactive dashboard.

## Learning Guides

- [SQL Filtering Guide](SQL_FILTERING_GUIDE.md) - a focused lesson on `WHERE`, `GROUP BY`, `HAVING`, and `ORDER BY` for operational reporting.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kalviumcommunity/SW2627-Data-Product-RetailRecon.git
 cd SW2627-Data-Product-RetailRecon
```

### 2. Create Virtual Environment

macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Project Structure

- **data/raw/** - Original inventory datasets.
- **data/processed/** - Cleaned datasets ready for analysis.
- **notebooks/** - Jupyter notebooks for EDA.
- **scripts/** - Python scripts for data processing.
- **output/** - Generated reports and charts.

---

## Notes

Copy `.env.example` to `.env` and replace the placeholder values with your own credentials. Never commit the `.env` file to Git.