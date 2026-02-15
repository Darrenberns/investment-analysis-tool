# Quick Start Guide - Investment Analysis Tool

⚡ **Get running in 5 minutes!**

## Local Windows Setup (Fast Track)

### 1. Install Python
```
Download: https://www.python.org/downloads/
✅ Check "Add Python to PATH" during installation
```

### 2. Setup Project
```bash
# Create folder
mkdir investment-tool
cd investment-tool

# Download files (place in this folder):
# - investment_app.py
# - dcf_engine.py
# - monte_carlo_engine.py
# - requirements.txt

# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Run the app
streamlit run investment_app.py
```

**That's it!** App opens at http://localhost:8501

---

## Deploy to Cloud (Free Hosting)

### 1. Create GitHub Repository
```
1. Go to github.com → Sign up/Sign in
2. Click "+" → "New repository"
3. Name: "investment-tool"
4. Make it Public
5. Click "Create repository"
```

### 2. Upload Files
```
Drag and drop to GitHub:
- investment_app.py
- dcf_engine.py
- monte_carlo_engine.py
- requirements.txt
```

### 3. Deploy to Streamlit Cloud
```
1. Go to streamlit.io/cloud
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: investment_app.py
6. Click "Deploy"
```

**Done!** Share your URL with anyone.

---

## First Investment Example

```
Name: "Product Launch"
Initial Investment: -500000
Discount Rate: 10%

Cash Flows:
Year 1: 150000
Year 2: 200000
Year 3: 220000
Year 4: 250000
Year 5: 280000
```

**Results:** NPV, IRR, charts, and more!

---

## Troubleshooting

**"Python not recognized"**
→ Reinstall Python with "Add to PATH" checked

**"Module not found"**
→ Activate venv: `venv\Scripts\activate`
→ Reinstall: `pip install -r requirements.txt`

**PowerShell error**
→ Run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`

**App won't deploy**
→ Check all 4 files uploaded to GitHub
→ Check requirements.txt syntax

---

## What's Included

✅ NPV, IRR, Profitability Index calculations
✅ Investment comparison and ranking
✅ Interactive cash flow charts
✅ Monte Carlo risk simulation
✅ Sensitivity analysis
✅ Professional metrics display

---

## Need More Help?

See **SETUP_INSTRUCTIONS.md** for detailed guide with screenshots and troubleshooting.

---

**Total Time:** 5 minutes local, 10 minutes cloud deployment
**Cost:** $0 (completely free)
