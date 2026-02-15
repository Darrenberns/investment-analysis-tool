# Investment Analysis Tool - Complete Setup Guide

## 📋 Overview

This guide walks you through:
1. Installing Python on Windows
2. Setting up the project locally
3. Running the app on your computer
4. Deploying to Streamlit Cloud (free hosting)

---

## Part 1: Local Windows Setup

### Step 1: Install Python (if not already installed)

1. **Download Python**
   - Go to https://www.python.org/downloads/
   - Download Python 3.11 or 3.12 (recommended)
   - Click "Download Python 3.11.x"

2. **Install Python**
   - Run the downloaded installer
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH" at the bottom
   - Click "Install Now"
   - Wait for installation to complete
   - Click "Close"

3. **Verify Installation**
   - Open Command Prompt (press `Win + R`, type `cmd`, press Enter)
   - Type: `python --version`
   - You should see: `Python 3.11.x` or similar
   - Type: `pip --version`
   - You should see pip version information

### Step 2: Create Project Folder

1. **Create a folder for your project**
   - Open File Explorer
   - Navigate to where you want your project (e.g., `C:\Users\YourName\Documents\`)
   - Right-click → New → Folder
   - Name it: `investment-tool`

2. **Download the project files**
   - You'll need these files in your `investment-tool` folder:
     - `investment_app.py` (the main Streamlit app)
     - `dcf_engine.py` (the calculation engine)
     - `monte_carlo_engine.py` (Monte Carlo simulation)
     - `requirements.txt` (dependencies)

### Step 3: Set Up Virtual Environment (Recommended)

1. **Open Command Prompt in your project folder**
   - Navigate to your folder in File Explorer
   - Click in the address bar, type `cmd`, press Enter
   - OR: Right-click in the folder while holding Shift → "Open PowerShell window here"

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```
   This creates a `venv` folder in your project

3. **Activate virtual environment**
   
   For Command Prompt:
   ```bash
   venv\Scripts\activate
   ```
   
   For PowerShell:
   ```bash
   venv\Scripts\Activate.ps1
   ```
   
   If PowerShell gives an error about execution policy, run:
   ```bash
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   Then try activating again.

   You should see `(venv)` at the beginning of your command line.

### Step 4: Install Dependencies

With your virtual environment activated, run:

```bash
pip install -r requirements.txt
```

This installs:
- Streamlit (the web framework)
- Pandas (data handling)
- NumPy (numerical calculations)
- SciPy (scientific calculations)
- Plotly (interactive charts)

Wait for installation to complete (may take 2-5 minutes).

### Step 5: Run the App Locally

1. **Start the Streamlit app**
   ```bash
   streamlit run investment_app.py
   ```

2. **Access the app**
   - Your default web browser should open automatically
   - If not, look for a URL in the terminal like: `http://localhost:8501`
   - Open that URL in your browser
   - 🎉 Your app is running!

3. **Stop the app**
   - Press `Ctrl + C` in the Command Prompt to stop the server
   - To run again later: activate venv and run `streamlit run investment_app.py`

### Troubleshooting Windows Installation

**Problem: "Python is not recognized"**
- Solution: Python not in PATH. Reinstall Python and check "Add Python to PATH"

**Problem: "pip is not recognized"**
- Solution: Run `python -m pip install --upgrade pip`

**Problem: PowerShell script execution error**
- Solution: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

**Problem: Module not found errors**
- Solution: Make sure virtual environment is activated (you see `(venv)`)
- Run `pip install -r requirements.txt` again

**Problem: Port 8501 already in use**
- Solution: Stop other Streamlit apps or use: `streamlit run investment_app.py --server.port 8502`

---

## Part 2: Deploy to Streamlit Cloud (Free Hosting)

Deploy your app for free so anyone can access it via a URL!

### Step 1: Create GitHub Account (if needed)

1. Go to https://github.com/
2. Click "Sign up"
3. Create a free account

### Step 2: Create a GitHub Repository

1. **Log in to GitHub**
2. **Click the "+" icon** (top right) → "New repository"
3. **Repository settings:**
   - Repository name: `investment-analysis-tool`
   - Description: "Professional investment analysis with DCF and Monte Carlo"
   - Public (so Streamlit Cloud can access it)
   - Check "Add a README file"
4. **Click "Create repository"**

### Step 3: Upload Your Files to GitHub

**Option A: GitHub Web Interface (Easier)**

1. In your new repository, click "Add file" → "Upload files"
2. Drag and drop these files from your computer:
   - `investment_app.py`
   - `dcf_engine.py`
   - `monte_carlo_engine.py`
   - `requirements.txt`
3. Click "Commit changes"

**Option B: Git Command Line (More Professional)**

1. **Install Git for Windows**
   - Download from: https://git-scm.com/download/win
   - Run installer with default settings

2. **Configure Git** (first time only)
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **In your project folder** (Command Prompt):
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/investment-analysis-tool.git
   git push -u origin main
   ```
   
   Replace `YOUR_USERNAME` with your GitHub username.
   You'll be prompted for your GitHub credentials.

### Step 4: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit: https://streamlit.io/cloud
   - Click "Sign up" or "Sign in with GitHub"
   - Authorize Streamlit to access your GitHub

2. **Deploy your app**
   - Click "New app"
   - **Repository:** Select `your-username/investment-analysis-tool`
   - **Branch:** main
   - **Main file path:** `investment_app.py`
   - Click "Deploy!"

3. **Wait for deployment** (2-5 minutes)
   - Streamlit Cloud will install dependencies
   - You'll see build logs
   - When complete, your app will launch!

4. **Access your app**
   - You'll get a URL like: `https://your-username-investment-analysis-tool.streamlit.app`
   - Share this URL with anyone!
   - Your app is now live 24/7 for free!

### Step 5: Update Your Deployed App

When you make changes:

**Option A: GitHub Web Interface**
1. Go to your GitHub repository
2. Click on the file you want to edit
3. Click the pencil icon (Edit)
4. Make changes
5. Click "Commit changes"
6. Streamlit Cloud auto-updates in ~1 minute!

**Option B: Git Command Line**
```bash
git add .
git commit -m "Description of changes"
git push
```
Streamlit Cloud auto-deploys your changes!

---

## Part 3: Using the App

### Creating Your First Investment

1. **Navigate to "➕ New Investment"**
2. **Fill in details:**
   - Name: "New Product Launch"
   - Description: "Launch of Widget 2.0"
   - Initial Investment: -500000 (negative!)
   - Discount Rate: 10%
3. **Add cash flows:**
   - Year 1: $150,000
   - Year 2: $200,000
   - Year 3: $220,000
   - Year 4: $250,000
   - Year 5: $280,000
4. **Click "Create Investment"**

### Analyzing the Investment

1. Go to "📊 Analyze Investment"
2. Select your investment
3. Review:
   - NPV, IRR, Profitability Index
   - Cash flow charts
   - Detailed breakdown table

### Running Monte Carlo Simulation

1. Go to "🎲 Monte Carlo Simulation"
2. Select your investment
3. Configure:
   - Distribution type: Normal or Triangular
   - Uncertainty: 20% (how much cash flows vary)
   - Simulations: 10,000
4. Click "Run Simulation"
5. Review risk metrics and probability distributions

### Comparing Investments

1. Create 2+ investments
2. Go to "⚖️ Compare Investments"
3. Select ranking metric (NPV, IRR, or PI)
4. View rankings and comparison charts

---

## Part 4: Advanced Configuration

### Customizing the App

Edit `investment_app.py` to customize:

**Change theme colors:**
```python
st.set_page_config(
    page_title="Your Company - Investment Tool",
    page_icon="🏢",
    # ... other settings
)
```

**Add company logo:**
```python
st.sidebar.image("logo.png", width=200)
```

**Change default values:**
```python
discount_rate = st.slider(
    "Discount Rate (%)",
    value=12.0,  # Change default here
    # ...
)
```

### Environment Variables (for sensitive data)

Create `.streamlit/secrets.toml` for API keys or passwords:

```toml
[general]
company_name = "Your Company"
admin_email = "admin@company.com"
```

Access in code:
```python
company = st.secrets["general"]["company_name"]
```

Note: Never commit `secrets.toml` to GitHub!

### Performance Tips

For large simulations:
```python
@st.cache_data  # Add caching
def run_expensive_calculation(params):
    # ... calculation
    return results
```

---

## Part 5: Common Issues & Solutions

### Local Issues

**App won't start:**
- Check Python is in PATH
- Verify virtual environment is activated
- Run `pip install -r requirements.txt` again

**Import errors:**
- Ensure all 3 Python files are in same folder
- Check file names match exactly (case-sensitive)

**Charts not displaying:**
- Update Plotly: `pip install --upgrade plotly`
- Clear browser cache

### Deployment Issues

**Build fails on Streamlit Cloud:**
- Check `requirements.txt` is correct
- Verify all Python files are uploaded
- Check build logs for specific errors

**App crashes after deployment:**
- Check file paths (use relative paths)
- Review Streamlit Cloud logs (click "Manage app" → "Logs")

**Changes not appearing:**
- Wait 2-3 minutes for auto-deploy
- Click "Reboot app" in Streamlit Cloud dashboard
- Clear browser cache

---

## Part 6: Next Steps

### Enhancements to Consider

1. **Add Data Export**
   ```python
   import io
   
   buffer = io.BytesIO()
   df.to_excel(buffer)
   st.download_button("Download Report", buffer, "report.xlsx")
   ```

2. **Add Authentication** (Streamlit Cloud)
   - Enable in app settings
   - Restrict access by email domain

3. **Connect to Database**
   - Use Streamlit's database connections
   - Store investments permanently

4. **Add More Analysis**
   - Scenario analysis
   - Break-even analysis
   - Real options valuation

### Resources

- **Streamlit Docs:** https://docs.streamlit.io
- **Plotly Charts:** https://plotly.com/python/
- **Python Finance:** https://pypi.org/project/pandas/
- **Community:** https://discuss.streamlit.io

---

## Quick Reference

### Daily Usage Commands

```bash
# Activate environment
venv\Scripts\activate

# Run app
streamlit run investment_app.py

# Install new package
pip install package-name
pip freeze > requirements.txt

# Update deployed app
git add .
git commit -m "Update description"
git push
```

### File Structure

```
investment-tool/
├── venv/                      # Virtual environment (don't upload)
├── investment_app.py          # Main Streamlit app
├── dcf_engine.py             # DCF calculations
├── monte_carlo_engine.py     # Monte Carlo simulations
├── requirements.txt          # Python dependencies
├── .streamlit/               # Optional config folder
│   └── secrets.toml          # Secrets (don't upload to GitHub!)
└── README.md                 # Project documentation
```

---

## Support

**Need help?**
- Check the Troubleshooting sections above
- Review Streamlit documentation
- GitHub Issues: Create issue in your repository
- Streamlit Community: https://discuss.streamlit.io

---

## Summary

✅ **Local Setup**: Python → Virtual Env → Install Packages → Run App  
✅ **Deployment**: GitHub Repository → Streamlit Cloud → Auto-Deploy  
✅ **Usage**: Create Investments → Analyze → Compare → Simulate  

**Your app is now professional-grade and accessible to anyone!**

---

*Last Updated: February 2026*
