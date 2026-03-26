## Climate Becky Website
This is a multi-page Dash web application that includes interactive visualizations of climate data.<br>
See the full website at https://climatebecky.com

### 📌 Project Overview
Originally developed as a standalone "Climate at a Glance" dashboard, this project has evolved into a modular, multi-page suite. It leverages Python, Dash, and Plotly for data dashboard displays.

### 🚀 Features

**Climate at a Glance (CAG)**: Interactive dashboard for analyzing long-term temperature and precipitation trends, anomalies, and extremes for the U.S.  
**Snowfall Tracker**: Real-time visualization of monthly snowfall totals compared to normal, with station-specific time-series analysis and state-level filtering.  
**Modular Architecture**: Uses Dash’s multi-page registration to keep analytical logic and UI components separated.

### 🛠️ Tech Stack

* **Language**: Python 3.x
* **Framework**: Dash / Plotly (for interactive UI and charting)
* **Data Processing**: Pandas / NumPy
* **Version Control**: Git (workflow)

### 📂 Repository Structure

```python
climate_dashboard/
├── app.py                # Main entry point; handles multi-page routing
├── requirements.txt      # Dependency list
├── .gitignore            # Clean repo management
└── pages/
    ├── cag_dashboard.py  # Climate at a Glance dashboard
    ├── snow_dashboard.py # Snowfall visualization tool
    └── *.py              # Remaining pages are for the website, including homepage, portfolio, and about me
```

---

### ⚙️ Installation & Local Setup

1. Clone the repository:

```python
git clone https://github.com/beckybol/climate_dashboard.git
```

2. Install dependencies:

```python
pip install -r requirements.txt
```

3. Run the app:

```python
python app.py
```