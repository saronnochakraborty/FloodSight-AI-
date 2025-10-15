# FloodSight — Live Flood & Rainfall Predictor

![Python](https://img.shields.io/badge/Python-3.13+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Yes-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**FloodSight** is an AI-powered application that provides **real-time rainfall and flood risk predictions** based on environmental and hydrological data.  
It features live-updating charts, prediction history, and risk alerts, all in a user-friendly interface.

---

## 🌦 Features

- Predict **rainfall (mm)** and **flood risk (%)** instantly  
- **Live charts** track historical predictions  
- Risk alerts: **Low**, **Moderate**, or **High**  
- **Standalone EXE included** in the release — download and run immediately  
- Optional CSV export of prediction history  

---

## ⚙️ How to Use

1. Download the latest release from the [GitHub Releases](https://github.com/saronnochakraborty/FloodSight/releases) page.  
2. Extract the ZIP if needed.  
3. Double-click **`FloodSight.exe`** — no installation or Python required.  
4. The app will open in your default browser.  
5. Input environmental data and view **real-time rainfall and flood risk predictions**.

> ✅ Fully standalone — plug-and-play.

---

## 📌 Optional Development Version

For users who want to run the app in development mode:

```bash
git clone https://github.com/saronnochakraborty/FloodSight.git
cd FloodSight/streamlit_app
python -m venv venv
# Activate venv
# Windows: venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

👤 Author
Saronno Chakraborty

📄 License
MIT License
