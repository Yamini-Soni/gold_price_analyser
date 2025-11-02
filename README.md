# 🥇 Gold Price Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=Streamlit&logoColor=white)

A Streamlit application that tracks and predicts gold prices across multiple countries using Yahoo Finance data and basic machine learning.

---

## ✨ Features

- 🔔 Real-time gold prices for:
  - 🇺🇸 USA (USD/oz)
  - 🇮🇳 India (INR/10g)
  - 🇦🇪 Dubai (AED/oz)
- 📈 10-day gold price predictions using Random Forest
- 📰 Basic news sentiment analysis using NewsAPI
- 💱 Currency conversion between USD, INR, and AED
- 📊 Interactive Plotly charts for historical trends

---

## 🛠️ Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/gold-price-predictor.git
cd gold-price-predictor
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Create a .env File
Create a .env file in the project root and add the following keys:

ini
Copy
Edit
NEWS_API_KEY="your_key_here"

EXCHANGE_API_KEY="your_key_here "    # Optional - for currency conversion
🚀 Usage
Run the Streamlit application:

bash
Copy
Edit
streamlit run gold_price_analyzer.py
📊 Data Sources
📉 Yahoo Finance – Primary gold price data (free)

📰 NewsAPI – Market news articles (free tier available)

💱 ExchangeRate-API – Currency conversion rates (free tier available)

⚠️ Limitations
Predictions use basic indicators like SMA and RSI

Historical data is limited to ~90 days (due to Yahoo Finance constraints)

Sentiment analysis is basic, using TextBlob only

📂 Project Structure
bash
Copy
Edit
gold-price-predictor/
├── gold_price_analyzer.py   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignores .env and cache files
└── README.md                # Project documentation
