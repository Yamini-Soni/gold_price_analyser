
# Gold Price Analysis Dashboard

A Streamlit-based web application to analyze gold prices (XAU/USD) using real-time and historical data from GoldAPI.io, news sentiment from NewsAPI, and AI insights from Groq.

## Features
- **Real-Time Gold Price**: Display current gold spot price updated via GoldAPI.io.
- **Historical Gold Price Chart**: Visualize gold price movements with candlestick charts over 7-100 days.
- **Key Metrics**: View latest price, price change, trading volume, and historical highs.
- **News Sentiment Analysis**: Analyze gold-related news sentiment using TextBlob.
- **AI Insights**: Get market outlook, risks, and recommendations powered by Groq's AI.
- **Customizable Time Range**: Adjust data ranges for historical prices (7-100 days) and news (1-14 days).

## Prerequisites
- Python 3.8+
- API keys for:
  - [GoldAPI.io](https://www.goldapi.io/) (for real-time and historical gold price data)
  - [NewsAPI.org](https://newsapi.org/) (for news data)
  - [xAI Groq API](https://x.ai/api) (for AI analysis)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/gold-price-analyzer.git
   cd gold-price-analyzer


Create a virtual environment and activate it:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Create a .env file in the project root and add your API keys (use .env.example as a template):METALS_API_KEY=your_metals_api_key
NEWS_API_KEY=your_newsapi_key
GROQ_API_KEY=your_groq_api_key


Run the Streamlit app:streamlit run gold_price_analyzer.py



Usage

Open the app in your browser (default: http://localhost:8501).
Use the sidebar to configure:
Historical gold price data range (7 to 90 days).
News data range (1 to 14 days).


Click Analyze Gold Prices to fetch real-time and historical data, view charts, sentiment analysis, and AI insights.

Project Structure
gold-price-analyzer/
├── gold_price_analyzer.py  # Main Streamlit app
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
├── .env.example          # Template for environment variables
└── .env                  # API keys (not tracked by Git)

Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/your-feature).
Commit changes (git commit -m 'Add your feature').
Push to the branch (git push origin feature/your-feature).
Open a Pull Request.

License
This project is licensed under the MIT License. See LICENSE for details.
Notes

Ensure API keys are kept secure and not committed to version control.
The free tier of Metals-API has a limit of 100 requests/month; consider upgrading for heavy usage.
Contact API providers for support with key generation or access issues.
Historical data in the free tier may lack OHLC (open, high, low, close) details; paid plans offer more granularity.


