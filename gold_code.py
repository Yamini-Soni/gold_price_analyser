import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
import os
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings('ignore')
from transformers import pipeline

# Load the Hugging Face sentiment analysis pipeline once
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# Load environment variables
load_dotenv()

# API Configuration - All FREE APIs
GOLDAPI_KEY = os.getenv('GOLDAPI_KEY')  # Free tier available
NEWS_API_KEY = os.getenv('NEWS_API_KEY')  # Free tier: newsapi.org
GROQ_API_KEY = os.getenv('GROQ_API_KEY')  # Free tier: console.groq.com
ALPHA_VANTAGE_KEY = os.getenv('ALPHA_VANTAGE_KEY')  # Free tier: alphavantage.co
EXCHANGE_API_KEY = os.getenv('EXCHANGE_API_KEY')  # Free tier: exchangerate-api.com

# Initialize Groq client
try:
    groq_client = Groq(api_key=["GROQ_API_KEY"]) 
except:
    groq_client = None

# Country-specific gold configurations
COUNTRY_CONFIG = {
    'USA': {
        'currency': 'USD',
        'symbol': 'GC=F',  # Gold futures
        'multiplier': 1.0,
        'unit': 'oz',
        'flag': '🇺🇸'
    },
    'India': {
        'currency': 'INR', 
        'symbol': 'GC=F',
        'multiplier': 88.79,  # USD to INR
        'unit': '10g',
        'flag': '🇮🇳'
    },
    'Dubai': {
        'currency': 'AED',
        'symbol': 'GC=F', 
        'multiplier': 3.67,  # USD to AED
        'unit': 'oz',
        'flag': '🇦🇪'
    }
}

class GoldPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, df):
        """Prepare features for ML model with proper NaN/inf handling"""
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['date', 'close']
        for col in required_cols:
            if col not in df.columns:
                if col == 'date':
                    df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
                elif col == 'close':
                    df['close'] = 2000  # Fallback price
        
        # Technical indicators with proper NaN handling
        df['sma_5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
        df['rsi'] = self.calculate_rsi(df['close'])
        df['volatility'] = df['close'].rolling(window=10, min_periods=1).std()
        df['price_change'] = df['close'].pct_change().replace([np.inf, -np.inf], np.nan)
        df['volume_change'] = df['volume'].pct_change().replace([np.inf, -np.inf], np.nan) if 'volume' in df.columns else 0
        
        # Time features
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        # Fill remaining NaNs with 0 and replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(df['volatility'].mean())
        
        return df
        
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator with proper NaN handling"""
        delta = prices.diff()
        
        # Make two series: one for gains and one for losses
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Fill initial NaN values with 50 (neutral RSI)
        return rsi.fillna(50)
    
    def train(self, df):
        """Train the prediction model with proper error handling"""
        try:
            df_features = self.prepare_features(df)
            
            # Define features for training
            feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility', 'price_change', 
                          'volume_change', 'hour', 'day_of_week']
            
            # Prepare training data
            X = df_features[feature_cols].iloc[:-1]
            y = df_features['close'].iloc[1:].values  # Predict next close price and convert pandas sequence to numpy arrays
            
            # Remove any remaining infinite values
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())
            
            if len(X) > 10:  # Minimum data requirement
                X_scaled = self.scaler.fit_transform(X)
                self.model.fit(X_scaled, y)
                self.is_trained = True
                return True
            return False

            
            

        except Exception as e:
            st.error(f"Training error: {e}")
            return False
    
    def evaluate(self, df):
        """Evaluate model accuracy (error metrics)"""
        if not self.is_trained:
            st.warning("Model not trained yet!")
            return None

        df_features = self.prepare_features(df)
        feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility',
                        'price_change', 'volume_change', 'hour', 'day_of_week']

        X = df_features[feature_cols].iloc[:-1].fillna(0)
        y_true = df_features['close'].iloc[1:].values

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print(f"📊 Model Evaluation:")
        print(f"MAE  : {mae:.2f}")
        print(f"MSE  : {mse:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R²   : {r2:.3f}")

        return mae, mse, rmse, r2


    def predict_next_prices(self, df, steps=10):
        """Predict next price movements with proper error handling"""
        if not self.is_trained:
            return None
        
        try:
            df_features = self.prepare_features(df)
            feature_cols = ['sma_5', 'sma_20', 'rsi', 'volatility', 'price_change',
                          'volume_change', 'hour', 'day_of_week']
            
            last_features = df_features[feature_cols].iloc[-1:].fillna(0)
            X_scaled = self.scaler.transform(last_features)
            
            predictions = []
            current_price = df['close'].iloc[-1]
            
            for i in range(steps):
                pred_price = self.model.predict(X_scaled)[0]
                predictions.append(pred_price)
                
                # Update features for next prediction (simplified approach)
                # Create a new row with the predicted price
                new_row = last_features.copy()
                new_row['price_change'] = (pred_price - current_price) / current_price if current_price != 0 else 0
                new_row['close'] = pred_price
                
                # Update technical indicators
                new_row['sma_5'] = (last_features['sma_5'].values[0] * 4 + pred_price) / 5
                new_row['sma_20'] = (last_features['sma_20'].values[0] * 19 + pred_price) / 20
                new_row['rsi'] = last_features['rsi'].values[0]  # Simplified - in reality should recalculate
                
                # Update for next iteration
                last_features = new_row[feature_cols]
                X_scaled = self.scaler.transform(last_features)
                current_price = pred_price
            
            return predictions
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

def get_exchange_rates():
    """Get current exchange rates (Free API)"""
    try:
        if EXCHANGE_API_KEY:
            url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/USD"
        else:
            # Fallback to free service (no key required but limited)
            url = "https://api.exchangerate-api.com/v4/latest/USD"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        return {
            'USD_INR': data['rates']['INR'],
            'USD_AED': data['rates']['AED']
        }
    except:
        # Fallback rates
        return {'USD_INR': 88.79, 'USD_AED': 3.67}

def get_multi_country_gold_data(days=30):
    """Fetch gold data for multiple countries using yfinance (FREE)"""
    try:
        # Get gold futures data
        gold_ticker = yf.Ticker("GC=F")
        hist = gold_ticker.history(period=f"{days}d")
        
        if hist.empty:
            st.error("No gold data available from Yahoo Finance")
            return None
        
        # Get exchange rates
        rates = get_exchange_rates()
        
        # Prepare data for each country
        country_data = {}
        
        for country, config in COUNTRY_CONFIG.items():
            df = hist.copy()
            df.reset_index(inplace=True)
            
            # Handle different column structures from yfinance
            expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            actual_columns = df.columns.tolist()
            
            # Map actual columns to expected columns
            column_mapping = {}
            for i, col in enumerate(expected_columns):
                if i < len(actual_columns):
                    column_mapping[actual_columns[i]] = col
            
            df = df.rename(columns=column_mapping)
            
            # Ensure we have all required columns
            for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                if col not in df.columns:
                    if col == 'Volume':
                        df[col] = 0  # Default volume
                    elif col == 'Date':
                        df[col] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
                    else:
                        df[col] = df.get('Close', 2000)  # Use close price as fallback
            
            # Select only the columns we need and rename to standard names
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            
            # Convert prices based on country
            if country == 'India':
                multiplier = rates['USD_INR'] / 28.3495 * 10  # Convert oz to 10g in INR
            elif country == 'Dubai':
                multiplier = rates['USD_AED']
            else:
                multiplier = 1.0
            
            # Apply multiplier to price columns
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df[col] = df[col] * multiplier
            
            df['country'] = country
            df['currency'] = config['currency']
            df['unit'] = config['unit']
            
            country_data[country] = df
        
        return country_data
    except Exception as e:
        st.error(f"Error fetching multi-country gold data: {e}")
        return None

def get_real_time_multi_country_prices():
    """Get real-time gold prices for all countries"""
    try:
        # Get current gold price
        gold_ticker = yf.Ticker("GC=F")
        hist = gold_ticker.history(period='1d')
        
        if hist.empty:
            current_price_usd = 2000  # Fallback price
        else:
            current_price_usd = hist['Close'].iloc[-1]
        
        rates = get_exchange_rates()
        
        prices = {}
        timestamp = datetime.now()
        
        for country, config in COUNTRY_CONFIG.items():
            if country == 'India':
                price = current_price_usd * rates['USD_INR'] / 28.3495 * 10
            elif country == 'Dubai':
                price = current_price_usd * rates['USD_AED']
            else:
                price = current_price_usd
            
            prices[country] = {
                'price': round(price, 2),
                'currency': config['currency'],
                'unit': config['unit'],
                'flag': config['flag'],
                'timestamp': timestamp
            }
        
        return prices
    except Exception as e:
        st.error(f"Error fetching real-time prices: {e}")
        return None

def get_news_data(days=20):
    """Fetch gold-related news from News API"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': 'gold price OR gold market OR XAU/USD OR gold investment',
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d'),
            'sortBy': 'publishedAt',
            'language': 'en',
            'apiKey': NEWS_API_KEY,
            'pageSize': 50
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'articles' in data:
            articles = []
            for article in data['articles']:
                if article['title'] and article['description']:
                    articles.append({
                        'title': article['title'],
                        'description': article['description'],
                        'url': article['url'],
                        'publishedAt': article['publishedAt'],
                        'source': article['source']['name']
                    })
            return articles
        return []
    except Exception as e:
        st.error(f"Error fetching news data: {e}")
        return []

def analyze_sentiment(text):
    """Analyze sentiment using Hugging Face DistilBERT"""
    try:
        # Truncate long text (BERT limit ~512 tokens)
        text = text[:512]

        # Run sentiment prediction
        result = sentiment_model(text)[0]  
        # Example: {'label': 'POSITIVE', 'score': 0.998}

        label = result['label'].capitalize()  # 'Positive' or 'Negative'
        polarity = round(result['score'], 3)  # Confidence score (0–1)

        return label, polarity

    except Exception as e:
        return 'Neutral', 0.0



def display_price_comparison(real_time_prices, show_inr_conversion=True):
    """Display global gold price comparison with INR conversion and reason for price differences."""

    if not real_time_prices:
        st.warning("No real-time gold price data available.")
        return

    st.subheader("🌍 Global Gold Price Comparison")

    rates = get_exchange_rates()
    usd_prices = {}
    inr_prices = {}

    # Convert all prices to USD and INR for fair comparison
    for country, data in real_time_prices.items():
        price = data['price']
        currency = data['currency']

        if country == 'India':
            # ₹/10g → USD/oz
            #usd_price = (price / rates['USD_INR']) * (31.1035 / 10)
            inr_price = price*31.1035 /10 # Already in INR
        elif country == 'Dubai':
            # AED/oz → USD/oz → INR/oz
            usd_price = price / rates['USD_AED']
            inr_price = (usd_price * rates['USD_INR'])
        elif country == 'USA':
            # USD/oz → INR/oz
            usd_price = price
            inr_price = price * rates['USD_INR']
        else:
            usd_price = price
            inr_price = price * rates['USD_INR']

        usd_prices[country] = usd_price
        inr_prices[country] = inr_price

    # Find cheapest and most expensive
    cheapest = min(inr_prices, key=inr_prices.get)
    most_expensive = max(inr_prices, key=inr_prices.get)

    # Display in three columns
    cols = st.columns(min(3, len(real_time_prices)))

    for i, (country, data) in enumerate(real_time_prices.items()):
        with cols[i % 3]:
            delta_text = ""
            delta_color = "normal"

            if country == cheapest:
                delta_text = "🏆 Cheapest"
                delta_color = "inverse"
            elif country == most_expensive:
                delta_text = "💎 Most Expensive"

            # Display main price metric
            st.metric(
                f"{data['flag']} {country}",
                f"{data['price']:.2f} {data['currency']}/{data['unit']}",
                delta_text,
                delta_color=delta_color
            )

            # Always show equivalent INR values
            st.caption(f"≈ ₹{inr_prices[country]:,.2f} per oz")

            # Also show conversion to 10g for consistency
            st.caption(f"≈ ₹{inr_prices[country] / 31.1035 * 10:,.2f} per 10g")

    # 📘 Explanation of differences
    st.subheader("📊 Why Do Gold Prices Differ Between Countries?")

    st.markdown("""
    Even though gold is a global commodity, its retail price varies by country because of these key factors:
    - **🔸 Import Duty:** India imposes ~15% import tax on gold imports, while Dubai and USA have very low or zero duties.
    - **🔸 GST / VAT:** India adds 3% GST on top of gold prices, while Dubai has 5% VAT but often waives it for tourists.
    - **🔸 Currency Exchange Rate:** Fluctuations in USD/INR and USD/AED affect local pricing.
    - **🔸 Making Charges:** In India, jewellers add making charges (up to 8–15%) for ornaments.
    - **🔸 Market Demand & Premiums:** Local festivals, investment demand, and central bank policies can raise or lower local prices.
    """)

    # Comparison summary
    st.info(f"""
    💡 **Summary Insight:**
    - Cheapest Market: **{cheapest}**
    - Most Expensive Market: **{most_expensive}**
    - Difference: ₹{(usd_prices[most_expensive] - usd_prices[cheapest]):,.2f} per oz
    """)

def main():
    st.set_page_config(
        page_title="Gold Price Predictor",
        page_icon="🥇",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🥇 Gold Price Prediction Dashboard")
    st.markdown("**Real-time gold prices and AI-powered predictions for India 🇮🇳, USA 🇺🇸, and Dubai 🇦🇪**")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Status Check
    st.sidebar.subheader("📡 API Status")
    api_status = {
       
        "News API": "✅" if NEWS_API_KEY else "❌", 
        "Exchange Rates": "✅" if EXCHANGE_API_KEY else "⚠️ (Limited)",
        "Yahoo Finance": "✅ (Free)"
    }
    
    for api, status in api_status.items():
        st.sidebar.text(f"{api}: {status}")
    
    # Time range selection
    days_data = st.sidebar.slider("📊 Historical Data (days)", 7, 90, 30)
    prediction_steps = st.sidebar.slider("🔮 Prediction Steps", 5, 60, 10)
    
    # Currency display option
    st.sidebar.subheader("💱 Currency Display")
    show_inr_conversion = st.sidebar.checkbox(
        "🇮🇳 Show prices in Indian Rupees", 
        value=True,
        help="Convert USD and AED prices to INR for easy comparison"
    )
    
    # Natural Language Query Interface
    
    
    # Main dashboard
    tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📈 Predictions", "📰 News & Analysis"])
    
    with tab1:
        if st.button("🔄 Get Live Gold Prices", type="primary"):
            with st.spinner("Fetching real-time gold prices from global markets..."):
                
                # Get real-time prices
                real_time_prices = get_real_time_multi_country_prices()
                if real_time_prices:
                    display_price_comparison(real_time_prices, show_inr_conversion)
                    
                    # Price analysis
                    st.subheader("💡 Price Analysis")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.info("**Why prices differ:**\n- Currency exchange rates\n- Local taxes and duties\n- Import/export costs\n- Market demand variations")
                    
                    with col2:
                        rates = get_exchange_rates()
                        st.info(f"**Current Exchange Rates:**\n- 1 USD = ₹{rates['USD_INR']:.2f} INR\n- 1 AED = ₹{rates['USD_INR']/rates['USD_AED']:.2f} INR")
                
                # Get historical data
                st.subheader("📊 Historical Trends")
                country_data = get_multi_country_gold_data(days_data)
                
                if country_data:
                    # Create combined chart
                    for country, df in country_data.items():
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=df['date'],
                            y=df['close'],
                            mode='lines+markers',
                            name=f"{country} ({df['currency'].iloc[0]})",
                            line=dict(width=2)
                        ))
                        fig.update_layout(
                            title=f"{country} Gold Price History",
                            xaxis_title="Date",
                            yaxis_title=f"Price ({df['currency'].iloc[0]})",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # --- Combined Graph: % Change Comparison ---
                    st.subheader("📈 Daily % Change Comparison Across Countries")
                    
                    fig_change = go.Figure()
                    
                    for country, df in country_data.items():
                        df['pct_change'] = df['close'].pct_change() * 100  # daily percentage change
                        fig_change.add_trace(go.Scatter(
                            x=df['date'],
                            y=df['pct_change'],
                            mode='lines',
                            name=f"{country}",
                            line=dict(width=9)
                        ))
                    
                    fig_change.update_layout(
                        title="Relative Gold Price Movements (Daily % Change)",
                        xaxis_title="Date",
                        yaxis_title="Daily % Change",
                        height=500,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig_change, use_container_width=True)
    
    with tab2:
        st.subheader("🔮 AI-Powered Price Predictions")
        
        if st.button("🚀 Generate Predictions", type="primary"):
            with st.spinner("Training AI models and generating predictions..."):
                country_data = get_multi_country_gold_data(days_data)
                
                if country_data:
                    predictions_data = {}
                    
                    for country, df in country_data.items():
                        predictor = GoldPricePredictor()
                        
                        if predictor.train(df):
                            predictor.evaluate(df)
                            predictions = predictor.predict_next_prices(df, prediction_steps)
                            if predictions:
                                predictions_data[country] = predictions
                    
                    if predictions_data:
                        # Display predictions
                        
                        st.subheader("📈 Short-term Predictions (Next 10 Steps)")
                        fig_pred = go.Figure()
                        for country, predictions in predictions_data.items():
                            current_price = country_data[country]['close'].iloc[-1]
                            
                            # Convert predictions into percentage change from current price
                            pct_change = [(p - current_price) / current_price * 100 for p in predictions]
                            time_steps = list(range(1, len(predictions) + 1))

                            fig_pred.add_trace(go.Scatter(
                                x=time_steps,
                                y=pct_change,  # 👈 percent change instead of raw price
                                mode='lines+markers',
                                line=dict(width=3),
                                marker=dict(size=7),
                                name=f"{COUNTRY_CONFIG[country]['flag']} {country}"
                            ))

                        fig_pred.update_layout(
                            title="Gold Price Predictions (% Change Comparison)",
                            xaxis_title="Prediction Steps (Next Day/Intervals)",
                            yaxis_title="Predicted % Change from Current Price",
                            height=450,
                            hovermode='x unified',
                            template='plotly_white'
                        )

                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        
                        # Prediction summary
                        st.subheader("📋 Prediction Summary")
                        
                        for country, predictions in predictions_data.items():
                            current_price = country_data[country]['close'].iloc[-1]
                            next_price = predictions[0]
                            change = ((next_price - current_price) / current_price) * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{COUNTRY_CONFIG[country]['flag']} {country}", 
                                        f"{current_price:.2f}", "Current")
                            with col2:
                                st.metric("Next Prediction", 
                                        f"{next_price:.2f}", f"{change:+.2f}%")
                            with col3:
                                trend = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
                                st.metric("Trend", trend)
    sentiment_analyzer = pipeline("sentiment-analysis")
    with tab3:
        st.subheader("📰 News Sentiment & Market Analysis")
        
        if st.button("📊 Analyze Market Sentiment", type="primary"):
            with st.spinner("Analyzing news and market sentiment..."):
                
                # 🗞️ Step 1: Fetch gold-related news for the last 7 days
                news_articles = get_news_data(21)
                
                if news_articles:
                    # 🧠 Step 2: Analyze sentiment using Hugging Face
                    news_with_sentiment = []
                    for article in news_articles:
                        text = f"{article['title']} {article['description']}"
                        
                        try:
                            # Hugging Face model returns: [{'label': 'POSITIVE', 'score': 0.98}]
                            result = sentiment_analyzer(text[:512])[0]  # truncate long texts
                            label = result['label'].capitalize()
                            score = result['score'] if label == 'Positive' else -result['score']
                        except Exception:
                            label, score = 'Neutral', 0.0
                        
                        news_with_sentiment.append({
                            **article,
                            'sentiment': label,
                            'sentiment_score': score
                        })
                    
                    # 📊 Step 3: Sentiment distribution pie chart
                    sentiment_counts = pd.DataFrame(news_with_sentiment)['sentiment'].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="Market Sentiment Distribution",
                            color_discrete_sequence=["#16a34a", "#dc2626", "#facc15"]  # Green, Red, Yellow
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    # 📈 Step 4: Average sentiment score & summary
                    with col2:
                        avg_sentiment = np.mean([a['sentiment_score'] for a in news_with_sentiment])
                        st.metric("Average Sentiment", f"{avg_sentiment:.3f}")
                        
                        if avg_sentiment > 0.3:
                            st.success("📈 Positive market sentiment")
                        elif avg_sentiment < -0.3:
                            st.error("📉 Negative market sentiment")
                        else:
                            st.info("➡️ Neutral market sentiment")
                
                    # 📰 Step 5: Display recent news articles with sentiment
                    st.subheader("📰 Recent Gold Market News")
                    sentiment_color = {
                        'Positive': '🟢',
                        'Negative': '🔴', 
                        'Neutral': '🟡'
                    }
                    
                    for i, article in enumerate(news_with_sentiment[:50]):
                        emoji = sentiment_color.get(article['sentiment'], '⚪')
                        with st.expander(f"{emoji} {article['title'][:80]}..."):
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Published:** {article['publishedAt']}")
                            st.write(f"**Sentiment:** {article['sentiment']} ({article['sentiment_score']:.3f})")
                            st.write(f"**Description:** {article['description']}")
                            st.write(f"**[Read More]({article['url']})**")
                    
                   
    
    # Footer with API setup instructions
    

if __name__ == "__main__":
    main()
