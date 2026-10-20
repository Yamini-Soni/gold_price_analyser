import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from transformers import pipeline

warnings.filterwarnings("ignore")
load_dotenv()


def get_secret(name, default=None):
    """Read a key from environment variables or Streamlit secrets."""
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


GOLDAPI_KEY = get_secret("GOLDAPI_KEY")
NEWS_API_KEY = get_secret("NEWS_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")
ALPHA_VANTAGE_KEY = get_secret("ALPHA_VANTAGE_KEY")
EXCHANGE_API_KEY = get_secret("EXCHANGE_API_KEY")

# Retained for compatibility with the original project. Groq is not currently
# used to generate any dashboard response.
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None

COUNTRY_CONFIG = {
    "USA": {"currency": "USD", "symbol": "GC=F", "multiplier": 1.0, "unit": "oz", "flag": "🇺🇸"},
    "India": {"currency": "INR", "symbol": "GC=F", "multiplier": 1.0, "unit": "10g", "flag": "🇮🇳"},
    "Dubai": {"currency": "AED", "symbol": "GC=F", "multiplier": 1.0, "unit": "oz", "flag": "🇦🇪"},
}

FEATURE_COLS = [
    "sma_5", "sma_20", "rsi", "volatility",
    "price_change", "volume_change", "hour", "day_of_week"
]


@st.cache_resource(show_spinner="Loading the news sentiment model...")
def get_sentiment_model():
    """Load the existing DistilBERT sentiment model once per app process."""
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )


class GoldPricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

    def prepare_features(self, df):
        """Prepare technical and time features used by the original model."""
        df = df.copy()

        # Normalize column names to the names expected by this project.
        df.columns = [str(c).lower() for c in df.columns]
        if "date" not in df.columns:
            df["date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq="D")
        if "close" not in df.columns:
            raise ValueError("Price data is missing the 'close' column.")
        if "volume" not in df.columns:
            df["volume"] = 0.0

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["date"] = df["date"].fillna(pd.Timestamp.now())

        df["sma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["rsi"] = self.calculate_rsi(df["close"])
        df["volatility"] = df["close"].rolling(window=10, min_periods=1).std()
        df["price_change"] = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["volume_change"] = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)
        df["hour"] = df["date"].dt.hour
        df["day_of_week"] = df["date"].dt.dayofweek

        # Keep the original app's simple missing-value handling, but fill each
        # feature using its own median instead of the volatility median.
        df = df.replace([np.inf, -np.inf], np.nan)
        for col in FEATURE_COLS:
            if df[col].isna().all():
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(df[col].median())
        return df

    def calculate_rsi(self, prices, window=14):
        """Calculate the project's rolling-average RSI-style indicator."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()

        # Avoid division-by-zero and return sensible boundary values.
        rs = gain.div(loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.where(loss.ne(0), 100)
        return rsi.fillna(50).clip(0, 100)

    def train(self, df):
        """Train the Random Forest on next-row closing prices."""
        try:
            df_features = self.prepare_features(df)
            X = df_features[FEATURE_COLS].iloc[:-1].copy()
            y = df_features["close"].iloc[1:].to_numpy()

            if len(X) <= 10:
                st.warning("Not enough historical rows to train. Please choose a longer period.")
                return False

            X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.is_trained = True
            return True
        except Exception as e:
            st.error(f"Training error: {e}")
            return False

    def evaluate(self, df):
        """Return training-period metrics, as in the original project."""
        if not self.is_trained:
            st.warning("Model not trained yet!")
            return None
        try:
            df_features = self.prepare_features(df)
            X = df_features[FEATURE_COLS].iloc[:-1].copy().fillna(0)
            y_true = df_features["close"].iloc[1:].to_numpy()
            y_pred = self.model.predict(self.scaler.transform(X))

            mae = mean_absolute_error(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")

            st.caption(
                f"Model metrics on the same historical period used for training — "
                f"MAE: {mae:.2f} · MSE: {mse:.2f} · RMSE: {rmse:.2f} · R²: {r2:.3f}"
            )
            return mae, mse, rmse, r2
        except Exception as e:
            st.warning(f"Could not calculate evaluation metrics: {e}")
            return None

    def predict_next_prices(self, df, steps=10):
        """Keep the original simplified recursive multi-step forecasting method."""
        if not self.is_trained:
            return None
        try:
            df_features = self.prepare_features(df)
            last_features = df_features[FEATURE_COLS].iloc[-1:].copy().fillna(0)
            current_price = float(df_features["close"].iloc[-1])
            predictions = []

            for _ in range(int(steps)):
                X_scaled = self.scaler.transform(last_features[FEATURE_COLS])
                pred_price = float(self.model.predict(X_scaled)[0])
                predictions.append(pred_price)

                new_row = last_features.copy()
                new_row["price_change"] = (
                    (pred_price - current_price) / current_price if current_price != 0 else 0
                )
                new_row["sma_5"] = (float(last_features["sma_5"].iloc[0]) * 4 + pred_price) / 5
                new_row["sma_20"] = (float(last_features["sma_20"].iloc[0]) * 19 + pred_price) / 20
                # Retain the original simplified behaviour: RSI, volatility,
                # volume_change and time features are not fully recalculated.
                last_features = new_row[FEATURE_COLS].copy()
                current_price = pred_price

            return predictions
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None


def get_exchange_rates():
    """Get USD exchange rates, with the original fallback behaviour."""
    try:
        if EXCHANGE_API_KEY:
            url = f"https://v6.exchangerate-api.com/v6/{EXCHANGE_API_KEY}/latest/USD"
        else:
            url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        rates = data.get("conversion_rates") or data.get("rates") or {}
        if "INR" not in rates or "AED" not in rates:
            raise ValueError("Exchange-rate response did not contain INR and AED rates.")
        return {"USD_INR": float(rates["INR"]), "USD_AED": float(rates["AED"])}
    except Exception:
        return {"USD_INR": 88.79, "USD_AED": 3.67}


def _normalize_yfinance_history(hist):
    """Normalize yfinance output, including possible MultiIndex columns."""
    if hist is None or hist.empty:
        return pd.DataFrame()

    df = hist.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # For a single ticker, keep the field name (Open, High, Close, etc.).
        df.columns = [
            next((str(part) for part in col if str(part).lower() in
                  {"open", "high", "low", "close", "volume"}), str(col[-1]))
            for col in df.columns
        ]
    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    # Yahoo Finance commonly calls the index column Date or Datetime.
    if "date" not in df.columns:
        if "datetime" in df.columns:
            df = df.rename(columns={"datetime": "date"})
        else:
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "date"})

    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            if col == "volume":
                df[col] = 0.0
            elif "close" in df.columns:
                df[col] = df["close"]
            else:
                return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume"]]


def get_multi_country_gold_data(days=30):
    """Fetch Yahoo Finance gold futures and convert for the selected displays."""
    try:
        end_date = datetime.now().date() + timedelta(days=1)
        start_date = end_date - timedelta(days=int(days))
        hist = yf.Ticker("GC=F").history(start=start_date.isoformat(), end=end_date.isoformat(), auto_adjust=False)
        base = _normalize_yfinance_history(hist)
        if base.empty:
            st.error("Yahoo Finance returned no usable gold data. Please try again later.")
            return None

        rates = get_exchange_rates()
        country_data = {}
        for country, config in COUNTRY_CONFIG.items():
            df = base.copy()
            if country == "India":
                # USD per troy ounce -> INR per 10 grams.
                multiplier = rates["USD_INR"] * 10 / 31.1034768
            elif country == "Dubai":
                # USD per troy ounce -> AED per troy ounce.
                multiplier = rates["USD_AED"]
            else:
                multiplier = 1.0

            for col in ["open", "high", "low", "close"]:
                df[col] = df[col] * multiplier
            df["country"] = country
            df["currency"] = config["currency"]
            df["unit"] = config["unit"]
            country_data[country] = df
        return country_data
    except Exception as e:
        st.error(f"Error fetching multi-country gold data: {e}")
        return None


def get_real_time_multi_country_prices():
    """Get latest available Yahoo Finance price and convert it for display."""
    try:
        hist = yf.Ticker("GC=F").history(period="5d", auto_adjust=False)
        base = _normalize_yfinance_history(hist)
        if base.empty:
            st.error("Yahoo Finance returned no recent gold prices.")
            return None

        current_price_usd = float(base["close"].iloc[-1])
        rates = get_exchange_rates()
        timestamp = pd.Timestamp(base["date"].iloc[-1]).to_pydatetime()
        prices = {}

        for country, config in COUNTRY_CONFIG.items():
            if country == "India":
                price = current_price_usd * rates["USD_INR"] * 10 / 31.1034768
            elif country == "Dubai":
                price = current_price_usd * rates["USD_AED"]
            else:
                price = current_price_usd
            prices[country] = {
                "price": round(price, 2),
                "currency": config["currency"],
                "unit": config["unit"],
                "flag": config["flag"],
                "timestamp": timestamp,
            }
        return prices
    except Exception as e:
        st.error(f"Error fetching latest prices: {e}")
        return None


def get_news_data(days=20):
    """Fetch gold-related articles from NewsAPI."""
    if not NEWS_API_KEY:
        st.warning("NewsAPI key is missing. Add NEWS_API_KEY to your .env file or Streamlit secrets.")
        return []
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(days))
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "gold price OR gold market OR XAU/USD OR gold investment",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "sortBy": "publishedAt",
            "language": "en",
            "apiKey": NEWS_API_KEY,
            "pageSize": 50,
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "error":
            st.warning(f"NewsAPI: {data.get('message', 'Request failed')}")
            return []

        articles = []
        for article in data.get("articles", []):
            title = article.get("title")
            description = article.get("description")
            if title and description:
                articles.append({
                    "title": title,
                    "description": description,
                    "url": article.get("url", ""),
                    "publishedAt": article.get("publishedAt", "Unknown"),
                    "source": (article.get("source") or {}).get("name", "Unknown source"),
                })
        return articles
    except Exception as e:
        st.warning(f"Error fetching news data: {e}")
        return []


def analyze_sentiment(text):
    """Analyze text using the same pretrained DistilBERT sentiment model."""
    try:
        model = get_sentiment_model()
        result = model(str(text)[:512])[0]
        return result["label"].capitalize(), round(float(result["score"]), 3)
    except Exception:
        return "Neutral", 0.0


def display_price_comparison(real_time_prices, show_inr_conversion=True):
    """Display prices after normalizing all markets to comparable units."""
    if not real_time_prices:
        st.warning("No latest gold price data available.")
        return

    st.subheader("🌍 Global Gold Price Comparison")
    rates = get_exchange_rates()
    usd_per_oz = {}
    inr_per_oz = {}

    for country, data in real_time_prices.items():
        price = float(data["price"])
        if country == "India":  # INR per 10g -> USD per troy ounce
            inr_per_oz[country] = price * 31.1034768 / 10
            usd_per_oz[country] = inr_per_oz[country] / rates["USD_INR"]
        elif country == "Dubai":  # AED per ounce -> USD per ounce
            usd_per_oz[country] = price / rates["USD_AED"]
            inr_per_oz[country] = usd_per_oz[country] * rates["USD_INR"]
        else:  # USD per ounce
            usd_per_oz[country] = price
            inr_per_oz[country] = price * rates["USD_INR"]

    cheapest = min(inr_per_oz, key=inr_per_oz.get)
    most_expensive = max(inr_per_oz, key=inr_per_oz.get)
    cols = st.columns(max(1, min(3, len(real_time_prices))))

    for i, (country, data) in enumerate(real_time_prices.items()):
        with cols[i % len(cols)]:
            note = "🏆 Lowest converted price" if country == cheapest else (
                "💎 Highest converted price" if country == most_expensive else ""
            )
            st.metric(
                f"{data['flag']} {country}",
                f"{data['price']:,.2f} {data['currency']}/{data['unit']}",
                note or None,
            )
            if show_inr_conversion:
                st.caption(f"≈ ₹{inr_per_oz[country]:,.2f} per troy ounce")
                st.caption(f"≈ ₹{inr_per_oz[country] / 31.1034768 * 10:,.2f} per 10g")

    st.subheader("📊 Why displayed gold prices can differ")
    st.markdown(
        "- Currency exchange rates affect converted prices.\n"
        "- Local taxes, import duties, market premiums and retail costs can differ.\n"
        "- These values are converted gold-futures prices, **not official local retail gold rates**."
    )
    difference = inr_per_oz[most_expensive] - inr_per_oz[cheapest]
    st.info(
        f"Lowest converted market: **{cheapest}** · Highest converted market: **{most_expensive}** · "
        f"Difference: **₹{difference:,.2f} per troy ounce**"
    )


def main():
    st.set_page_config(
        page_title="Gold Price Predictor",
        page_icon="🥇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("🥇 Gold Price Prediction Dashboard")
    st.markdown("**Gold prices and model predictions for India 🇮🇳, USA 🇺🇸, and Dubai 🇦🇪**")

    st.sidebar.header("⚙️ Configuration")
    st.sidebar.subheader("📡 API Status")
    st.sidebar.text(f"News API: {'✅ configured' if NEWS_API_KEY else '⚠️ key missing'}")
    st.sidebar.text(f"Exchange rates: {'✅ API key configured' if EXCHANGE_API_KEY else '⚠️ using public/fallback rates'}")
    st.sidebar.text("Yahoo Finance: uses yfinance")

    days_data = st.sidebar.slider("📊 Historical Data (days)", 7, 90, 30)
    prediction_steps = st.sidebar.slider("🔮 Prediction Steps", 5, 60, 10)
    show_inr_conversion = st.sidebar.checkbox(
        "🇮🇳 Show prices in Indian Rupees", value=True,
        help="Show equivalent INR prices for comparison."
    )

    tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📈 Predictions", "📰 News & Analysis"])

    with tab1:
        if st.button("🔄 Get Live Gold Prices", type="primary", key="live_prices"):
            with st.spinner("Fetching latest gold prices..."):
                real_time_prices = get_real_time_multi_country_prices()
                if real_time_prices:
                    display_price_comparison(real_time_prices, show_inr_conversion)
                    rates = get_exchange_rates()
                    col1, col2 = st.columns(2)
                    col1.info("**Price notes:** values are converted from gold futures and do not include all local retail costs.")
                    col2.info(
                        f"**Exchange rates:** 1 USD = ₹{rates['USD_INR']:.2f} · "
                        f"1 AED ≈ ₹{rates['USD_INR'] / rates['USD_AED']:.2f}"
                    )

            st.subheader("📊 Historical Trends")
            country_data = get_multi_country_gold_data(days_data)
            if country_data:
                for country, df in country_data.items():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=df["date"], y=df["close"], mode="lines+markers",
                        name=f"{country} ({df['currency'].iloc[0]})",
                    ))
                    fig.update_layout(
                        title=f"{country} Gold Price History",
                        xaxis_title="Date",
                        yaxis_title=f"Price ({df['currency'].iloc[0]}/{df['unit'].iloc[0]})",
                        height=400,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.subheader("📈 Daily % Change Comparison Across Countries")
                fig_change = go.Figure()
                for country, df in country_data.items():
                    daily_change = df["close"].pct_change() * 100
                    fig_change.add_trace(go.Scatter(
                        x=df["date"], y=daily_change, mode="lines", name=country,
                    ))
                fig_change.update_layout(
                    title="Relative Gold Price Movements (Daily % Change)",
                    xaxis_title="Date", yaxis_title="Daily % Change",
                    height=500, hovermode="x unified",
                )
                st.plotly_chart(fig_change, use_container_width=True)

    with tab2:
        st.subheader("🔮 AI-Powered Price Predictions")
        if st.button("🚀 Generate Predictions", type="primary", key="generate_predictions"):
            with st.spinner("Training the model and generating predictions..."):
                country_data = get_multi_country_gold_data(days_data)
                predictions_data = {}
                if country_data:
                    for country, df in country_data.items():
                        predictor = GoldPricePredictor()
                        if predictor.train(df):
                            predictor.evaluate(df)
                            predictions = predictor.predict_next_prices(df, prediction_steps)
                            if predictions:
                                predictions_data[country] = predictions

                if predictions_data:
                    st.subheader(f"📈 Short-term Predictions (Next {prediction_steps} Steps)")
                    fig_pred = go.Figure()
                    for country, predictions in predictions_data.items():
                        current_price = float(country_data[country]["close"].iloc[-1])
                        pct_change = [
                            (p - current_price) / current_price * 100 if current_price else 0
                            for p in predictions
                        ]
                        fig_pred.add_trace(go.Scatter(
                            x=list(range(1, len(predictions) + 1)),
                            y=pct_change, mode="lines+markers",
                            name=f"{COUNTRY_CONFIG[country]['flag']} {country}",
                        ))
                    fig_pred.update_layout(
                        title="Gold Price Predictions (% Change Comparison)",
                        xaxis_title="Prediction Step",
                        yaxis_title="Predicted % Change from Current Price",
                        height=450, hovermode="x unified", template="plotly_white",
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

                    st.subheader("📋 Prediction Summary")
                    for country, predictions in predictions_data.items():
                        current_price = float(country_data[country]["close"].iloc[-1])
                        next_price = float(predictions[0])
                        change = (next_price - current_price) / current_price * 100 if current_price else 0
                        col1, col2, col3 = st.columns(3)
                        col1.metric(f"{COUNTRY_CONFIG[country]['flag']} {country}", f"{current_price:,.2f}", "Current")
                        col2.metric("Next Prediction", f"{next_price:,.2f}", f"{change:+.2f}%")
                        trend = "📈 Bullish" if change > 0 else "📉 Bearish" if change < 0 else "➡️ Neutral"
                        col3.metric("Trend", trend)
                else:
                    st.warning("No predictions were generated. Check the available historical data.")

    with tab3:
        st.subheader("📰 News Sentiment & Market Analysis")
        if st.button("📊 Analyze Market Sentiment", type="primary", key="analyze_sentiment"):
            news_articles = get_news_data(21)
            if news_articles:
                try:
                    model = get_sentiment_model()
                    with st.spinner("Analyzing news sentiment..."):
                        news_with_sentiment = []
                        for article in news_articles:
                            text = f"{article['title']} {article['description']}"
                            try:
                                result = model(text[:512])[0]
                                label = result["label"].capitalize()
                                score = float(result["score"]) if label == "Positive" else -float(result["score"])
                            except Exception:
                                label, score = "Neutral", 0.0
                            news_with_sentiment.append({
                                **article, "sentiment": label, "sentiment_score": score
                            })

                    sentiment_counts = pd.DataFrame(news_with_sentiment)["sentiment"].value_counts()
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_pie = px.pie(
                            values=sentiment_counts.values,
                            names=sentiment_counts.index,
                            title="News Sentiment Distribution",
                            color_discrete_sequence=["#16a34a", "#dc2626", "#facc15"],
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with col2:
                        avg_sentiment = float(np.mean([a["sentiment_score"] for a in news_with_sentiment]))
                        st.metric("Average Sentiment Score", f"{avg_sentiment:.3f}")
                        st.caption("This is general text sentiment, not a direct gold-price forecast.")
                        if avg_sentiment > 0.3:
                            st.success("📈 Mostly positive article sentiment")
                        elif avg_sentiment < -0.3:
                            st.error("📉 Mostly negative article sentiment")
                        else:
                            st.info("➡️ Mixed or neutral article sentiment")

                    st.subheader("📰 Recent Gold Market News")
                    sentiment_color = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🟡"}
                    for article in news_with_sentiment[:50]:
                        emoji = sentiment_color.get(article["sentiment"], "⚪")
                        with st.expander(f"{emoji} {article['title'][:80]}"):
                            st.write(f"**Source:** {article['source']}")
                            st.write(f"**Published:** {article['publishedAt']}")
                            st.write(f"**Sentiment:** {article['sentiment']} ({article['sentiment_score']:.3f})")
                            st.write(f"**Description:** {article['description']}")
                            if article["url"]:
                                st.markdown(f"[Read more]({article['url']})")
                except Exception as e:
                    st.error(f"Could not load or run the sentiment model: {e}")
                    st.info("The price dashboard and prediction tab can still be used without running news sentiment.")
            else:
                st.info("No news articles were returned. Check your NEWS_API_KEY and NewsAPI access.")

    st.caption("Prototype for learning purposes. Price forecasts are estimates, not financial advice.")


if __name__ == "__main__":
    main()
