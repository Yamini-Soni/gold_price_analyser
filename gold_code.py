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
