import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import requests
import os

from dotenv import load_dotenv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from typing import Tuple, Optional, Dict, Any
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class StockAnalysisApp:
    def __init__(self):
        self.setup_streamlit()
        self.setup_api_key()
        
    def setup_streamlit(self):
        """Configure Streamlit page settings and layout"""
        st.set_page_config(
            page_title="Advanced Indian Stock Analysis",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better UI
        st.markdown("""
            <style>
            .main {
                padding: 0rem 1rem;
            }
            .stAlert {
                padding: 1rem;
                margin: 1rem 0;
            }
            .st-emotion-cache-1y4p8pa {
                max-width: 100%;
            }
            </style>
            """, unsafe_allow_html=True)
        
    def setup_api_key(self):
        """Validate Groq API key configuration"""
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        if not self.groq_api_key:
            st.sidebar.error("⚠️ Groq API key not found! Please set GROQ_API_KEY in your environment variables.")
            
    @staticmethod
    def download_stock_data(symbol: str, start_date: datetime.date, end_date: datetime.date) -> Tuple[Optional[pd.DataFrame], Optional[yf.Ticker]]:
        """Download stock data with error handling"""
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            if df.empty:
                st.error(f"No data found for {symbol}. Please verify the stock symbol.")
                return None, None
            return df, stock
        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            st.error(f"Failed to fetch data: {str(e)}")
            return None, None

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the stock data"""
        try:
            data['SMA20'] = SMAIndicator(data['Close'], window=20).sma_indicator()
            data['SMA50'] = SMAIndicator(data['Close'], window=50).sma_indicator()
            data['RSI'] = RSIIndicator(data['Close']).rsi()
            return data
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            st.error("Failed to calculate technical indicators")
            return data

    def create_stock_chart(self, data: pd.DataFrame) -> None:
        """Create an interactive stock chart with technical indicators"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Price Action & Moving Averages', 'RSI'),
                row_heights=[0.7, 0.3]
            )

            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price'
                ),
                row=1, col=1
            )

            # Moving averages
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA20'], name='SMA20', line=dict(color='blue', width=1)),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=data.index, y=data['SMA50'], name='SMA50', line=dict(color='red', width=1)),
                row=1, col=1
            )

            # RSI
            fig.add_trace(
                go.Scatter(x=data.index, y=data['RSI'], name='RSI', line=dict(color='purple')),
                row=2, col=1
            )

            # Add RSI levels
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)

            # Update layout
            fig.update_layout(
                height=800,
                template='plotly_dark',
                showlegend=True,
                xaxis_rangeslider_visible=False
            )

            # Add range selector
            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                )
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            logger.error(f"Error creating chart: {str(e)}")
            st.error("Failed to create stock chart")

    def get_groq_analysis(self, symbol: str, stock_data: pd.DataFrame, fundamental_data: Dict[str, Any]) -> Optional[str]:
        """Get stock analysis from Groq LLM"""
        if not self.groq_api_key:
            return None

        prompt = f"""
        Analyze the following Indian stock data as a financial expert:

        Stock: {symbol}
        Current Price: ₹{fundamental_data.get('currentPrice', 'N/A')}
        Market Stats:
        - 52W High: ₹{fundamental_data.get('fiftyTwoWeekHigh', 'N/A')}
        - 52W Low: ₹{fundamental_data.get('fiftyTwoWeekLow', 'N/A')}
        - P/E Ratio: {fundamental_data.get('trailingPE', 'N/A')}
        - Market Cap: ₹{fundamental_data.get('marketCap', 'N/A'):,}
        - Beta: {fundamental_data.get('beta', 'N/A')}

        Recent Closing Prices:
        {stock_data['Close'].tail().to_string()}

        Provide a detailed analysis including:
        1. Technical Position
        2. Valuation Assessment
        3. Risk Analysis
        4. Investment Recommendation
        5. Key Price Levels to Watch
        """

        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "mixtral-8x7b-32768",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.5,
                    "max_tokens": 1024
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {str(e)}")
            st.error("Failed to get AI analysis. Please try again later.")
            return None

    def run_price_prediction(self, data: pd.DataFrame, forecast_days: int, model_type: str) -> None:
        """Run price prediction analysis"""
        try:
            # Prepare data
            df = data[['Close']].copy()
            df['Target'] = df.Close.shift(-forecast_days)
            
            scaler = StandardScaler()
            x = scaler.fit_transform(df.drop(['Target'], axis=1))
            
            # Split data
            x_forecast = x[-forecast_days:]
            x = x[:-forecast_days]
            y = df.Target.values[:-forecast_days]
            
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
            
            # Select and train model
            model = LinearRegression() if model_type == 'Linear Regression' else RandomForestRegressor(n_estimators=100)
            model.fit(x_train, y_train)
            
            # Make predictions
            test_preds = model.predict(x_test)
            forecast_preds = model.predict(x_forecast)
            
            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("R² Score", f"{r2_score(y_test, test_preds):.4f}")
            with col2:
                st.metric("Mean Absolute Error", f"₹{mean_absolute_error(y_test, test_preds):.2f}")
            
            # Display predictions
            st.subheader("Price Forecasts")
            for i, pred in enumerate(forecast_preds, 1):
                st.metric(f"Day {i}", f"₹{pred:.2f}")
            
            # Plot predictions
            self.plot_predictions(data, forecast_preds, forecast_days)
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            st.error("Failed to generate price predictions")

    def plot_predictions(self, data: pd.DataFrame, predictions: np.ndarray, forecast_days: int) -> None:
        """Plot the predictions against historical data"""
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=data.index[-30:],
            y=data['Close'].tail(30),
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Predictions
        future_dates = pd.date_range(start=data.index[-1], periods=forecast_days+1, freq='D')[1:]
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price (₹)',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_sidebar(self) -> Tuple[str, datetime.date, datetime.date, str]:
        """Render sidebar controls"""
        st.sidebar.title("📊 Analysis Controls")
        
        symbol = st.sidebar.text_input(
            'Stock Symbol (NSE)',
            value='RELIANCE.NS',
            help="Enter NSE stock symbol (e.g., RELIANCE.NS, TCS.NS)"
        ).upper()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                'Start Date',
                value=datetime.date.today() - datetime.timedelta(days=365)
            )
        with col2:
            end_date = st.date_input('End Date', value=datetime.date.today())
            
        analysis_type = st.sidebar.radio(
            "Analysis Type",
            ["Technical Analysis", "Price Prediction"],
            format_func=lambda x: x.replace("_", " ")
        )
        
        return symbol, start_date, end_date, analysis_type

    def main(self):
        """Main application logic"""
        st.title("🚀 Advanced Indian Stock Analysis")
        
        symbol, start_date, end_date, analysis_type = self.render_sidebar()
        
        if st.sidebar.button('Analyze', type='primary'):
            if start_date >= end_date:
                st.sidebar.error("Start date must be before end date")
                return
                
            with st.spinner('Fetching data...'):
                data, stock_info = self.download_stock_data(symbol, start_date, end_date)
                
            if data is not None and stock_info is not None:
                data = self.calculate_technical_indicators(data)
                
                if analysis_type == "Technical Analysis":
                    # Display stock info
                    info = stock_info.info
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current Price", f"₹{info.get('currentPrice', 'N/A'):,.2f}")
                    with col2:
                        st.metric("Market Cap", f"₹{info.get('marketCap', 'N/A'):,.0f}")
                    with col3:
                        st.metric("P/E Ratio", f"{info.get('trailingPE', 'N/A'):.2f}")
                    with col4:
                        st.metric("Beta", f"{info.get('beta', 'N/A'):.2f}")
                    
                    # Technical Analysis
                    st.subheader("Technical Analysis")
                    self.create_stock_chart(data)
                    
                    # AI Analysis
                    if self.groq_api_key:
                        with st.expander("📈 AI Analysis", expanded=True):
                            with st.spinner("Generating AI analysis..."):
                                analysis = self.get_groq_analysis(symbol, data, info)
                                if analysis:
                                    st.markdown(analysis)
                
                else:  # Price Prediction
                    st.subheader("Price Prediction")
                    col1, col2 = st.columns(2)
                    with col1:
                        model_type = st.selectbox(
                            'Select Model',
                            ['Linear Regression', 'Random Forest Regressor']
                        )
                    with col2:
                        forecast_days = st.number_input(
                            'Forecast Days',
                            min_value=1,
                            max_value=30,
                            value=5
                        )
                    
                    self.run_price_prediction(data, forecast_days, model_type)

if __name__ == '__main__':
    app = StockAnalysisApp()
    app.main()