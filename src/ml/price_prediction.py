import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

from src.data.market import MarketDataFetcher

logger = logging.getLogger(__name__)

@dataclass
class PredictionResult:
    symbol: str
    predictions: List[float]  # Future price predictions
    confidence_score: float  # Model confidence (0-1)
    model_type: str
    features_used: List[str]
    prediction_horizon: int  # Days ahead
    timestamp: datetime
    current_price: float
    predicted_direction: str  # 'up', 'down', 'sideways'
    probability_up: float
    probability_down: float
    target_prices: Dict[str, float]  # 1d, 5d, 30d predictions

class FeatureEngineer:
    """Create features for machine learning models"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features"""
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f'ma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ma_ratio_{window}'] = df['Close'] / df[f'ma_{window}']
        
        # Bollinger Bands
        df['bb_upper'] = df['ma_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['bb_lower'] = df['ma_20'] - (df['Close'].rolling(window=20).std() * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['Close'])
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=60).mean()
        
        # Momentum features
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
        
        # Pattern features
        df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
        df['higher_low'] = (df['Low'] > df['Low'].shift(1)).astype(int)
        df['doji'] = (abs(df['Close'] - df['Open']) < (df['High'] - df['Low']) * 0.1).astype(int)
        
        return df
    
    def create_lag_features(self, data: pd.DataFrame, target_col: str = 'Close', lags: int = 5) -> pd.DataFrame:
        """Create lagged features for time series prediction"""
        df = data.copy()
        
        for i in range(1, lags + 1):
            df[f'{target_col}_lag_{i}'] = df[target_col].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        return df
    
    def create_future_targets(self, data: pd.DataFrame, horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Create future price targets"""
        df = data.copy()
        
        for horizon in horizons:
            df[f'target_{horizon}d'] = df['Close'].shift(-horizon)
            df[f'target_return_{horizon}d'] = (df['Close'].shift(-horizon) / df['Close']) - 1
            df[f'target_direction_{horizon}d'] = (df[f'target_return_{horizon}d'] > 0).astype(int)
        
        return df
    
    def prepare_features(self, data: pd.DataFrame, target_horizons: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        df = self.create_technical_features(data)
        df = self.create_lag_features(df)
        df = self.create_future_targets(df, target_horizons)
        
        # Remove infinite and NaN values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram

class MLPricePredictor:
    """Machine Learning Price Prediction System"""
    
    def __init__(self, model_type: str = "ensemble"):
        self.model_type = model_type
        self.models = {}
        self.scalers = {}
        self.feature_engineer = FeatureEngineer()
        self.market_fetcher = MarketDataFetcher()
        self.model_dir = "models"
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Initialize models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear': Ridge(alpha=1.0),
            'ensemble': None  # Will be created later
        }
    
    def prepare_data(self, symbol: str, period: str = "2y") -> Tuple[pd.DataFrame, List[str]]:
        """Prepare data for training"""
        try:
            # Get market data
            raw_data = self.market_fetcher.get_stock_data(symbol, period=period)
            if raw_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Engineer features
            data = self.feature_engineer.prepare_features(raw_data)
            
            # Select feature columns (exclude targets and non-numeric)
            feature_columns = []
            exclude_cols = ['target_1d', 'target_5d', 'target_10d', 'target_return_1d', 
                          'target_return_5d', 'target_return_10d', 'target_direction_1d', 
                          'target_direction_5d', 'target_direction_10d', 'Date']
            
            for col in data.columns:
                if col not in exclude_cols and data[col].dtype in ['float64', 'int64']:
                    if not data[col].isna().all():
                        feature_columns.append(col)
            
            return data, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing data for {symbol}: {str(e)}")
            raise
    
    def train_model(self, symbol: str, horizon: int = 5, test_size: float = 0.2) -> Dict[str, Any]:
        """Train prediction model for a symbol"""
        try:
            data, feature_columns = self.prepare_data(symbol)
            
            target_col = f'target_{horizon}d'
            if target_col not in data.columns:
                raise ValueError(f"Target column {target_col} not found")
            
            # Prepare features and target
            X = data[feature_columns].fillna(method='ffill').fillna(0)
            y = data[target_col].fillna(method='ffill')
            
            # Remove rows with NaN targets
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < 50:
                raise ValueError(f"Insufficient data for training: {len(X)} samples")
            
            # Time series split (more appropriate for financial data)
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, test_idx = list(tscv.split(X))[-1]  # Use last split
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train models
            model_scores = {}
            trained_models = {}
            
            for model_name, model in self.models.items():
                if model_name == 'ensemble':
                    continue
                    
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    score = {
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred)
                    }
                    
                    model_scores[model_name] = score
                    trained_models[model_name] = model
                    
                except Exception as e:
                    logger.warning(f"Error training {model_name}: {str(e)}")
                    continue
            
            if not trained_models:
                raise ValueError("No models successfully trained")
            
            # Create ensemble model (simple average)
            ensemble_predictions = []
            for model in trained_models.values():
                ensemble_predictions.append(model.predict(X_test_scaled))
            
            if ensemble_predictions:
                ensemble_pred = np.mean(ensemble_predictions, axis=0)
                ensemble_score = {
                    'mse': mean_squared_error(y_test, ensemble_pred),
                    'mae': mean_absolute_error(y_test, ensemble_pred),
                    'r2': r2_score(y_test, ensemble_pred)
                }
                model_scores['ensemble'] = ensemble_score
            
            # Save best model
            best_model_name = min(model_scores.items(), key=lambda x: x[1]['mse'])[0]
            
            model_data = {
                'models': trained_models,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'symbol': symbol,
                'horizon': horizon,
                'scores': model_scores,
                'best_model': best_model_name
            }
            
            # Save to disk
            model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}d_model.pkl")
            joblib.dump(model_data, model_path)
            
            return {
                'symbol': symbol,
                'horizon': horizon,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_count': len(feature_columns),
                'model_scores': model_scores,
                'best_model': best_model_name,
                'model_path': model_path
            }
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            raise
    
    def predict_price(self, symbol: str, horizon: int = 5) -> PredictionResult:
        """Make price prediction for a symbol"""
        try:
            # Try to load existing model
            model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}d_model.pkl")
            
            if os.path.exists(model_path):
                try:
                    model_data = joblib.load(model_path)
                except:
                    # Retrain if model loading fails
                    logger.info(f"Retraining model for {symbol}")
                    self.train_model(symbol, horizon)
                    model_data = joblib.load(model_path)
            else:
                # Train new model
                logger.info(f"Training new model for {symbol}")
                self.train_model(symbol, horizon)
                model_data = joblib.load(model_path)
            
            # Get recent data for prediction
            recent_data = self.market_fetcher.get_stock_data(symbol, period="6mo")
            if recent_data.empty:
                raise ValueError(f"No recent data available for {symbol}")
            
            # Prepare features
            data_with_features = self.feature_engineer.prepare_features(recent_data)
            latest_features = data_with_features[model_data['feature_columns']].iloc[-1:].fillna(0)
            
            # Scale features
            features_scaled = model_data['scaler'].transform(latest_features)
            
            # Make predictions with all models
            predictions = {}
            for model_name, model in model_data['models'].items():
                try:
                    pred = model.predict(features_scaled)[0]
                    predictions[model_name] = pred
                except:
                    continue
            
            if not predictions:
                raise ValueError("No predictions generated")
            
            # Ensemble prediction
            ensemble_pred = np.mean(list(predictions.values()))
            predictions['ensemble'] = ensemble_pred
            
            # Use best model prediction
            best_model = model_data['best_model']
            final_prediction = predictions.get(best_model, ensemble_pred)
            
            current_price = recent_data['Close'].iloc[-1]
            
            # Calculate prediction metrics
            predicted_return = (final_prediction - current_price) / current_price
            predicted_direction = 'up' if predicted_return > 0.01 else 'down' if predicted_return < -0.01 else 'sideways'
            
            # Confidence based on model performance
            best_score = model_data['scores'][best_model]
            confidence = max(0, min(1, best_score['r2'])) if best_score['r2'] > 0 else 0.1
            
            # Probability estimates (simplified)
            if predicted_return > 0:
                prob_up = 0.6 + (confidence * 0.3)
                prob_down = 1 - prob_up
            elif predicted_return < 0:
                prob_down = 0.6 + (confidence * 0.3)
                prob_up = 1 - prob_down
            else:
                prob_up = prob_down = 0.5
            
            # Create target prices for different horizons
            target_prices = {}
            for h in [1, 5, 10, 30]:
                if h <= horizon:
                    target_prices[f"{h}d"] = final_prediction
                else:
                    # Extrapolate for longer horizons
                    daily_return = (predicted_return + 1) ** (1/horizon) - 1
                    target_prices[f"{h}d"] = current_price * ((1 + daily_return) ** h)
            
            return PredictionResult(
                symbol=symbol,
                predictions=[final_prediction],
                confidence_score=confidence,
                model_type=best_model,
                features_used=model_data['feature_columns'],
                prediction_horizon=horizon,
                timestamp=datetime.now(),
                current_price=current_price,
                predicted_direction=predicted_direction,
                probability_up=prob_up,
                probability_down=prob_down,
                target_prices=target_prices
            )
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {str(e)}")
            raise
    
    def batch_predict(self, symbols: List[str], horizon: int = 5) -> Dict[str, PredictionResult]:
        """Make predictions for multiple symbols"""
        results = {}
        
        for symbol in symbols:
            try:
                result = self.predict_price(symbol, horizon)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {str(e)}")
                continue
        
        return results
    
    def get_model_performance(self, symbol: str, horizon: int = 5) -> Dict[str, Any]:
        """Get model performance metrics"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_{horizon}d_model.pkl")
            
            if not os.path.exists(model_path):
                return {"error": "Model not found. Train the model first."}
            
            model_data = joblib.load(model_path)
            
            return {
                'symbol': symbol,
                'horizon': horizon,
                'model_scores': model_data['scores'],
                'best_model': model_data['best_model'],
                'feature_count': len(model_data['feature_columns']),
                'features_used': model_data['feature_columns'][:10]  # Top 10 features
            }
            
        except Exception as e:
            logger.error(f"Error getting model performance for {symbol}: {str(e)}")
            return {"error": str(e)}
