import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# Try to import TensorFlow/Keras, but make it optional
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
    
    # Set up dummy classes when TensorFlow is not available
    class Model:
        pass
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Deep learning features will be limited.")
    
    # Create dummy classes when TensorFlow is not available
    class tf:
        pass
    
    class keras:
        class Sequential:
            pass
        class Model:
            pass
        class callbacks:
            class EarlyStopping:
                pass
            class ReduceLROnPlateau:
                pass
        class optimizers:
            class Adam:
                pass
        class models:
            @staticmethod
            def load_model(path):
                raise ImportError("TensorFlow not available")
        
    class layers:
        class LSTM:
            pass
        class Dense:
            pass
        class Dropout:
            pass

from src.data.market import MarketDataFetcher
from src.ml.price_prediction import PredictionResult

logger = logging.getLogger(__name__)

@dataclass
class LSTMPrediction:
    symbol: str
    predictions: List[float]
    confidence_intervals: Dict[str, List[float]]  # 'lower' and 'upper' bounds
    model_accuracy: float
    sequence_length: int
    features_used: List[str]
    timestamp: datetime
    prediction_horizon: int

class LSTMPredictor:
    """LSTM-based deep learning price predictor"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 5):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM prediction. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = MinMaxScaler()
        self.feature_scaler = MinMaxScaler()
        self.market_fetcher = MarketDataFetcher()
        self.model_dir = "models/lstm"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
    
    def prepare_lstm_data(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare data for LSTM training"""
        try:
            # Create features
            df = data.copy()
            
            # Price-based features
            df['returns'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            df['volume_ma'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['bb_position'] = self._calculate_bollinger_position(df['Close'])
            
            # Moving averages ratios
            for window in [5, 10, 20]:
                ma = df['Close'].rolling(window=window).mean()
                df[f'ma_ratio_{window}'] = df['Close'] / ma
            
            # Select features for LSTM
            feature_cols = [
                'Close', 'High', 'Low', 'Open', 'Volume',
                'returns', 'high_low_ratio', 'close_open_ratio', 'volume_ratio',
                'rsi', 'bb_position', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20'
            ]
            
            # Remove NaN values
            df = df.dropna()
            
            if len(df) < self.sequence_length + self.prediction_horizon:
                raise ValueError(f"Insufficient data: need at least {self.sequence_length + self.prediction_horizon} samples")
            
            # Prepare feature matrix
            features = df[feature_cols].values
            target = df[target_col].values
            
            # Scale features and target separately
            features_scaled = self.feature_scaler.fit_transform(features)
            target_scaled = self.scaler.fit_transform(target.reshape(-1, 1)).flatten()
            
            # Create sequences
            X, y = [], []
            for i in range(self.sequence_length, len(features_scaled) - self.prediction_horizon + 1):
                X.append(features_scaled[i-self.sequence_length:i])
                y.append(target_scaled[i:i+self.prediction_horizon])
            
            X = np.array(X)
            y = np.array(y)
            
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {str(e)}")
            raise
    
    def build_lstm_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        model = keras.Sequential([
            # First LSTM layer with dropout
            layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(units=50, return_sequences=True),
            layers.Dropout(0.2),
            
            # Third LSTM layer
            layers.LSTM(units=50, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense layers
            layers.Dense(units=25, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(units=self.prediction_horizon, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_lstm_model(self, symbol: str, period: str = "2y", validation_split: float = 0.2) -> Dict[str, Any]:
        """Train LSTM model for price prediction"""
        try:
            # Get market data
            raw_data = self.market_fetcher.get_stock_data(symbol, period=period)
            if raw_data.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Prepare LSTM data
            X, y, feature_cols = self.prepare_lstm_data(raw_data)
            
            logger.info(f"Training LSTM for {symbol}: {X.shape[0]} samples, {X.shape[1]} timesteps, {X.shape[2]} features")
            
            # Split data (time series split)
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Build model
            input_shape = (X.shape[1], X.shape[2])
            model = self.build_lstm_model(input_shape)
            
            # Training callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss', 
                    patience=10, 
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.5, 
                    patience=5, 
                    min_lr=0.0001
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=callbacks,
                verbose=0
            )
            
            # Evaluate model
            train_predictions = model.predict(X_train)
            val_predictions = model.predict(X_val)
            
            # Calculate metrics (for first prediction day)
            train_mae = mean_absolute_error(y_train[:, 0], train_predictions[:, 0])
            val_mae = mean_absolute_error(y_val[:, 0], val_predictions[:, 0])
            
            # Save model and scalers
            model_path = os.path.join(self.model_dir, f"{symbol}_lstm_model")
            model.save(model_path)
            
            scalers_path = os.path.join(self.model_dir, f"{symbol}_scalers.pkl")
            joblib.dump({
                'scaler': self.scaler,
                'feature_scaler': self.feature_scaler,
                'feature_cols': feature_cols,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon
            }, scalers_path)
            
            self.model = model
            
            return {
                'symbol': symbol,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'features': len(feature_cols),
                'train_mae': train_mae,
                'val_mae': val_mae,
                'model_path': model_path,
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model for {symbol}: {str(e)}")
            raise
    
    def predict_lstm(self, symbol: str) -> LSTMPrediction:
        """Make LSTM price prediction"""
        try:
            # Load model and scalers
            model_path = os.path.join(self.model_dir, f"{symbol}_lstm_model")
            scalers_path = os.path.join(self.model_dir, f"{symbol}_scalers.pkl")
            
            if not os.path.exists(model_path):
                # Train model if it doesn't exist
                logger.info(f"Training new LSTM model for {symbol}")
                self.train_lstm_model(symbol)
            
            model = keras.models.load_model(model_path)
            scalers_data = joblib.load(scalers_path)
            
            self.scaler = scalers_data['scaler']
            self.feature_scaler = scalers_data['feature_scaler']
            feature_cols = scalers_data['feature_cols']
            
            # Get recent data
            recent_data = self.market_fetcher.get_stock_data(symbol, period="6mo")
            if len(recent_data) < self.sequence_length:
                raise ValueError(f"Insufficient recent data for {symbol}")
            
            # Prepare features for prediction
            df = recent_data.copy()
            
            # Add same features as training
            df['returns'] = df['Close'].pct_change()
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            df['volume_ma'] = df['Volume'].rolling(window=10).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_ma']
            df['rsi'] = self._calculate_rsi(df['Close'])
            df['bb_position'] = self._calculate_bollinger_position(df['Close'])
            
            for window in [5, 10, 20]:
                ma = df['Close'].rolling(window=window).mean()
                df[f'ma_ratio_{window}'] = df['Close'] / ma
            
            df = df.dropna()
            
            # Get last sequence for prediction
            features = df[feature_cols].values
            features_scaled = self.feature_scaler.transform(features)
            
            # Prepare input sequence
            X_pred = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, len(feature_cols))
            
            # Make prediction
            prediction_scaled = model.predict(X_pred, verbose=0)[0]
            
            # Inverse transform prediction
            predictions = self.scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()
            
            # Calculate confidence intervals (simplified using prediction variance)
            # In practice, you might use techniques like Monte Carlo Dropout
            prediction_std = np.std(prediction_scaled) * self.scaler.scale_[0]
            lower_bound = predictions - (1.96 * prediction_std)  # 95% CI
            upper_bound = predictions + (1.96 * prediction_std)
            
            confidence_intervals = {
                'lower': lower_bound.tolist(),
                'upper': upper_bound.tolist()
            }
            
            # Estimate model accuracy (simplified)
            recent_prices = df['Close'].values[-10:]  # Last 10 prices
            price_volatility = np.std(recent_prices) / np.mean(recent_prices)
            model_accuracy = max(0.5, 1 - price_volatility)  # Simple accuracy estimate
            
            return LSTMPrediction(
                symbol=symbol,
                predictions=predictions.tolist(),
                confidence_intervals=confidence_intervals,
                model_accuracy=model_accuracy,
                sequence_length=self.sequence_length,
                features_used=feature_cols,
                timestamp=datetime.now(),
                prediction_horizon=self.prediction_horizon
            )
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction for {symbol}: {str(e)}")
            raise
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_bollinger_position(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        bb_upper = sma + (std * 2)
        bb_lower = sma - (std * 2)
        bb_position = (prices - bb_lower) / (bb_upper - bb_lower)
        return bb_position.fillna(0.5)

class SimpleNeuralNetwork:
    """Simple neural network for price prediction (when TensorFlow is not available)"""
    
    def __init__(self):
        self.weights = None
        self.model_trained = False
    
    def train_simple_nn(self, symbol: str) -> Dict[str, Any]:
        """Train a simple neural network using sklearn MLPRegressor"""
        try:
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from src.ml.price_prediction import MLPricePredictor
            
            # Use the existing ML predictor but with neural network
            ml_predictor = MLPricePredictor()
            data, feature_columns = ml_predictor.prepare_data(symbol)
            
            target_col = 'target_5d'
            X = data[feature_columns].fillna(0)
            y = data[target_col].fillna(method='ffill')
            
            # Remove NaN targets
            valid_idx = ~y.isna()
            X = X[valid_idx]
            y = y[valid_idx]
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train neural network
            nn_model = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            nn_model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = nn_model.score(X_train_scaled, y_train)
            test_score = nn_model.score(X_test_scaled, y_test)
            
            # Save model
            model_data = {
                'model': nn_model,
                'scaler': scaler,
                'feature_columns': feature_columns,
                'train_score': train_score,
                'test_score': test_score
            }
            
            model_path = os.path.join("models", f"{symbol}_simple_nn.pkl")
            joblib.dump(model_data, model_path)
            
            return {
                'symbol': symbol,
                'model_type': 'Simple Neural Network',
                'train_score': train_score,
                'test_score': test_score,
                'features': len(feature_columns)
            }
            
        except Exception as e:
            logger.error(f"Error training simple NN for {symbol}: {str(e)}")
            raise
