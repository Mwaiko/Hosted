from Models.Ensemble import TransformerRNNRegressionEnsemble, TransformerRegressor
from Models.Gru import GRURegressor
from Models.LSTM import LSTMRegressor
from Models.DecisionTree import FixedHybridRegimePipeline
from data.DataProcessor import DataProcess
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, Coin, Interval, Batches, Sequence):
        self.Coin = Coin
        self.Interval = Interval
        self.Batches = Batches
        self.Sequence = Sequence
        self.Dataprocessor = DataProcess(self.Sequence, self.Interval)
    
    def get_current_price(self) -> float:
        symbol = "BTCUSDT"
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        data = response.json()
        price = float(data['price'])
        print(f"Symbol: {data['symbol']}, Price: {price}")
        return price
    
    def EvaluateAllModels(self):
        print(" <<< Model Evaluation Beginning >>> ")
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test, Predictor = self.Dataprocessor.PreProcess(augment=False)
        X_train = np.array(X_train, dtype='float32')
        y_train = np.array(y_train, dtype='float32')
        X_test = np.array(X_test, dtype='float32')
        y_test = np.array(y_test, dtype='float32')
        
        print(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Data shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1,shuffle=False
        )
        
        print("<<< Model Preprocessing Complete >>>")
        print(f"Training data shape: {X_train.shape}, {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
        print(f"Test data shape: {X_test.shape}, {y_test.shape}")
        # Test enhanced pipeline  
        print("\n2. ENHANCED PIPELINE:")
        tree_method = "gradient_boosting"
        pipeline = FixedHybridRegimePipeline(
            n_regimes=3,
            tree_method=tree_method,
            use_global_fallback=True,
            regime_consistency_check=True,
            use_balanced_kmeans=True,
            verbose=True
        )
        
        # Train the pipeline
        pipeline.fit(X_train, y_train, X_val, y_val)
        # Evaluate with detailed regime analysis
        #evaluation_results = pipeline.evaluate_with_detailed_regime_analysis(X_test, y_test)
        pipeline.save_model(f"Infrastructure/{self.Coin}")

        
    def Train(self):
        print(" <<< Model Training Beginning >>> ")
        X_train, X_test, y_train, y_test, Predictor = self.Dataprocessor.PreProcess(augment=False)
        self.Predictor = Predictor
        
        X_train = np.array(X_train, dtype='float32')
        y_train = np.array(y_train, dtype='float32')
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, 
            test_size=0.12,        
            random_state=42,      
            shuffle=True          
        )
        
        # Determine appropriate sequence length
        if len(X_train.shape) == 3 and X_train.shape[1] > 1:
            # We have sequential data
            sequence_length = X_train.shape[1]
        elif len(X_train.shape) == 2:
            # We have feature vectors, use self.Sequence but ensure it's valid
            sequence_length = max(1, self.Sequence)
        else:
            sequence_length = max(1, self.Sequence)
        
        print("<<< Training Pipeline >>>")
        print(f"Using sequence_length: {sequence_length}")
        tree_method = "gradient_boosting"
        pipeline = FixedHybridRegimePipeline(
            n_regimes=3,
            tree_method=tree_method,
            use_global_fallback=False,
            regime_consistency_check=True,
            use_balanced_kmeans=True,
            verbose=True
        )
        
        loaded_model  = pipeline.load_model()
        if loaded_model == 0:
            self.enhanced_pipeline = loaded_model
        # Train with full monitorin
        else:
            self.enhanced_pipeline = pipeline.fit(X_train, y_train, X_val, y_val)
        
    def Predict(self, data):
        
        # Convert to numpy array
        if isinstance(data, list):
            data_array = np.array([data])
        elif isinstance(data, np.ndarray):
            data_array = data
        else:
            data_array = np.array([data])  # Handles scalars
        
        print(f" INPUT DEBUG: Initial array shape: {data_array.shape}")
        
    
        print(f" INPUT DEBUG: Data sample: {data_array}")
        
        try:
            # Check if pipeline has prediction_type attribute or assume binary
            prediction_type = getattr(self.enhanced_pipeline, 'prediction_type', 'binary')
            
            if prediction_type == 'binary':
                print("ðŸ” PIPELINE DEBUG: Making binary predictions...")
                results = self.enhanced_pipeline.predict(
                    data_array, 
                    return_probabilities=True, 
                )
                
                # Extract results from dictionary
                probabilities = results['probabilities']
                regimes = results['regimes']
                binary_predictions = results['predictions']
                confidence = results['confidence']
                models_used = results['models_used']
                
                print(f"OUTPUT DEBUG: probabilities shape: {probabilities.shape}")
                print(f"OUTPUT DEBUG: regimes shape: {regimes.shape}")
                print(f"OUTPUT DEBUG: binary_predictions shape: {binary_predictions.shape}")
                print(f"OUTPUT DEBUG: confidence shape: {confidence.shape}")
                print(f"OUTPUT DEBUG: models used: {set(models_used)}")
                
                # Single prediction case
                if len(probabilities) == 1:
                    latest_prob = probabilities[0]
                    latest_regime = regimes[0]
                    latest_signal = binary_predictions[0]
                    latest_confidence = confidence[0]
                    model_used = models_used[0]
                    
                    signal_text = "BUY" if latest_signal == 1 else "SELL"
                    
                    # Get regime threshold - handle both regime-specific and global models
                    if latest_regime in self.enhanced_pipeline.regime_thresholds:
                        regime_threshold = self.enhanced_pipeline.regime_thresholds[latest_regime]
                    elif -1 in self.enhanced_pipeline.regime_thresholds:  # Global model fallback
                        regime_threshold = self.enhanced_pipeline.regime_thresholds[-1]
                    else:
                        regime_threshold = 0.5  # Default fallback
                    
                    print(f" <<< Prediction Probability : {latest_prob:.4f} >>> ")
                    print(f" <<< Binary Signal : {latest_signal} ({signal_text}) >>> ")
                    print(f" <<< Confidence Score : {latest_confidence:.4f} >>> ")
                    print(f" <<< Regime : {latest_regime} >>> ")
                    print(f" <<< Model Used : {model_used} >>> ")
                    print(f" <<< Regime Threshold : {regime_threshold:.4f} >>> ")
                    
                    return latest_prob, latest_regime, latest_signal
                else:
                    print(f" <<< Generated {len(probabilities)} predictions >>> ")
                    print(f" <<< Average confidence: {np.mean(confidence):.4f} >>> ")
                    print(f" <<< Regime distribution: {np.bincount(regimes)} >>> ")
                    
                    return probabilities, regimes, binary_predictions
            
            else:
                print("PIPELINE DEBUG: Making regression predictions...")
                
                
                results = self.enhanced_pipeline.predict(
                    data_array, 
                    return_probabilities=True, 
                    return_regimes=True
                )
                
                # For regression-like output, use probabilities as continuous predictions
                predictions = results['probabilities']  # Use probabilities as continuous values
                regimes = results['regimes']
                
                print(f"OUTPUT DEBUG: predictions shape: {predictions.shape}")
                print(f"OUTPUT DEBUG: regimes shape: {regimes.shape}")
                
                if len(predictions) == 1:
                    latest_pred = predictions[0]
                    latest_regime = regimes[0]
                    
                    print(f" <<< Prediction : {latest_pred:.4f} >>> ")
                    print(f" <<< Regime : {latest_regime} >>> ")
                    
                    return latest_pred, latest_regime
                else:
                    print(f" <<< Generated {len(predictions)} predictions >>> ")
                    return predictions, regimes
        
        except Exception as e:
            print(f"PREDICTION ERROR: {str(e)}")
            
            # Enhanced error debugging
            try:
                print(f"ERROR DEBUG: Pipeline fitted: {self.enhanced_pipeline.is_fitted}")
                print(f"ERROR DEBUG: Pipeline trained regimes: {list(self.enhanced_pipeline.deep_models.keys())}")
                if hasattr(self.enhanced_pipeline, 'scaler') and hasattr(self.enhanced_pipeline.scaler, 'n_features_in_'):
                    print(f"ERROR DEBUG: Scaler feature count: {self.enhanced_pipeline.scaler.n_features_in_}")
                else:
                    print(f"ERROR DEBUG: Scaler not properly fitted")
                
                print(f"ERROR DEBUG: Input data shape: {data_array.shape}")
                
                # Check if the error is due to feature count mismatch
                if hasattr(self.enhanced_pipeline, 'scaler') and hasattr(self.enhanced_pipeline.scaler, 'n_features_in_'):
                    expected_features = self.enhanced_pipeline.scaler.n_features_in_
                    actual_features = data_array.shape[1]
                    if expected_features != actual_features:
                        print(f"ERROR DEBUG: Feature mismatch! Expected: {expected_features}, Got: {actual_features}")
                
                # Check available models
                if hasattr(self.enhanced_pipeline, 'deep_models'):
                    print(f"ERROR DEBUG: Available deep models: {list(self.enhanced_pipeline.deep_models.keys())}")
                if hasattr(self.enhanced_pipeline, 'global_model'):
                    print(f"ERROR DEBUG: Has global model: {self.enhanced_pipeline.global_model is not None}")
                
                # Try to get regime stats
                if hasattr(self.enhanced_pipeline, 'regime_stats'):
                    print(f"ERROR DEBUG: Regime stats: {self.enhanced_pipeline.regime_stats}")
                    
            except Exception as debug_error:
                print(f"ERROR DEBUG: Could not get debug info: {debug_error}")
            
            raise e
def run_test():
    model = Model("DOGE",5,16,1)
    curr_price = [0.19765, 0.19858, -4.139462078889267e-06, -0.00018140818327468358, 0.0001772687211957943, 64.10835214446948, 84.45378151260577, 71.28851540616283, 0.19827333333333333, 1.6831856934246114, 0.0, 1.0]
    model.Train()
    model.Predict(curr_price)
    