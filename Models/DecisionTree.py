import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import warnings
import  pandas as pd
warnings.filterwarnings('ignore')
from keras.models import load_model,save_model
# Force TensorFlow to use CPU only (fixes CUDA issues on AMD)
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)


class FixedHybridRegimePipeline:
    """
    FIXED VERSION addressing regime mapping issues:
    1. Proper regime label remapping after clustering
    2. Consistent regime assignment between train/test
    3. Enhanced regime tracking during evaluation
    4. Fixed leaf feature extraction compatibility
    5. Improved regime validation and debugging
    """
    
    def __init__(self, n_regimes=3, tree_method='random_forest', 
                 use_balanced_kmeans=True, min_regime_size=50,
                 ensemble_prediction=False, confidence_threshold=0.1,
                 use_global_fallback=True, regime_consistency_check=True,
                 random_state=42, verbose=True):
        
        self.n_regimes = n_regimes
        self.tree_method = tree_method
        self.use_balanced_kmeans = use_balanced_kmeans
        self.min_regime_size = min_regime_size
        self.ensemble_prediction = ensemble_prediction
        self.confidence_threshold = confidence_threshold
        self.use_global_fallback = use_global_fallback
        self.regime_consistency_check = regime_consistency_check
        self.random_state = random_state
        self.verbose = verbose
        
        # Core components
        self.tree_model = None
        self.regime_clusterer = None
        self.regime_models = {}
        self.global_model = None
        self.scaler = StandardScaler()
        
        # CRITICAL FIX: Regime mapping tracking
        self.regime_label_mapping = {}  # Maps cluster labels to sequential regime IDs
        self.active_regimes = []  # List of actually used regime IDs
        
        # Training state
        self.is_fitted = False
        self.regime_thresholds = {}
        self.global_threshold = 0.5
        self.regime_performance = {}
        self.regime_stats = {}  # Enhanced regime statistics
    
    def _build_tree_model(self):
        """Build tree model with enhanced stability"""
        if self.verbose:
            print(f"Building {self.tree_method} model...")
            
        if self.tree_method == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=1500,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.tree_method == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=2000,
                max_depth=6,
                learning_rate=0.1,
                min_samples_split=30,
                min_samples_leaf=15,
                subsample=0.8,
                random_state=self.random_state
            )
        
        return model
    
    def _get_tree_leaf_features(self, X):
        """
        FIXED: Handle different apply() outputs for different tree methods
        """
        if self.verbose:
            print(f"Extracting leaf features from {len(X)} samples...")
        
        leaf_indices = self.tree_model.apply(X)
        
        if self.verbose:
            print(f"Raw leaf indices shape: {leaf_indices.shape}")
        
        # Handle different dimensionalities
        if len(leaf_indices.shape) == 2:
            # RandomForest case: (n_samples, n_estimators)
            processed_indices = leaf_indices
            
        elif len(leaf_indices.shape) == 3:
            # GradientBoosting case: (n_samples, n_estimators, n_classes)
            # Take the mean across classes to create 2D representation
            processed_indices = np.mean(leaf_indices, axis=2)
            if self.verbose:
                print(f"Averaged across classes, new shape: {processed_indices.shape}")
                    
        else:
            raise ValueError(f"Unexpected leaf indices shape: {leaf_indices.shape}")
        
        if self.verbose:
            print(f"Final leaf features shape: {processed_indices.shape}")
            print(f"Feature value range: [{processed_indices.min():.1f}, {processed_indices.max():.1f}]")
        
        return processed_indices
    
    def _discover_regimes_with_mapping(self, X, y):
        """
        CRITICAL FIX: Discover regimes and create proper label mapping
        """
        if self.verbose:
            print("=" * 60)
            print("STEP 1: TRAINING TREE MODEL")
            print("=" * 60)
        
        # Train tree model
        self.tree_model = self._build_tree_model()
        self.tree_model.fit(X, y)
        
        if self.verbose:
            print("Tree model trained successfully!")
        
        # Extract leaf features
        leaf_indices = self._get_tree_leaf_features(X)
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 2: REGIME DISCOVERY WITH PROPER MAPPING")
            print("=" * 60)
        
        # Perform clustering
        if self.use_balanced_kmeans:
            clusterer = BalancedKMeans(
                n_clusters=self.n_regimes,
                min_cluster_size=self.min_regime_size,
                random_state=self.random_state,
                verbose=self.verbose
            )
        else:
            clusterer = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=20
            )
        
        raw_regimes = clusterer.fit_predict(leaf_indices)
        self.regime_clusterer = clusterer
        
        if self.verbose:
            unique_raw, counts_raw = np.unique(raw_regimes, return_counts=True)
            print(f"Raw cluster labels: {unique_raw}")
            print(f"Raw cluster sizes: {counts_raw}")
        
        # CRITICAL FIX: Create proper regime mapping
        # Map cluster labels to sequential regime IDs (0, 1, 2, ...)
        unique_clusters = np.unique(raw_regimes)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Remove any invalid labels
        
        # Sort by cluster size (largest first) for better interpretability
        cluster_sizes = [(cluster, np.sum(raw_regimes == cluster)) for cluster in unique_clusters]
        cluster_sizes.sort(key=lambda x: x[1], reverse=True)
        
        self.regime_label_mapping = {}
        self.active_regimes = []
        
        for new_id, (old_cluster, size) in enumerate(cluster_sizes):
            if size >= self.min_regime_size:  # Only include sufficiently large regimes
                self.regime_label_mapping[old_cluster] = new_id
                self.active_regimes.append(new_id)
        
        if self.verbose:
            print(f"Regime label mapping: {self.regime_label_mapping}")
            print(f"Active regimes: {self.active_regimes}")
        
        # Apply mapping to get final regime assignments
        mapped_regimes = np.full(len(raw_regimes), -1, dtype=int)  # Default to -1 (invalid)
        
        for old_label, new_label in self.regime_label_mapping.items():
            mask = raw_regimes == old_label
            mapped_regimes[mask] = new_label
        
        # Handle samples that didn't map to valid regimes
        invalid_mask = mapped_regimes == -1
        if np.sum(invalid_mask) > 0:
            if self.verbose:
                print(f"WARNING: {np.sum(invalid_mask)} samples have invalid regime assignments")
                print("Assigning them to the largest regime...")
            
            # Assign invalid samples to the largest regime (regime 0)
            if len(self.active_regimes) > 0:
                mapped_regimes[invalid_mask] = self.active_regimes[0]
            else:
                # Emergency fallback
                mapped_regimes[invalid_mask] = 0
                self.active_regimes = [0]
                self.regime_label_mapping[0] = 0
        
        # Store regime statistics
        for regime_id in self.active_regimes:
            regime_mask = mapped_regimes == regime_id
            regime_targets = y[regime_mask]
            
            self.regime_stats[regime_id] = {
                'size': np.sum(regime_mask),
                'positive_rate': np.mean(regime_targets) if len(regime_targets) > 0 else 0,
                'target_distribution': np.bincount(regime_targets.astype(int)) if len(regime_targets) > 0 else [0, 0]
            }
        
        if self.verbose:
            print(f"\nFinal regime assignments:")
            unique_mapped, counts_mapped = np.unique(mapped_regimes, return_counts=True)
            for regime_id, count in zip(unique_mapped, counts_mapped):
                if regime_id in self.regime_stats:
                    stats = self.regime_stats[regime_id]
                    print(f"  Regime {regime_id}: {count} samples ({count/len(mapped_regimes)*100:.1f}%)")
                    print(f"    - Positive rate: {stats['positive_rate']:.3f}")
                    print(f"    - Target dist: {stats['target_distribution']}")
        
        return mapped_regimes
    
    def _predict_regimes_with_mapping(self, X):
        """
        CRITICAL FIX: Predict regimes with proper mapping applied
        """
        if not hasattr(self, 'regime_clusterer') or self.regime_clusterer is None:
            raise ValueError("Regime clusterer not fitted!")
        
        # Get leaf features
        leaf_indices = self._get_tree_leaf_features(X)
        
        # Get raw cluster predictions
        raw_predictions = self.regime_clusterer.predict(leaf_indices)
        
        # Apply the learned mapping
        mapped_predictions = np.full(len(raw_predictions), -1, dtype=int)
        
        for old_label, new_label in self.regime_label_mapping.items():
            mask = raw_predictions == old_label
            mapped_predictions[mask] = new_label
        
        # Handle any unmapped predictions (assign to largest regime)
        invalid_mask = mapped_predictions == -1
        if np.sum(invalid_mask) > 0:
            if self.verbose:
                print(f"WARNING: {np.sum(invalid_mask)} test samples have invalid regime predictions")
            if len(self.active_regimes) > 0:
                mapped_predictions[invalid_mask] = self.active_regimes[0]
            else:
                mapped_predictions[invalid_mask] = 0
        
        return mapped_predictions
    
    def _build_neural_network(self, input_dim, regime_size=None, is_global=False):
        """Build neural network with adaptive architecture"""
        
        if is_global:
            layers = [256, 128, 64, 32]
            dropout = 0.2
            lr = 0.001
        else:
            if regime_size and regime_size < 200:
                layers = [32, 16]
                dropout = 0.4
                lr = 0.0005
            elif regime_size and regime_size < 500:
                layers = [64, 32]
                dropout = 0.3
                lr = 0.001
            else:
                layers = [128, 64, 32]
                dropout = 0.2
                lr = 0.001
        
        model = Sequential()
        model.add(Dense(layers[0], activation='relu', input_shape=(input_dim,)))
        model.add(Dropout(dropout))
        
        for layer_size in layers[1:]:
            model.add(Dense(layer_size, activation='relu'))
            model.add(Dropout(dropout))
        
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _find_optimal_threshold(self, model, X_val, y_val):
        """Find optimal threshold on validation data"""
        val_probs = model.predict(X_val, verbose=0).flatten()
        
        thresholds = np.linspace(0.1, 0.9, 81)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (val_probs > threshold).astype(int)
            
            if len(np.unique(y_val)) < 2:
                continue
                
            score = f1_score(y_val, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score
    
    def _train_models(self, regimes, X_scaled, y, X_val_scaled=None, y_val=None):
        """Train both regime-specific models and global fallback model"""
        
        if self.verbose:
            print("\n" + "=" * 60)
            print("STEP 3: TRAINING SPECIALIZED MODELS")
            print("=" * 60)
        
        # Train global fallback model
        if self.use_global_fallback:
            if self.verbose:
                print("Training global fallback model...")
            
            self.global_model = self._build_neural_network(
                X_scaled.shape[1], 
                regime_size=len(X_scaled), 
                is_global=True
            )
            
            callbacks = [
                EarlyStopping(monitor='val_loss' if X_val_scaled is not None else 'loss', 
                            patience=15, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(monitor='val_loss' if X_val_scaled is not None else 'loss',
                                patience=8, factor=0.5, min_lr=1e-6, verbose=0)
            ]
            
            val_data = (X_val_scaled, y_val) if X_val_scaled is not None else None
            
            self.global_model.fit(
                X_scaled, y,
                validation_data=val_data,
                epochs=80,
                batch_size=64,
                callbacks=callbacks,
                verbose=0
            )
            
            # Set global threshold
            if X_val_scaled is not None:
                self.global_threshold, _ = self._find_optimal_threshold(
                    self.global_model, X_val_scaled, y_val
                )
            
            if self.verbose:
                print(f"Global model trained (threshold: {self.global_threshold:.3f})")
        
        # Train regime-specific models
        if self.verbose:
            print(f"\nTraining {len(self.active_regimes)} regime-specific models...")
        
        for regime_id in self.active_regimes:
            regime_mask = regimes == regime_id
            X_r = X_scaled[regime_mask]
            y_r = y[regime_mask]
            
            if len(X_r) < 20:
                if self.verbose:
                    print(f"Skipping regime {regime_id} (only {len(X_r)} samples)")
                continue
            
            if self.verbose:
                print(f"  Training regime {regime_id} model ({len(X_r)} samples)...")
            
            # Get validation data for this regime
            X_val_r, y_val_r = None, None
            if X_val_scaled is not None and y_val is not None:
                val_regimes = self._predict_regimes_with_mapping(X_val_scaled)
                val_regime_mask = val_regimes == regime_id
                if np.sum(val_regime_mask) > 5:
                    X_val_r = X_val_scaled[val_regime_mask]
                    y_val_r = y_val[val_regime_mask]
            
            # Build and train model
            model_r = self._build_neural_network(X_r.shape[1], len(X_r))
            
            epochs = min(100, max(30, len(X_r) // 4))
            batch_size = min(32, max(8, len(X_r) // 8))
            
            callbacks = [
                EarlyStopping(monitor='val_loss' if X_val_r is not None else 'loss',
                            patience=10, restore_best_weights=True, verbose=0)
            ]
            
            val_data = (X_val_r, y_val_r) if X_val_r is not None else None
            
            model_r.fit(
                X_r, y_r,
                validation_data=val_data,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )
            
            # Find optimal threshold
            if X_val_r is not None and len(np.unique(y_val_r)) > 1:
                optimal_threshold, val_score = self._find_optimal_threshold(
                    model_r, X_val_r, y_val_r
                )
            else:
                # Fallback to training data
                train_probs = model_r.predict(X_r, verbose=0).flatten()
                optimal_threshold = self._find_threshold_on_train(y_r, train_probs)
                val_score = 0.0
            
            self.regime_models[regime_id] = model_r
            self.regime_thresholds[regime_id] = optimal_threshold
            
            if self.verbose:
                print(f"    - Threshold: {optimal_threshold:.3f}, Val score: {val_score:.3f}")
        
        if self.verbose:
            print(f"Model training completed! Trained {len(self.regime_models)} regime models.")
    
    def _find_threshold_on_train(self, y_true, y_probs):
        """Fallback threshold finding on training data"""
        thresholds = np.linspace(0.1, 0.9, 41)
        best_threshold = 0.5
        best_score = 0
        
        for threshold in thresholds:
            y_pred = (y_probs > threshold).astype(int)
            score = f1_score(y_true, y_pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Enhanced training pipeline with proper regime mapping"""
        
        if self.verbose:
            print("=" * 80)
            print("FIXED HYBRID REGIME-SWITCHING MODEL TRAINING")
            print("=" * 80)
            print(f"Training data: {X_train.shape}")
            if X_val is not None:
                print(f"Validation data: {X_val.shape}")
        
        # Prepare targets
        y_train = (y_train > 0.5).astype(int) if np.max(y_train) <= 1 else (y_train > np.median(y_train)).astype(int)
        if y_val is not None:
            y_val = (y_val > 0.5).astype(int) if np.max(y_val) <= 1 else (y_val > np.median(y_val)).astype(int)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Discover regimes with proper mapping
        regimes_train = self._discover_regimes_with_mapping(X_train, y_train)
        
        # Train models
        self._train_models(regimes_train, X_scaled, y_train, X_val_scaled, y_val)
        
        self.is_fitted = True
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("TRAINING COMPLETED!")
            print("=" * 80)
            print(f"Active regimes: {self.active_regimes}")
            print(f"Trained models: {list(self.regime_models.keys())}")
            print("=" * 80)
        
        return self
    
    def predict(self, X_test, return_details=True):
        """Enhanced prediction with detailed regime tracking"""
        
        if not self.is_fitted:
            raise ValueError("Pipeline not fitted!")
        
        if self.verbose:
            print(f"Predicting on {len(X_test)} test samples...")
        
        X_scaled = self.scaler.transform(X_test)
        regimes = self._predict_regimes_with_mapping(X_test)
        
        if self.verbose:
            print(f"Test regime distribution:")
            unique_regimes, counts = np.unique(regimes, return_counts=True)
            for regime_id, count in zip(unique_regimes, counts):
                percentage = (count / len(regimes)) * 100
                print(f"  Regime {regime_id}: {count} samples ({percentage:.1f}%)")
        
        predictions = []
        probabilities = []
        models_used = []
        confidence_scores = []
        
        for i, regime in enumerate(regimes):
            sample = X_scaled[i:i+1]
            
            # Try regime-specific model first
            if regime in self.regime_models and regime in self.regime_thresholds:
                model = self.regime_models[regime]
                threshold = self.regime_thresholds[regime]
                prob = model.predict(sample, verbose=0)[0][0]
                confidence = abs(prob - threshold)
                model_used = f'regime_{regime}'
                
            # Fallback to global model
            elif self.use_global_fallback and self.global_model is not None:
                prob = self.global_model.predict(sample, verbose=0)[0][0]
                threshold = self.global_threshold
                confidence = abs(prob - threshold)
                model_used = 'global_fallback'
                
            else:
                # Last resort: use any available regime model
                if self.regime_models:
                    fallback_regime = list(self.regime_models.keys())[0]
                    model = self.regime_models[fallback_regime]
                    threshold = self.regime_thresholds[fallback_regime]
                    prob = model.predict(sample, verbose=0)[0][0]
                    confidence = abs(prob - threshold)
                    model_used = f'fallback_{fallback_regime}'
                else:
                    raise ValueError("No models available for prediction!")
            
            pred = 1 if prob > threshold else 0
            predictions.append(pred)
            probabilities.append(prob)
            models_used.append(model_used)
            confidence_scores.append(confidence)
        
        if self.verbose:
            # Detailed model usage analysis
            model_usage = {}
            for model_type in models_used:
                model_usage[model_type] = model_usage.get(model_type, 0) + 1
            
            print(f"\nModel usage summary:")
            for model_type, count in model_usage.items():
                percentage = (count / len(models_used)) * 100
                print(f"  {model_type}: {count} samples ({percentage:.1f}%)")
        
        results = {
            'predictions': np.array(predictions),
            'probabilities': np.array(probabilities),
            'regimes': regimes,
            'models_used': models_used,
            'confidence': np.array(confidence_scores)
        }
        
        if return_details:
            return results
        else:
            return results['predictions']
    
    def evaluate_with_detailed_regime_analysis(self, X_test, y_test):
        """
        ENHANCED: Comprehensive evaluation with detailed regime-level analysis
        """
        if self.verbose:
            print(f"\n" + "=" * 80)
            print("DETAILED EVALUATION WITH REGIME ANALYSIS")
            print("=" * 80)
        
        y_test = (y_test > 0.5).astype(int) if np.max(y_test) <= 1 else (y_test > np.median(y_test)).astype(int)
        
        # Get predictions with full details
        results = self.predict(X_test, return_details=True)
        y_pred = results['predictions']
        y_probs = results['probabilities']
        regimes = results['regimes']
        models_used = results['models_used']
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_probs) if len(np.unique(y_test)) > 1 else 0.5
        }
        
        # Model usage analysis
        model_usage = {}
        for model_type in models_used:
            model_usage[model_type] = model_usage.get(model_type, 0) + 1
        
        # Per-regime detailed analysis
        regime_analysis = {}
        unique_regimes = np.unique(regimes)
        
        for regime_id in unique_regimes:
            mask = regimes == regime_id
            if np.sum(mask) == 0:
                continue
                
            regime_y_test = y_test[mask]
            regime_y_pred = y_pred[mask]
            regime_y_probs = y_probs[mask]
            regime_models = np.array(models_used)[mask]
            
            # Calculate metrics for this regime
            regime_acc = accuracy_score(regime_y_test, regime_y_pred)
            regime_precision = precision_score(regime_y_test, regime_y_pred, zero_division=0)
            regime_recall = recall_score(regime_y_test, regime_y_pred, zero_division=0)
            regime_f1 = f1_score(regime_y_test, regime_y_pred, zero_division=0)
            regime_auc = roc_auc_score(regime_y_test, regime_y_probs) if len(np.unique(regime_y_test)) > 1 else 0.5
            
            # Model usage within this regime
            regime_model_usage = {}
            for model in regime_models:
                regime_model_usage[model] = regime_model_usage.get(model, 0) + 1
            
            regime_analysis[regime_id] = {
                'sample_count': np.sum(mask),
                'percentage': (np.sum(mask) / len(regimes)) * 100,
                'target_distribution': np.bincount(regime_y_test.astype(int)),
                'metrics': {
                    'accuracy': regime_acc,
                    'precision': regime_precision,
                    'recall': regime_recall,
                    'f1_score': regime_f1,
                    'roc_auc': regime_auc
                },
                'model_usage': regime_model_usage,
                'avg_confidence': np.mean(results['confidence'][mask]),
                'prediction_distribution': np.bincount(regime_y_pred.astype(int))
            }
        
        # Print detailed results
        if self.verbose:
            print("OVERALL PERFORMANCE:")
            for metric, value in overall_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            print(f"\nGLOBAL MODEL USAGE:")
            total_samples = sum(model_usage.values())
            for model_type, count in model_usage.items():
                percentage = (count / total_samples) * 100
                print(f"  {model_type}: {count} samples ({percentage:.1f}%)")
            
            print(f"\nPER-REGIME DETAILED ANALYSIS:")
            print("=" * 80)
            
            for regime_id in sorted(regime_analysis.keys()):
                analysis = regime_analysis[regime_id]
                print(f"\nREGIME {regime_id}:")
                print(f"  Sample count: {analysis['sample_count']} ({analysis['percentage']:.1f}% of test set)")
                print(f"  Target distribution: {analysis['target_distribution']}")
                print(f"  Prediction distribution: {analysis['prediction_distribution']}")
                print(f"  Average confidence: {analysis['avg_confidence']:.3f}")
                
                print(f"  Performance metrics:")
                for metric, value in analysis['metrics'].items():
                    print(f"    {metric}: {value:.4f}")
                
                print(f"  Model usage within regime:")
                for model, count in analysis['model_usage'].items():
                    regime_percentage = (count / analysis['sample_count']) * 100
                    print(f"    {model}: {count} samples ({regime_percentage:.1f}%)")
                print("-" * 40)
            
            print("=" * 80)
        
        return {
            'overall_metrics': overall_metrics,
            'model_usage': model_usage,
            'regime_analysis': regime_analysis,
            'detailed_results': results
        }
    
    def save_model(self, filepath, include_metadata=True):
    
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted pipeline. Train the model first.")
        
        import pickle
        import json
        from pathlib import Path
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"Saving pipeline to {filepath}...")
        
        # Prepare model data
        model_data = {
            # Core pipeline configuration
            'config': {
                'n_regimes': self.n_regimes,
                'tree_method': self.tree_method,
                'use_balanced_kmeans': self.use_balanced_kmeans,
                'min_regime_size': self.min_regime_size,
                'ensemble_prediction': self.ensemble_prediction,
                'confidence_threshold': self.confidence_threshold,
                'use_global_fallback': self.use_global_fallback,
                'regime_consistency_check': self.regime_consistency_check,
                'random_state': self.random_state,
                'verbose': self.verbose
            },
            
            # Core fitted components
            'tree_model': self.tree_model,
            'regime_clusterer': self.regime_clusterer,
            'scaler': self.scaler,
            
            # Regime mapping and state
            'regime_label_mapping': self.regime_label_mapping,
            'active_regimes': self.active_regimes,
            'regime_thresholds': self.regime_thresholds,
            'global_threshold': self.global_threshold,
            'is_fitted': self.is_fitted,
            
            # Statistics and metadata
            'regime_stats': self.regime_stats,
            'regime_performance': self.regime_performance
        }
        
        # Save sklearn components and main data
        with open(f"{filepath}_main.pkl", 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if self.verbose:
            print(f"  Saved main components to {filepath}_main.pkl")
        
        # Save neural network models separately (they need special handling)
        neural_models = {}
        
        # Save regime-specific models
        for regime_id, model in self.regime_models.items():
            model_path = f"{filepath}_regime_{regime_id}.keras"
            save_model(model, model_path)
        
            neural_models[f'regime_{regime_id}'] = model_path
            if self.verbose:
                print(f"  Saved regime {regime_id} model to {model_path}")
        
        # Save global model if exists
        if self.global_model is not None:
            global_model_path = f"{filepath}_global.keras"
            self.global_model.save(global_model_path)
            neural_models['global'] = global_model_path
            if self.verbose:
                print(f"  Saved global model to {global_model_path}")
        
        # Save neural model paths mapping
        with open(f"{filepath}_models.json", 'w') as f:
            json.dump(neural_models, f, indent=2)
        
        if self.verbose:
            print(f"  Saved model paths to {filepath}_models.json")
        
        # Save metadata if requested
        if include_metadata:
            metadata = {
                'training_timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else str(np.datetime64('now')),
                'regime_count': len(self.active_regimes),
                'total_models_saved': len(self.regime_models) + (1 if self.global_model else 0),
                'active_regimes': self.active_regimes,
                'regime_statistics': self.regime_stats,
                'model_architecture_summary': self._get_model_architecture_summary()
            }
            
            with open(f"{filepath}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            if self.verbose:
                print(f"  Saved metadata to {filepath}_metadata.json")
        
        # Create a summary file with loading instructions
        summary = f"""
Hybrid Regime Pipeline Model Files
==================================

Model saved successfully with the following components:

Main Components:
- {filepath}_main.pkl (sklearn models, scaler, regime mappings)
- {filepath}_models.json (neural network model paths)

Neural Network Models:
"""
        for model_name, model_path in neural_models.items():
            summary += f"- {model_path} ({model_name})\n"
        
        if include_metadata:
            summary += f"- {filepath}_metadata.json (training metadata)\n"
        
        summary += f"""
Configuration:
- Regimes: {len(self.active_regimes)} active regimes {self.active_regimes}
- Tree method: {self.tree_method}
- Global fallback: {'Enabled' if self.use_global_fallback else 'Disabled'}

To load this model, use:
    pipeline = FixedHybridRegimePipeline.load_model('{filepath}')
"""
        
        with open(f"{filepath}_README.txt", 'w') as f:
            f.write(summary)
        
        if self.verbose:
            print(f"  Created README at {filepath}_README.txt")
            print(f"\nModel saved successfully!")
            print(f"  Total files created: {3 + len(neural_models) + (1 if include_metadata else 0)}")
            print(f"  Active regimes: {self.active_regimes}")
            print(f"  Models saved: {len(self.regime_models)} regime + {1 if self.global_model else 0} global")
    
    @classmethod
    def load_model(cls, filepath, verbose=True):
        """
        Load a saved pipeline from disk
        
        Args:
            filepath (str): Base filepath (without extension) used when saving
            verbose (bool): Whether to print loading progress
            
        Returns:
            FixedHybridRegimePipeline: Loaded and ready-to-use pipeline
        """
        import pickle
        import json
        from pathlib import Path
        
        if verbose:
            print(f"Loading pipeline from {filepath}...")
        
        # Check if main file exists
        main_file = f"{filepath}_main.pkl"
        if not Path(main_file).exists():
            #raise FileNotFoundError(f"Main model file not found: {main_file}")
            return 0
        # Load main components
        with open(main_file, 'rb') as f:
            model_data = pickle.load(f)
        
        if verbose:
            print(f"  Loaded main components from {main_file}")
        
        # Create new instance with original configuration
        pipeline = cls(**model_data['config'])
        
        # Restore fitted components
        pipeline.tree_model = model_data['tree_model']
        pipeline.regime_clusterer = model_data['regime_clusterer']
        pipeline.scaler = model_data['scaler']
        
        # Restore regime mapping and state
        pipeline.regime_label_mapping = model_data['regime_label_mapping']
        pipeline.active_regimes = model_data['active_regimes']
        pipeline.regime_thresholds = model_data['regime_thresholds']
        pipeline.global_threshold = model_data['global_threshold']
        pipeline.is_fitted = model_data['is_fitted']
        
        # Restore statistics
        pipeline.regime_stats = model_data['regime_stats']
        pipeline.regime_performance = model_data['regime_performance']
        
        if verbose:
            print(f"  Restored pipeline configuration and state")
            print(f"    - Active regimes: {pipeline.active_regimes}")
            print(f"    - Regime mappings: {len(pipeline.regime_label_mapping)}")
        
        # Load neural network model paths
        models_file = f"{filepath}_models.json"
        if Path(models_file).exists():
            with open(models_file, 'r') as f:
                neural_models = json.load(f)
            
            if verbose:
                print(f"  Loading {len(neural_models)} neural network models...")
            
            # Load regime-specific models
            for model_name, model_path in neural_models.items():
                if model_name.startswith('regime_'):
                    regime_id = int(model_name.split('_')[1])
                    if Path(model_path).exists():
                        pipeline.regime_models[regime_id] = load_model(model_path)
                        if verbose:
                            print(f"    Loaded regime {regime_id} model from {model_path}")
                    else:
                        print(f"    Warning: Regime model file not found: {model_path}")
                
                elif model_name == 'global':
                    if Path(model_path).exists():
                        pipeline.global_model = load_model(model_path)
                        if verbose:
                            print(f"    Loaded global model from {model_path}")
                    else:
                        print(f"    Warning: Global model file not found: {model_path}")
        
        else:
            print(f"  Warning: Neural models file not found: {models_file}")
        
        # Verify model integrity
        missing_models = []
        for regime_id in pipeline.active_regimes:
            if regime_id not in pipeline.regime_models:
                missing_models.append(regime_id)
        
        if missing_models:
            print(f"  Warning: Missing models for regimes: {missing_models}")
        
        if verbose:
            print(f"  Model loading completed!")
            print(f"    - Regime models loaded: {len(pipeline.regime_models)}")
            print(f"    - Global model: {'Loaded' if pipeline.global_model else 'Not found'}")
            print(f"    - Pipeline ready for prediction: {pipeline.is_fitted}")
        
        return pipeline
    
    def _get_model_architecture_summary(self):
        """Get summary of neural network architectures"""
        summary = {}
        
        # Regime models
        for regime_id, model in self.regime_models.items():
            summary[f'regime_{regime_id}'] = {
                'layers': len(model.layers),
                'parameters': model.count_params(),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
        
        # Global model
        if self.global_model:
            summary['global'] = {
                'layers': len(self.global_model.layers),
                'parameters': self.global_model.count_params(),
                'input_shape': self.global_model.input_shape,
                'output_shape': self.global_model.output_shape
            }
        
        return summary


# Enhanced BalancedKMeans
class BalancedKMeans:
    """Balanced KMeans with proper label handling"""
    
    def __init__(self, n_clusters, min_cluster_size=None, random_state=42, verbose=False):
        self.n_clusters = n_clusters
        self.min_cluster_size = min_cluster_size or 50
        self.random_state = random_state
        self.verbose = verbose
        self.cluster_centers_ = None
        self.labels_ = None
        
    def fit(self, X):
        if self.verbose:
            print(f"    Fitting balanced KMeans with {self.n_clusters} clusters...")
        
        # Initial clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=20)
        initial_labels = kmeans.fit_predict(X)
        
        unique_labels, counts = np.unique(initial_labels, return_counts=True)
        small_clusters = unique_labels[counts < self.min_cluster_size]
        
        if self.verbose:
            print(f"      Initial cluster sizes: {dict(zip(unique_labels, counts))}")
            if len(small_clusters) > 0:
                print(f"      Small clusters to be merged: {small_clusters}")
        
        if len(small_clusters) > 0:
            final_labels = initial_labels.copy()
            centers = kmeans.cluster_centers_
            
            # Merge small clusters into nearest large ones
            for small_cluster in small_clusters:
                small_mask = initial_labels == small_cluster
                small_center = centers[small_cluster]
                
                # Find nearest large cluster
                distances = []
                valid_clusters = []
                for i, center in enumerate(centers):
                    if i not in small_clusters:
                        distances.append(np.linalg.norm(small_center - center))
                        valid_clusters.append(i)
                
                if valid_clusters:
                    nearest_cluster = valid_clusters[np.argmin(distances)]
                    final_labels[small_mask] = nearest_cluster
                    if self.verbose:
                        print(f"        Merged cluster {small_cluster} into cluster {nearest_cluster}")
            
            self.labels_ = final_labels
            # Recalculate centers for final clusters
            unique_final = np.unique(final_labels)
            self.cluster_centers_ = np.array([X[final_labels == i].mean(axis=0) for i in unique_final])
            
        else:
            self.labels_ = initial_labels
            self.cluster_centers_ = kmeans.cluster_centers_
        
        if self.verbose:
            final_unique, final_counts = np.unique(self.labels_, return_counts=True)
            print(f"      Final cluster sizes: {dict(zip(final_unique, final_counts))}")
        
        return self
    
    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted!")
        
        # Calculate distances to all cluster centers
        distances = np.array([[np.linalg.norm(x - center) for center in self.cluster_centers_] for x in X])
        return np.argmin(distances, axis=1)
    
    def fit_predict(self, X):
        return self.fit(X).labels_


# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("TESTING FIXED REGIME MAPPING PIPELINE")
    print("=" * 80)
    
    # Generate sample data with clear regime structure
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    print(f"Generating test data:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    
    X = np.random.randn(n_samples, n_features)
    
    # Create three distinct regimes with different patterns
    regime_1_mask = (X[:, 0] > 0) & (X[:, 1] > 0)  # Quadrant 1
    regime_2_mask = (X[:, 0] <= 0) & (X[:, 1] > 0)  # Quadrant 2
    regime_3_mask = X[:, 1] <= 0  # Bottom half
    
    y = np.zeros(n_samples)
    # Different patterns for each regime
    y[regime_1_mask] = (X[regime_1_mask, 0] + X[regime_1_mask, 1] > 0.5).astype(int)
    y[regime_2_mask] = (X[regime_2_mask, 2] > 0).astype(int) 
    y[regime_3_mask] = (X[regime_3_mask, 0] * X[regime_3_mask, 2] > 0).astype(int)
    
    print(f"True regime structure:")
    print(f"  Regime 1: {np.sum(regime_1_mask)} samples")
    print(f"  Regime 2: {np.sum(regime_2_mask)} samples")
    print(f"  Regime 3: {np.sum(regime_3_mask)} samples")
    print(f"  Overall target distribution: {np.bincount(y.astype(int))}")
    
    # Split data
    train_idx = int(0.6 * n_samples)
    val_idx = int(0.8 * n_samples)
    
    X_train, X_val, X_test = X[:train_idx], X[train_idx:val_idx], X[val_idx:]
    y_train, y_val, y_test = y[:train_idx], y[train_idx:val_idx], y[val_idx:]
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    print("\n" + "=" * 80)
    print("TESTING FIXED PIPELINE")
    print("=" * 80)
    
    # Test both tree methods
    for tree_method in ['random_forest', 'gradient_boosting']:
        print(f"\n{'-'*60}")
        print(f"TESTING WITH {tree_method.upper()}")
        print(f"{'-'*60}")
        
        # Initialize fixed pipeline
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
        evaluation_results = pipeline.evaluate_with_detailed_regime_analysis(X_test, y_test)
        
        print(f"\n{tree_method.upper()} SUMMARY:")
        print(f"  Overall F1 Score: {evaluation_results['overall_metrics']['f1_score']:.4f}")
        print(f"  Overall Accuracy: {evaluation_results['overall_metrics']['accuracy']:.4f}")
        print(f"  Number of regimes discovered: {len(evaluation_results['regime_analysis'])}")
        
        # Check if regime mapping is working properly
        regime_usage = {}
        for model_type, count in evaluation_results['model_usage'].items():
            if model_type.startswith('regime_'):
                regime_id = model_type.split('_')[1]
                regime_usage[regime_id] = count
        
        print(f"  Regime model usage:")
        for regime_id, count in regime_usage.items():
            print(f"    Regime {regime_id}: {count} samples")
        
        if len(regime_usage) > 1:
            print("  ✓ Multiple regimes are being used - mapping working correctly!")
        else:
            print("  ⚠ Only one regime being used - potential mapping issue!")
    
    print("\n" + "=" * 80)
    print("TESTING COMPLETED!")
    print("Key fixes implemented:")
    print("  ✓ Proper regime label mapping after clustering")
    print("  ✓ Consistent regime assignment between train/test")
    print("  ✓ Enhanced regime tracking during evaluation")
    print("  ✓ Fixed leaf feature extraction for both tree methods")
    print("  ✓ Detailed regime-level performance analysis")
    print("  ✓ Better handling of invalid regime assignments")
    
    # Test save/load functionality
    print("\n" + "=" * 80)
    print("TESTING SAVE/LOAD FUNCTIONALITY")
    print("=" * 80)
    
    # Use the last trained pipeline for save/load testing
    save_path = "test_model/my_regime_pipeline"
    
    print(f"Saving pipeline to {save_path}...")
    pipeline.save_model(save_path, include_metadata=True)
    
    print(f"\nLoading pipeline from {save_path}...")
    loaded_pipeline = FixedHybridRegimePipeline.load_model(save_path, verbose=True)
    
    # Test predictions with loaded model
    print(f"\nTesting loaded pipeline predictions...")
    original_results = pipeline.predict(X_test[:10], return_details=True)
    loaded_results = loaded_pipeline.predict(X_test[:10], return_details=True)
    
    # Compare predictions
    predictions_match = np.array_equal(original_results['predictions'], loaded_results['predictions'])
    probabilities_match = np.allclose(original_results['probabilities'], loaded_results['probabilities'], rtol=1e-5)
    regimes_match = np.array_equal(original_results['regimes'], loaded_results['regimes'])
    
    print(f"Validation of loaded model:")
    print(f"  Predictions match: {predictions_match}")
    print(f"  Probabilities match: {probabilities_match}")
    print(f"  Regime assignments match: {regimes_match}")
    
    if predictions_match and probabilities_match and regimes_match:
        print("  ✓ Save/load functionality working perfectly!")
    else:
        print("  ⚠ Some discrepancies found in loaded model")
        
    # Quick evaluation comparison
    print(f"\nBrief evaluation comparison:")
    original_eval = pipeline.evaluate_with_detailed_regime_analysis(X_test, y_test)
    loaded_eval = loaded_pipeline.evaluate_with_detailed_regime_analysis(X_test, y_test)
    
    print(f"  Original F1: {original_eval['overall_metrics']['f1_score']:.4f}")
    print(f"  Loaded F1: {loaded_eval['overall_metrics']['f1_score']:.4f}")
    print(f"  Difference: {abs(original_eval['overall_metrics']['f1_score'] - loaded_eval['overall_metrics']['f1_score']):.6f}")
    
    print("=" * 80)