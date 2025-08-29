from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, balanced_accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

class ThresholdOptimizer:
   
    
    def __init__(self):
        self.optimal_threshold = 0.5
        self.method_used = None
        self.threshold_metrics = {}
    
    def youden_threshold(self, y_true, y_probs):
        
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, youden_scores[optimal_idx]
    
    def balanced_accuracy_threshold(self, y_true, y_probs):
        """
        Maximizes balanced accuracy: (sensitivity + specificity) / 2
        Explicitly balances both classes
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        balanced_accuracies = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            balanced_acc = balanced_accuracy_score(y_true, y_pred)
            balanced_accuracies.append(balanced_acc)
        
        optimal_idx = np.argmax(balanced_accuracies)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, balanced_accuracies[optimal_idx]
    
    def geometric_mean_threshold(self, y_true, y_probs):
        """
        Maximizes geometric mean of sensitivity and specificity
        sqrt(sensitivity * specificity) - good for imbalanced datasets
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        specificity = 1 - fpr
        geometric_means = np.sqrt(tpr * specificity)
        optimal_idx = np.argmax(geometric_means)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, geometric_means[optimal_idx]
    
    def cost_sensitive_threshold(self, y_true, y_probs, fp_cost=1.0, fn_cost=1.0):
        """
        Cost-sensitive threshold optimization
        fp_cost: cost of false positive
        fn_cost: cost of false negative
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        total_costs = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            # Calculate confusion matrix components
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            tp = np.sum((y_true == 1) & (y_pred == 1))
            
            # Calculate total cost
            total_cost = fp * fp_cost + fn * fn_cost
            total_costs.append(total_cost)
        
        optimal_idx = np.argmin(total_costs)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, total_costs[optimal_idx]
    
    def precision_recall_f1_threshold(self, y_true, y_probs, beta=1.0):
        """
        Optimizes F-beta score (F1 when beta=1)
        beta > 1 emphasizes recall, beta < 1 emphasizes precision
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Calculate F-beta scores
        beta_squared = beta ** 2
        f_beta_scores = (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)
        
        # Handle NaN values
        f_beta_scores = np.nan_to_num(f_beta_scores)
        
        optimal_idx = np.argmax(f_beta_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return optimal_threshold, f_beta_scores[optimal_idx]
    
    def class_balance_threshold(self, y_true, y_probs):
        """
        Finds threshold that minimizes the difference in class predictions
        Aims for 50-50 split in predictions
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        balance_scores = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            class_1_ratio = np.mean(y_pred)
            # Score is better when closer to 0.5 (balanced)
            balance_score = 1 - abs(class_1_ratio - 0.5) * 2
            balance_scores.append(balance_score)
        
        optimal_idx = np.argmax(balance_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        return optimal_threshold, balance_scores[optimal_idx]
    
    def find_optimal_threshold(self, y_true, y_probs, method='balanced_accuracy', **kwargs):
        """
        Main method to find optimal threshold using specified method
        
        Args:
            y_true: True binary labels
            y_probs: Predicted probabilities
            method: Threshold selection method
                   - 'youden': Youden's J statistic
                   - 'balanced_accuracy': Balanced accuracy
                   - 'geometric_mean': Geometric mean of sens/spec
                   - 'cost_sensitive': Cost-sensitive optimization
                   - 'f1': F1-score optimization
                   - 'class_balance': Aims for 50-50 prediction split
                   - 'combined': Uses multiple methods and averages
            **kwargs: Additional parameters for specific methods
        """
        
        if method == 'youden':
            threshold, score = self.youden_threshold(y_true, y_probs)
            
        elif method == 'balanced_accuracy':
            threshold, score = self.balanced_accuracy_threshold(y_true, y_probs)
            
        elif method == 'geometric_mean':
            threshold, score = self.geometric_mean_threshold(y_true, y_probs)
            
        elif method == 'cost_sensitive':
            fp_cost = kwargs.get('fp_cost', 1.0)
            fn_cost = kwargs.get('fn_cost', 1.0)
            threshold, score = self.cost_sensitive_threshold(y_true, y_probs, fp_cost, fn_cost)
            
        elif method == 'f1':
            beta = kwargs.get('beta', 1.0)
            threshold, score = self.precision_recall_f1_threshold(y_true, y_probs, beta)
            
        elif method == 'class_balance':
            threshold, score = self.class_balance_threshold(y_true, y_probs)
            
        elif method == 'combined':
            # Use multiple methods and combine results
            thresholds = []
            methods_to_use = ['youden', 'balanced_accuracy', 'geometric_mean', 'f1']
            
            for m in methods_to_use:
                thresh, _ = self.find_optimal_threshold(y_true, y_probs, method=m)
                thresholds.append(thresh)
            
            # Use median of all thresholds
            threshold = np.median(thresholds)
            score = self.evaluate_threshold(y_true, y_probs, threshold)['balanced_accuracy']
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.optimal_threshold = threshold
        self.method_used = method
        
        print(f"🎯 Optimal Threshold ({method}): {threshold:.3f}")
        print(f"📊 Method Score: {score:.4f}")
        
        return threshold
    
    def evaluate_threshold(self, y_true, y_probs, threshold):
        """
        Comprehensive evaluation of a threshold
        """
        y_pred = (y_probs >= threshold).astype(int)
        
        # Calculate metrics
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = sensitivity
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        balanced_acc = (sensitivity + specificity) / 2
        
        class_1_ratio = np.mean(y_pred)
        
        metrics = {
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'balanced_accuracy': balanced_acc,
            'class_1_prediction_ratio': class_1_ratio,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        
        return metrics
    
    def compare_methods(self, y_true, y_probs):
        """
        Compare all threshold selection methods
        """
        methods = ['youden', 'balanced_accuracy', 'geometric_mean', 'f1', 'class_balance']
        results = {}
        
        print("🔍 Comparing Threshold Selection Methods:")
        print("-" * 80)
        
        for method in methods:
            threshold = self.find_optimal_threshold(y_true, y_probs, method=method)
            metrics = self.evaluate_threshold(y_true, y_probs, threshold)
            results[method] = metrics
            
            print(f"\n{method.upper()}:")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  Class 1 Ratio: {metrics['class_1_prediction_ratio']:.4f}")
        
        return results
    
    def plot_threshold_analysis(self, y_true, y_probs):
        """
        Plot threshold analysis visualization
        """
        thresholds = np.arange(0.01, 1.00, 0.01)
        metrics_over_thresholds = {
            'sensitivity': [], 'specificity': [], 'f1_score': [], 
            'balanced_accuracy': [], 'precision': [], 'class_1_ratio': []
        }
        
        for threshold in thresholds:
            metrics = self.evaluate_threshold(y_true, y_probs, threshold)
            metrics_over_thresholds['sensitivity'].append(metrics['sensitivity'])
            metrics_over_thresholds['specificity'].append(metrics['specificity'])
            metrics_over_thresholds['f1_score'].append(metrics['f1_score'])
            metrics_over_thresholds['balanced_accuracy'].append(metrics['balanced_accuracy'])
            metrics_over_thresholds['precision'].append(metrics['precision'])
            metrics_over_thresholds['class_1_ratio'].append(metrics['class_1_prediction_ratio'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Sensitivity vs Specificity
        ax1.plot(thresholds, metrics_over_thresholds['sensitivity'], label='Sensitivity', linewidth=2)
        ax1.plot(thresholds, metrics_over_thresholds['specificity'], label='Specificity', linewidth=2)
        ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Sensitivity vs Specificity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: F1 and Balanced Accuracy
        ax2.plot(thresholds, metrics_over_thresholds['f1_score'], label='F1-Score', linewidth=2)
        ax2.plot(thresholds, metrics_over_thresholds['balanced_accuracy'], label='Balanced Accuracy', linewidth=2)
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Score')
        ax2.set_title('F1-Score vs Balanced Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Precision
        ax3.plot(thresholds, metrics_over_thresholds['precision'], label='Precision', linewidth=2, color='orange')
        ax3.set_xlabel('Threshold')
        ax3.set_ylabel('Precision')
        ax3.set_title('Precision vs Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Class Balance
        ax4.plot(thresholds, metrics_over_thresholds['class_1_ratio'], label='Class 1 Prediction Ratio', linewidth=2, color='red')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Perfect Balance (0.5)')
        ax4.set_xlabel('Threshold')
        ax4.set_ylabel('Class 1 Ratio')
        ax4.set_title('Predicted Class Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig


# Enhanced method for the HybridCryptoPipeline class
def enhanced_find_optimal_thresh(self, y_true, y_probs, method='balanced_accuracy', **kwargs):
    """
    Enhanced threshold optimization method for the HybridCryptoPipeline class
    
    Args:
        y_true: True binary labels
        y_probs: Predicted probabilities
        method: Threshold selection method
               - 'balanced_accuracy': Maximizes balanced accuracy (recommended for balanced classes)
               - 'youden': Youden's J statistic (good for medical/diagnostic applications)
               - 'geometric_mean': Geometric mean of sensitivity and specificity
               - 'cost_sensitive': Cost-sensitive optimization
               - 'f1': F1-score optimization
               - 'class_balance': Aims for 50-50 prediction split
               - 'combined': Uses multiple methods
        **kwargs: Additional parameters (e.g., fp_cost, fn_cost for cost_sensitive)
    
    Returns:
        optimal_threshold: The best threshold value
    """
    
    # Initialize threshold optimizer
    threshold_optimizer = ThresholdOptimizer()
    
    # Find optimal threshold
    optimal_threshold = threshold_optimizer.find_optimal_threshold(
        y_true, y_probs, method=method, **kwargs
    )
    
    # Evaluate the optimal threshold
    metrics = threshold_optimizer.evaluate_threshold(y_true, y_probs, optimal_threshold)
    
    # Print detailed results
    print(f"🎯 Threshold Optimization Results ({method}):")
    print(f"   Optimal Threshold: {optimal_threshold:.3f}")
    print(f"   Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"   Sensitivity (Recall): {metrics['sensitivity']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"   Class 1 Prediction Ratio: {metrics['class_1_prediction_ratio']:.4f}")
    print(f"   Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    
    return optimal_threshold

