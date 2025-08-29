import requests
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
class DataProcess:
    def __init__(self,Sequence,Interval):
        self.Sequence_Length = Sequence
        self.interval = Interval
            
        # In DataProcess.py, modify create_sequences()
    def create_sequences(self, datapoints):
        Sequences = []
        labels = []
        min_change_threshold = 0.00025  # 0.1% minimum price change
        if len(datapoints) == 0:
            return Sequences,labels
        for i in range(0, (len(datapoints) - self.Sequence_Length)):
            Sequence = datapoints[i][:-1]
            current_price = datapoints[i][-1]
            future_price = datapoints[i + 1][-1]
            
            price_change = (future_price - current_price) / current_price
            
            # ✅ PROPER THRESHOLD LOGIC:
            # Only label as 1 (BUY) if price increases MORE than threshold
            # Only label as 0 (SELL) if price decreases MORE than threshold
            # Skip if change is within threshold (neutral)
            
            if price_change > min_change_threshold:
                label = 1  # Strong upward movement
                Sequences.append(Sequence)
                labels.append(label)
            elif price_change < -min_change_threshold:
                label = 0  # Strong downward movement
                Sequences.append(Sequence)
                labels.append(label)
            # Skip cases where -threshold ≤ price_change ≤ threshold
        
        return Sequences, labels
    def create_sequences_with_Thresh(self,datapoints,Threshold):
        Sequences = []
        labels = []
        for i in range(0,(len(datapoints) -self.Sequence_Length)):
            Sequence = datapoints[i][:-1]
            future_price = datapoints[i + 1][-1]   
            label = 1 if future_price > Threshold else 0
            Sequences.append(Sequence)
            labels.append(label)
        return Sequences,labels
        # ------------------------------------------------------------------
    def augment_sequences(self, X, y, factor=2, noise_std=0.005, max_shift=2):
        X, y = np.array(X), np.array(y)           # (samples, timesteps)
        X_aug, y_aug = [X], [y]

        for _ in range(factor):
            X_noise = X + np.random.normal(0, noise_std, X.shape)

            shift = np.random.randint(-max_shift, max_shift + 1)
            if shift > 0:          # shift right → pad on the left
                X_shift = np.pad(X_noise[:, :-shift],
                                ((0, 0), (shift, 0)),
                                mode='edge')
            elif shift < 0:        # shift left → pad on the right
                X_shift = np.pad(X_noise[:, -shift:],
                                ((0, 0), (0, -shift)),
                                mode='edge')
            else:
                X_shift = X_noise

            X_aug.append(X_shift)
            y_aug.append(y)

        return np.vstack(X_aug), np.hstack(y_aug)
    def PreProcess(self, threshold=None, augment=False, aug_factor=2,
                   test_ratio=0.15):
        
        with open("Model_Data.json", 'r') as f:
            Data = json.load(f)

        # Ensure we have enough samples
        if len(Data) < self.Sequence_Length + 1:
            raise ValueError("Not enough data points for the requested sequence length.")

        # Build the future window used for live prediction
        Predictor = [row[:-1] for row in Data[-self.Sequence_Length:]]

        # Shuffle before splitting
        random.shuffle(Data)

        split_idx = max(1, int(len(Data) * (1 - test_ratio)))
        Train_Data = Data[:split_idx]
        Test_Data  = Data[split_idx:]
        #random.shuffle(Train_Data)
        #random.shuffle(Test_Data)
        #Train_Data,Test_Data = train_test_split(Data,train_size=0.9,random_state=42)
        # Build sequences
        if threshold is None:
            X_train, y_train = self.create_sequences(Train_Data)
            X_test,  y_test  = self.create_sequences(Test_Data)
        else:
            X_train, y_train = self.create_sequences_with_Thresh(Train_Data, threshold)
            X_test,  y_test  = self.create_sequences_with_Thresh(Test_Data, threshold)

        # Optional augmentation
        if augment and len(X_train) > 0:
            X_train, y_train = self.augment_sequences(X_train, y_train,
                                                      factor=aug_factor)

        # Print label distributions
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print("y_train distribution:", dict(zip(unique_train, counts_train)))

        unique_test, counts_test = np.unique(y_test, return_counts=True)
        print("y_test  distribution:", dict(zip(unique_test, counts_test)))

        return X_train, X_test, y_train, y_test, Predictor