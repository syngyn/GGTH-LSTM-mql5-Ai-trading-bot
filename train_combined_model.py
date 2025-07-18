import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import sys
import traceback

from data_processing import load_and_align_data, create_features

class CombinedLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, num_regression_outputs):
        super(CombinedLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc_regression = nn.Linear(hidden_size, num_regression_outputs)
        self.fc_classification = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden_state = out[:, -1, :]
        regression_output = self.fc_regression(last_hidden_state)
        classification_logits = self.fc_classification(last_hidden_state)
        return regression_output, classification_logits

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "models")
DATA_DIR = SCRIPT_DIR
REQUIRED_FILES = {
    "EURUSD": "EURUSD60.csv", "EURJPY": "EURJPY60.csv", "USDJPY": "USDJPY60.csv",
    "GBPUSD": "GBPUSD60.csv", "EURGBP": "EURGBP60.csv", "USDCAD": "USDCAD60.csv",
    "USDCHF": "USDCHF60.csv"
}

INPUT_FEATURES = 15
HIDDEN_SIZE, NUM_LAYERS, SEQ_LEN = 128, 2, 20
OUTPUT_STEPS = 5 # <--- CORE CHANGE
NUM_CLASSES = 3
EPOCHS, BATCH_SIZE, LEARNING_RATE = 30, 64, 0.001
LOOKAHEAD_BARS, PROFIT_THRESHOLD_ATR = 5, 0.75

if __name__ == "__main__":
    main_df, feature_names = create_features(load_and_align_data(REQUIRED_FILES, DATA_DIR))

    if len(main_df) < SEQ_LEN + OUTPUT_STEPS:
        print(f"FATAL ERROR: Not enough data for sequence and lookahead."); sys.exit(1)
    
    # --- Create Both Target Types ---
    print("Creating regression and classification targets...")
    # Regression targets
    regr_targets = []
    for i in range(1, OUTPUT_STEPS + 1):
        regr_targets.append(main_df['EURUSD_close'].shift(-i))
    regr_target_df = pd.concat(regr_targets, axis=1)
    regr_target_df.columns = [f'target_regr_{i}' for i in range(OUTPUT_STEPS)]
    
    # Classification targets
    future_price = main_df['EURUSD_close'].shift(-LOOKAHEAD_BARS)
    atr_threshold = main_df['eurusd_atr'] * PROFIT_THRESHOLD_ATR
    conditions = [future_price > main_df['EURUSD_close'] + atr_threshold, future_price < main_df['EURUSD_close'] - atr_threshold]
    choices = [2, 0] # 2=Buy, 0=Sell
    class_target_s = pd.Series(np.select(conditions, choices, default=1), index=main_df.index, name='target_class')
    
    # Combine and clean
    main_df = pd.concat([main_df, regr_target_df, class_target_s], axis=1)
    main_df.dropna(inplace=True)

    X = main_df[feature_names].values
    y_regr = main_df[[f'target_regr_{i}' for i in range(OUTPUT_STEPS)]].values
    y_class = main_df['target_class'].values
    
    # --- Scaling ---
    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)
    
    target_scaler = StandardScaler()
    y_regr_scaled = target_scaler.fit_transform(y_regr)
    
    # --- Build Sequences ---
    print("Building sequences...")
    X_seq, y_regr_seq, y_class_seq = [], [], []
    for i in range(len(X_scaled) - SEQ_LEN):
        X_seq.append(X_scaled[i:i + SEQ_LEN])
        y_regr_seq.append(y_regr_scaled[i + SEQ_LEN - 1])
        y_class_seq.append(y_class[i + SEQ_LEN - 1])
        
    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    y_regr_tensor = torch.tensor(np.array(y_regr_seq), dtype=torch.float32)
    y_class_tensor = torch.tensor(np.array(y_class_seq), dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_regr_tensor, y_class_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Created {len(X_tensor)} sequences.")
    
    # --- Model, Loss, and Optimizer ---
    model = CombinedLSTM(input_size=INPUT_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, num_regression_outputs=OUTPUT_STEPS)
    regr_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n--- Starting Combined Model Training ---")
    model.train()
    for epoch in range(EPOCHS):
        total_loss, total_regr_loss, total_class_loss = 0, 0, 0
        for i, (xb, yb_regr, yb_class) in enumerate(loader):
            optimizer.zero_grad()
            pred_regr, pred_class_logits = model(xb)
            
            loss_regr = regr_loss_fn(pred_regr, yb_regr)
            loss_class = class_loss_fn(pred_class_logits, yb_class)
            
            combined_loss = loss_regr + loss_class
            
            combined_loss.backward()
            optimizer.step()
            
            total_loss += combined_loss.item()
            total_regr_loss += loss_regr.item()
            total_class_loss += loss_class.item()
            
        avg_loss = total_loss / len(loader)
        avg_regr_loss = total_regr_loss / len(loader)
        avg_class_loss = total_class_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Loss: {avg_loss:.4f} (Regr: {avg_regr_loss:.4f}, Class: {avg_class_loss:.4f})")
        
    print("\n--- Training Complete ---")
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    MODEL_FILE = os.path.join(MODEL_SAVE_PATH, "lstm_model_regression.pth") 
    SCALER_FILE_TARGET = os.path.join(MODEL_SAVE_PATH, "scaler_regression.pkl")
    SCALER_FILE_FEATURE = os.path.join(MODEL_SAVE_PATH, "scaler.pkl")
    
    torch.save({"model_state": model.state_dict()}, MODEL_FILE)
    print(f"(+) Combined model saved to {MODEL_FILE}")
    joblib.dump(target_scaler, SCALER_FILE_TARGET)
    print(f"(+) Regression TARGET scaler saved to {SCALER_FILE_TARGET}")
    joblib.dump(feature_scaler, SCALER_FILE_FEATURE)
    print(f"(+) FEATURE scaler saved to {SCALER_FILE_FEATURE}")