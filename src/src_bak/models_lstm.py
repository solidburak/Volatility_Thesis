# src/models_lstm.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Dataset Class (Sliding Window Generation)
# ---------------------------------------------------------
class SequenceDataset(Dataset):
    def __init__(self, features, target, sequence_length):
        """
        features: numpy array shape (Time, Features)
        target: numpy array shape (Time,)
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(target, dtype=torch.long)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.features) - self.sequence_length

    def __getitem__(self, index):
        # Grab a window of past 'sequence_length' days
        x = self.features[index : index + self.sequence_length]
        # Predict the target for the day immediately AFTER the window
        y = self.target[index + self.sequence_length]
        return x, y

# ---------------------------------------------------------
# 2. LSTM Architecture
# ---------------------------------------------------------
class RegimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes=3, dropout=0.3):
        super(RegimeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The LSTM Core
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(16, num_classes)
        
    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the *last* time step only
        out = out[:, -1, :] 
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out) # Returns shape: (batch_size, 3)
        
        return out

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
def train_and_evaluate_lstm(df, feature_cols, target_col='Target_Smooth_10d', seq_length=21, epochs=30, lr=0.001, test_size=0.2):
    """
    Handles scaling, splitting, weighting, and training the PyTorch LSTM.
    """
    print(f"Preparing Data for LSTM (Sequence Length: {seq_length} days)...")
    
    # 1. Isolate Data
    # Drop rows with NaNs to ensure contiguous sequences
    df = df.dropna(subset=feature_cols + [target_col])
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # 2. Chronological Split
    split_idx = int(len(df) * (1 - test_size))
    
    X_train_raw, X_test_raw = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 3. Scaling (Critical for Neural Networks)
    # Fit scaler ONLY on training data to prevent leakage
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    
    # 4. Create DataLoaders
    train_dataset = SequenceDataset(X_train, y_train, seq_length)
    test_dataset = SequenceDataset(X_test, y_test, seq_length)
    
    # shuffle=True is generally safe here because the sequences themselves preserve time
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 5. Handle Class Imbalance with PyTorch Weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    print("\nCalculated Class Weights (Calm, Elevated, Shock):")
    print(np.round(class_weights, 4))
    
    # 6. Initialize Model
    model = RegimeLSTM(
        input_size=len(feature_cols), 
        hidden_size=32, 
        num_layers=2
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Added weight decay for regularization
    
    # 7. Training Phase
    print("\nStarting Training...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {total_loss/len(train_loader):.4f}")
            
    # 8. Evaluation Phase
    print("\n" + "="*50)
    print("LSTM Evaluation on Unseen Test Data")
    print("="*50)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    print(classification_report(
        all_targets, 
        all_preds, 
        target_names=['Class 0 (Calm)', 'Class 1 (Elevated)', 'Class 2 (Shock)'],
        zero_division=0
    ))
    
    return model, scaler
