"""Verbatim copy of the 2025 baseline dual-head BiLSTM (baseline/models.py, class LSTM).

Only the forward path is kept (training loop / plotting dropped); layer names and shapes are
identical so `baseline/model5.pth` loads with strict=True.
"""
import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size=20, seq_length=1000, hidden_size1=128,
                 hidden_size2=64, dropout=0.2,
                 num_phases=9, num_functions=3):
        super(LSTM, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_phases = num_phases
        self.num_functions = num_functions

        self.optimizer = None
        self.learning_rate = None

        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1,
                             batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1 * 2, hidden_size=hidden_size2,
                             batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size2 * 2, 64)
        self.relu = nn.ReLU()
        self.phase_output = nn.Linear(64, self.num_phases)
        self.function_output = nn.Linear(64, self.num_functions)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)
        last_output = lstm2_out[:, -1, :]
        shared_features = self.relu(self.dense(last_output))
        shared_features = self.dropout(shared_features)
        phase_logits = self.phase_output(shared_features)
        function_logits = self.function_output(shared_features)
        return phase_logits, function_logits
