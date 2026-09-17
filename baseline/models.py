import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size=20, seq_length=1000, hidden_size1=128, 
                 hidden_size2=64, dropout=0.2,
                 num_phases=9, num_functions=3):
        super(LSTM, self).__init__()

        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_phases = num_phases
        self.num_functions = num_functions

        # Add optimizer as class attribute
        self.optimizer = None
        self.learning_rate = None

        # First Bidirectional LSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size1,
            batch_first=True,
            bidirectional=True
        )

        # Second Bidirectional LSTM layer
        self.lstm2 = nn.LSTM(
            input_size=hidden_size1 * 2,  # * 2 because bidirectional
            hidden_size=hidden_size2,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(dropout)

        # Shared dense layer
        self.dense = nn.Linear(hidden_size2 * 2, 64)  # * 2 because bidirectional
        self.relu = nn.ReLU()

        # Output layers
        self.phase_output = nn.Linear(64, self.num_phases)  # phases
        self.function_output = nn.Linear(64, self.num_functions)  # functions

        # Remove the softmax layer since CrossEntropyLoss includes it
        #self.softmax = nn.Softmax(dim=1)

        # Initialize loss tracking
        self.phase_losses = []
        self.function_losses = []
        self.total_losses = []
        self.val_phase_losses = []
        self.val_function_losses = []
        self.val_total_losses = []

        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_length, input_size)

        # Apply mask if provided
        if mask is not None:
            x = x * mask.unsqueeze(-1)

        # First LSTM layer
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)

        # Second LSTM layer
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.dropout(lstm2_out)

        # Take the last output
        last_output = lstm2_out[:, -1, :]

        # Shared dense layer
        shared_features = self.relu(self.dense(last_output))
        # Try a dropout layer here
        shared_features = self.dropout(shared_features)

        # Output branches
        phase_logits = self.phase_output(shared_features)
        function_logits = self.function_output(shared_features)

        # Apply softmax
        #phase_probs = self.softmax(phase_logits)
        #function_probs = self.softmax(function_logits)

        #return phase_probs, function_probs
        return phase_logits, function_logits

    def train_model(self, 
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: torch.utils.data.DataLoader,
                    num_epochs: int,
                    learning_rate: float = 0.001,
                    device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
        """
        Train the model with continuous epoch counting and optimizer state preservation
        """
        self.to(device)
        print(f'Training on {device}')

        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # Initialize optimizer only if it doesn't exist or learning rate changed
        if self.optimizer is None or self.learning_rate != learning_rate:
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
            self.learning_rate = learning_rate

        for epoch in range(num_epochs):
            epoch_phase_loss = 0.0
            epoch_function_loss = 0.0
            epoch_total_loss = 0.0
            num_batches = 0
            
            # Reload the data
            dataset.load()

            self.train()
            for X, y in train_loader:
                # Move data to device
                X, y = X.to(device), y.to(device)

                # Split target into phase and function
                #phase_target = y[:, :self.num_phases]  # Phase columns
                #function_target = y[:, self.num_phases:]  # Remaining are function columns
                # For CrossEntropyLoss, targets should be class indices
                # Assuming y is one-hot encoded, convert to class indices
                phase_target = torch.argmax(y[:, :self.num_phases], dim=1)
                function_target = torch.argmax(y[:, self.num_phases:], dim=1)

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                #phase_pred, function_pred = self(X)
                # Forward pass
                phase_logits, function_logits = self(X)

                # Calculate losses
                #phase_loss = self.criterion(phase_pred, phase_target)
                #function_loss = self.criterion(function_pred, function_target)
                phase_loss = self.criterion(phase_logits, phase_target) / self.num_phases
                function_loss = self.criterion(function_logits, function_target) / self.num_functions
                total_loss = phase_loss + function_loss

                # Backward pass and optimize
                total_loss.backward()
                self.optimizer.step()

                # Accumulate losses
                epoch_phase_loss += phase_loss.item()
                epoch_function_loss += function_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1

                # Print loss
                #if num_batches % 10 == 0:
                #    print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{num_batches}/{len(train_loader)}]')
                #    print(f'Phase Loss: {phase_loss.item():.4f}')
                #    print(f'Function Loss: {function_loss.item():.4f}')
                #    print(f'Total Loss: {total_loss.item():.4f}\n')

            # Calculate average losses for the epoch
            avg_phase_loss = epoch_phase_loss / num_batches
            avg_function_loss = epoch_function_loss / num_batches
            avg_total_loss = epoch_total_loss / num_batches

            # Store losses
            self.phase_losses.append(avg_phase_loss)
            self.function_losses.append(avg_function_loss)
            self.total_losses.append(avg_total_loss)

            # Validate the model
            self.eval()
            with torch.no_grad():
                # Get first batch from validation loader
                dataset_val.load()
                X_val, y_val = next(iter(val_loader))
                X_val, y_val = X_val.to(device), y_val.to(device)

                # Split validation target
                val_phase_target = y_val[:, :self.num_phases]
                val_function_target = y_val[:, self.num_phases:]

                # Forward pass
                val_phase_pred, val_function_pred = self(X_val)

                # Calculate validation losses
                val_phase_loss = self.criterion(val_phase_pred, val_phase_target) / self.num_phases
                val_function_loss = self.criterion(val_function_pred, val_function_target) / self.num_functions
                val_total_loss = val_phase_loss + val_function_loss

                # Store validation losses
                self.val_phase_losses.append(val_phase_loss.item())
                self.val_function_losses.append(val_function_loss.item())
                self.val_total_losses.append(val_total_loss.item())     

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train ~ Phase: {avg_phase_loss:.4f}, Function: {avg_function_loss:.4f}, Total: {avg_total_loss:.4f}')
            print(f'Val   ~ Phase: {val_phase_loss.item():.4f}, Function: {val_function_loss.item():.4f}, Total: {val_total_loss.item():.4f}')

    def plot_losses(self) -> None:
        """
        Plot the training losses
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.phase_losses) + 1)

        plt.plot(epochs, self.phase_losses, 'b-', label='Phase Loss')
        plt.plot(epochs, self.function_losses, 'r-', label='Function Loss')
        plt.plot(epochs, self.total_losses, 'g-', label='Total Loss')

        # Added validation losses
        plt.plot(epochs, self.val_phase_losses, 'b--', label='Val Phase Loss')
        plt.plot(epochs, self.val_function_losses, 'r--', label='Val Function Loss')
        plt.plot(epochs, self.val_total_losses, 'g--', label='Val Total Loss')

        plt.title('Training & Validation Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (Cross Entropy)')
        plt.legend()
        plt.grid(True)
        plt.show()
