import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna

# Define the directory for results
base_output_dir = r"/raid4/rprasai/merra2_files/station_with_more_sample_records/model_results"
file_output_dir = r"/raid4/rprasai/merra2_files/station_with_more_sample_records/model_results"

# Define columns and data directory
columns = ['TOTEXTTAU', 'BCEXTTAU', 'BCSCATAU', 'DUEXTTAU', 'DUSCATAU', 
           'OCEXTTAU', 'OCSCATAU', 'SSEXTTAU', 'SUEXTTAU', 'TOTANGSTR', 
           'SSSCATAU']
data_dir = r"/raid4/rprasai/merra2_files/station_with_more_sample_records/sample_records"

# Custom Dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the DNN model
class DNN(nn.Module):
    def __init__(self, droprate=0.2, input_size=11, n_first_layer=64, 
                 n_second_layer=32, n_third_layer=16, activation_function="relu"):
        super(DNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, n_first_layer),
            nn.BatchNorm1d(n_first_layer),
            self.get_activation(activation_function),
            nn.Dropout(droprate)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(n_first_layer, n_second_layer),
            nn.BatchNorm1d(n_second_layer),
            self.get_activation(activation_function),
            nn.Dropout(droprate)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(n_second_layer, n_third_layer),
            nn.BatchNorm1d(n_third_layer),
            self.get_activation(activation_function),
            nn.Dropout(droprate)
        )
        self.layer4 = nn.Linear(n_third_layer, 1)

    def get_activation(self, activation_function):
        if activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "tanh":
            return nn.Tanh()
        elif activation_function == "sigmoid":
            return nn.Sigmoid()
        else:
            raise ValueError("Invalid activation function.")

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        return y

# Define the objective function for Optuna
def objective(trial, file_path):
    activation_function = trial.suggest_categorical("activation_function", ["relu", "tanh", "sigmoid"])
    num_epochs = trial.suggest_int("num_epochs", 10, 100)
    droprate = trial.suggest_float("droprate", 0.3, 0.6)

    # Load and process data
    data = pd.read_csv(file_path)
    data = data.replace(-999, np.nan).dropna()
    X = data[columns].values
    y = data['AOD550_AVG'].values

    # Check if there are enough samples
    if len(X) < 2:
        print(f"Warning: Not enough data in file {file_path} for training.")
        return float('inf')  # Return high loss to indicate failure

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    batch_size = min(32, len(train_dataset))  # Dynamic batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = DNN(droprate=droprate, activation_function=activation_function)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_loss = float('inf')
    patience = 10  # Number of epochs to wait before stopping
    trigger_times = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Calculate validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}")

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            trigger_times = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print("Early stopping triggered")
                break

    return best_loss  # Return the best loss for Optuna optimization

# Process each file in the directory
for file_name in os.listdir(data_dir):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_dir, file_name)

        # Create a directory for results based on the file name
        file_output_dir = os.path.join(base_output_dir, os.path.splitext(file_name)[0])
        os.makedirs(file_output_dir, exist_ok=True)

        # Create study for hyperparameter optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, file_path), n_trials=50)

        # Best parameters
        best_params = study.best_params
        print("Best hyperparameters: ", best_params)

        # Load and process data again for final training
        data = pd.read_csv(file_path)
        data = data.replace(-999, np.nan).dropna()
        X = data[columns].values
        y = data['AOD550_AVG'].values

        # Check if there are enough samples
        if len(X) < 2:
            print(f"Warning: Not enough data in file {file_name} for training.")
            continue  # Skip to the next file

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Convert data to tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

        # Create DataLoader with a dynamic batch size
        batch_size = min(32, len(X_train))
        train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Instantiate and train the model with best parameters
        model = DNN(droprate=best_params['droprate'], activation_function=best_params['activation_function'])
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses = []
        val_losses = []
        best_loss = float('inf')
        patience = 10
        trigger_times = 0

        for epoch in range(best_params['num_epochs']):
            model.train()
            epoch_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Calculate validation loss
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test_tensor)
                val_loss = criterion(val_outputs, y_test_tensor)
                val_losses.append(val_loss.item())

            print(f"Epoch {epoch+1}/{best_params['num_epochs']}, Loss: {avg_loss:.4f}, Val Loss: {val_loss.item():.4f}")

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(file_output_dir, 'best_model_final.pth'))
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print("Early stopping triggered")
                    break

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(file_output_dir, 'training_validation_loss.png'))
        plt.close()

        # Evaluation on the test set
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test_tensor).numpy().flatten()

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)

        # Print metrics
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")

        # Plot predictions vs true values
        plt.figure(figsize=(8, 6))
        plt.hist2d(y_test, y_pred, bins=150, cmap='viridis', cmin=1)
        plt.colorbar()
        plt.xlim(0, 2)
        plt.ylim(0, 2)
        plt.xlabel('AOD True Values')
        plt.ylabel('AOD Predictions')

        # Best fit line
        m, b = np.polyfit(y_test, y_pred, 1)
        plt.plot(y_test, m * y_test + b, color='red', label='Best Fit Line')

        # 1:1 line
        plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='green', linestyle='--', label='1:1 Line')

        # Calculate and display metrics on the plot
        r2 = r2_score(y_test, y_pred)
        N = len(y_test)

        textstr = '\n'.join((f"MSE: {mse:.4f}",
                             f"RMSE: {rmse:.4f}",
                             f"MAE: {mae:.4f}",
                             f"R^2: {r2:.4f}",
                             r'N = %d' % (N)))

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.legend(loc='upper left')
        plt.title(f'DNN Predictions vs True Values - {file_name}')
        plt.savefig(os.path.join(file_output_dir, 'predictions_vs_true_values.png'))
        plt.close()

        # Save predicted values
        predicted_df = pd.DataFrame({'True Values': y_test, 'Predictions': y_pred})
        predicted_df.to_csv(os.path.join(file_output_dir, 'predicted_values.csv'), index=False)
