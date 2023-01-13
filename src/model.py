import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style = "darkgrid")

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_outputs):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.num_outputs = num_outputs
        
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.fc3 = nn.Linear(hidden_size_2, num_outputs)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def fit(self, X, y, X_val, y_val, optimizer, loss_fn, nb_epochs=10, batch_size=32, print_tracking = True):
        # Convert inputs to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        
        tracking_loss_train = list(); tracking_loss_val = list()
        
        data = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        
        for epoch in range(nb_epochs):
            # Initialize variables for storing the losses
            train_loss = 0
            num_samples = 0
            
            print(f"".center(100, "="))
            # Loop over the dataloader to get the mini-batches
            for inputs, targets in dataloader:
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward prop
                outputs = self.forward(inputs)
                
                # Calcule de la loss
                loss = loss_fn(outputs, targets)
                
                # Retroprop
                loss.backward()
                
                # Actualisation des paramètres
                optimizer.step()
                
                # Accumulate the loss and number of samples
                train_loss += loss.item() * inputs.size(0)
                num_samples += inputs.size(0)
                
            # Compute the average loss on the training set
            train_loss = train_loss / num_samples

            # Evaluate the model on the validation set
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = loss_fn(val_outputs, y_val).item()

            # Print the average loss on the training set and the validation set
            if print_tracking:
                tracking_loss_train.append(train_loss)
                tracking_loss_val.append(val_loss)
                print(f"Epoch {epoch+1}: Train loss = {train_loss:.4f} / Validation loss = {val_loss:.4f}")
        
        if print_tracking:
            plt.figure(figsize = (15, 5))
            plt.title("Tracking de la loss en fonction du nombre d'epochs", fontsize = 13, fontweight = "bold")
            plt.plot(tracking_loss_train, label = 'Train set loss', marker = "o", color = "C0")
            plt.plot(tracking_loss_val, label = 'Validation set loss', marker = "o", color = "C3")
            plt.ylabel("Huber Loss")
            plt.xlabel("Nombre d'epochs")
            plt.legend()
            plt.show()
        
        return self
    
    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.forward(X).detach().numpy()