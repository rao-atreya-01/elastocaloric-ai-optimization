import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import Linear

# 1. LOAD THE PROPRIETARY DATASET
print("Loading Proprietary Dataset...")
dataset_dir = "proprietary_dataset"
files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]

dataset = []
for f in files:
    data = torch.load(os.path.join(dataset_dir, f))
    dataset.append(data)

print(f"Successfully loaded {len(dataset)} 3D geometries.")

# Split into Training Data (80%) and Test Data (20%)
train_size = int(0.8 * len(dataset))
train_dataset = dataset[:train_size]
test_dataset = dataset[train_size:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. DESIGN THE NEURAL NETWORK ARCHITECTURE
class MetamaterialAI(torch.nn.Module):
    def __init__(self):
        super(MetamaterialAI, self).__init__()
        # Graph Convolutions: Learning how nodes connect to each other
        self.conv1 = GraphConv(3, 64)  # Input: 3 (X,Y,Z coords)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 64)
        
        # Linear Layers: Translating the 3D graph into a single Stress number
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 1)    # Output: 1 (Max Stress)

    def forward(self, x, edge_index, batch):
        # Pass data through the graph layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        
        # Pool the entire 3D shape into one "thought" vector
        x = global_mean_pool(x, batch)
        
        # Predict the Stress
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# 3. INITIALIZE THE BRAIN
# This automatically routes math to your Mac's Apple Silicon GPU (MPS)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Training on: {device.type.upper()}")

model = MetamaterialAI().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = torch.nn.MSELoss() # Mean Squared Error

# 4. START THE TRAINING LOOP
epochs = 50
print("\n--- INITIATING NEURAL NETWORK TRAINING ---")

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    
    # Train the AI
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # The AI guesses the stress
        predictions = model(batch.x, batch.edge_index, batch.batch)
        
        # We punish it based on how far off it is from the real FEniCSx math
        loss = loss_fn(predictions, batch.y.view(-1, 1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Test the AI on shapes it has never seen before
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            predictions = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(predictions, batch.y.view(-1, 1))
            test_loss += loss.item()
            
    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:03d} | Training Error: {total_loss/len(train_loader):.2f} | Testing Error: {test_loss/len(test_loader):.2f}")

print("\nTraining Complete! Saving the Brain...")
torch.save(model.state_dict(), "elastocaloric_brain.pth")
print("Saved as 'elastocaloric_brain.pth'. The AI is ready to design.")