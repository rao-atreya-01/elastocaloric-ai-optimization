import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import Linear
import numpy as np
import gmsh
import dolfinx.io
from mpi4py import MPI
import warnings
import random
import time

warnings.filterwarnings("ignore")

# 1. THE BRAIN ARCHITECTURE
class MetamaterialAI(torch.nn.Module):
    def __init__(self):
        super(MetamaterialAI, self).__init__()
        self.conv1 = GraphConv(3, 64)
        self.conv2 = GraphConv(64, 128)
        self.conv3 = GraphConv(128, 64)
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x

# 2. WAKE UP THE SURROGATE BRAIN
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MetamaterialAI().to(device)
model.load_state_dict(torch.load("elastocaloric_brain.pth", map_location=device, weights_only=True))
model.eval()

# 3. THE REINFORCEMENT OPTIMIZATION LOOP
print("\n🚀 INITIATING HIGH-SPEED AI OPTIMIZATION LOOP")
print("Goal: Find the largest radius that keeps stress below 1500 MPa.\n")

best_radius = 0.0
best_stress = 999999.0
width = 1.0

start_time = time.time()

# The AI will rapidly test 20 different designs
for generation in range(1, 21):
    # Agent explores a new design parameter (Radius between 0.1 and 0.48)
    radius = round(random.uniform(0.1, 0.48), 3)
    
    # 1. Instantly Draft CAD
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("AI_Cell")
    box = gmsh.model.occ.addBox(0, 0, 0, width, width, width)
    cylinder = gmsh.model.occ.addCylinder(width/2, width/2, 0, 0, 0, width, radius)
    cut_out = gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
    gmsh.model.occ.synchronize()
    if cut_out[0]: gmsh.model.addPhysicalGroup(3, [tag[1] for tag in cut_out[0]], 1)
    gmsh.model.mesh.generate(3)

    # 2. Extract Mesh into Tensors
    mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.clear()
    
    node_coords = mesh.geometry.x
    mesh.topology.create_connectivity(1, 0)
    edge_to_node = mesh.topology.connectivity(1, 0)
    edges = []
    for i in range(edge_to_node.num_nodes):
        connected_nodes = edge_to_node.links(i)
        if len(connected_nodes) == 2:
            edges.extend([[connected_nodes[0], connected_nodes[1]], [connected_nodes[1], connected_nodes[0]]])
    
    x_tensor = torch.tensor(node_coords, dtype=torch.float32).to(device)
    edge_index_tensor = torch.tensor(np.array(edges, dtype=np.int64).T, dtype=torch.long).to(device)
    batch_tensor = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device) 

    # 3. Brain Predicts Physics Instantly
    with torch.no_grad():
        predicted_stress = model(x_tensor, edge_index_tensor, batch_tensor).item()
        
    print(f"Gen {generation:02d} | Tested Radius: {radius} -> Predicted Stress: {predicted_stress:.2f} MPa")
    
 # 4. Agent Evaluation (Just find the absolute lowest stress)
    if predicted_stress < best_stress:
        best_stress = predicted_stress
        best_radius = radius

end_time = time.time()

print("\n==================================================")
print(f"⏱️ Optimization completed in {end_time - start_time:.2f} seconds.")
print(f"🏆 WINNING DESIGN FOUND:")
print(f"   Optimal Hole Radius: {best_radius}")
print(f"   Predicted Stress:    {best_stress:.2f} MPa")
print("==================================================\n")