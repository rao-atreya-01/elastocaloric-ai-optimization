import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool
from torch.nn import Linear
import numpy as np
import gmsh
import dolfinx.io
from mpi4py import MPI
import warnings

# Suppress annoying warnings for a clean terminal
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

# 2. WAKE UP THE AI
print("Waking up the Artificial Intelligence...")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MetamaterialAI().to(device)

# Load the memories you trained earlier
model.load_state_dict(torch.load("elastocaloric_brain.pth", map_location=device, weights_only=True))
model.eval() # Put it in 'guessing' mode

# 3. GENERATE A BRAND NEW SHAPE 
width = 1.0
radius = 0.38 # Forcing a very thin, fragile wall to test the AI!
print(f"Drafting new CAD geometry: Width={width}, Radius={radius}...")

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("AI_Inference_Cell")
box = gmsh.model.occ.addBox(0, 0, 0, width, width, width)
cylinder = gmsh.model.occ.addCylinder(width/2, width/2, 0, 0, 0, width, radius)
cut_out = gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
gmsh.model.occ.synchronize()
if cut_out[0]: gmsh.model.addPhysicalGroup(3, [tag[1] for tag in cut_out[0]], 1)
gmsh.model.mesh.generate(3)

# Extract nodes and edges for the Neural Network
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
edge_index = np.array(edges, dtype=np.int64).T

# 4. ASK THE AI FOR INSTANT PHYSICS
print("Translating shape into Neural Network Tensors...")
x_tensor = torch.tensor(node_coords, dtype=torch.float32).to(device)
edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
batch_tensor = torch.zeros(x_tensor.size(0), dtype=torch.long).to(device) 

print("Asking the Brain to predict internal stress...")
with torch.no_grad():
    predicted_stress = model(x_tensor, edge_index_tensor, batch_tensor)

print("\n==================================================")
print(f"🧠 AI PREDICTION: {predicted_stress.item():.2f} MPa")
print("⚡ Time taken: A fraction of a second.")
print("==================================================\n")