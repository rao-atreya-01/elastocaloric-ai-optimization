import os
import torch
import concurrent.futures
from torch_geometric.data import Data

# Import the custom environment we just built
from elastocaloric_env import ElastocaloricEnv 

SAVE_DIR = "./proprietary_dataset"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_single_sample(sample_id):
    """Generates ONE valid lattice graph."""
    env = ElastocaloricEnv()
    valid = False
    file_path = ""
    
    while not valid:
        # 1. Guess a geometry
        action = env.action_space.sample()
        
        # 2. Simulate it
        state, reward, done, truncated, info = env.step(action)
        
        # 3. Check constraints
        if info.get("printable") == True and info.get("fea_failed") == False:
            valid = True
            
            # 4. Pull the raw mesh arrays
            nodes = info["node_coordinates"]
            edges = info["edge_connectivity"]
            max_stress = info["max_von_mises"]
            
            # 5. Convert to PyTorch Tensors
            x = torch.tensor(nodes, dtype=torch.float)
            edge_index = torch.tensor(edges, dtype=torch.long)
            y = torch.tensor([max_stress], dtype=torch.float)
            
            # 6. Package and Save
            graph_data = Data(x=x, edge_index=edge_index, y=y)
            file_path = os.path.join(SAVE_DIR, f"lattice_{sample_id}.pt")
            torch.save(graph_data, file_path)
            
    env.close()
    return file_path

def build_dataset_parallel(total_samples=10, num_cores=4):
    print(f"Starting parallel generation of {total_samples} lattices on {num_cores} cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        sample_ids = range(total_samples)
        for i, _ in enumerate(executor.map(generate_single_sample, sample_ids)):
            print(f"--> Progress: {i + 1} / {total_samples} graphs saved to disk.")
    print("Dataset generation complete! Your IP moat is built.")

if __name__ == "__main__":
    build_dataset_parallel(total_samples=10, num_cores=4)
