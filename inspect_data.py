import torch
import os

dataset_dir = "proprietary_dataset"
files = [f for f in os.listdir(dataset_dir) if f.endswith('.pt')]

if not files:
    print("No data found!")
else:
    # Load the first graph
    file_path = os.path.join(dataset_dir, files[0])
    graph_data = torch.load(file_path)
    
    print(f"--- LOADING: {files[0]} ---")
    print("\n1. THE FULL GRAPH OBJECT:")
    print(graph_data)
    
    print("\n2. THE EXACT TENSORS PACKED INSIDE:")
    for key in graph_data.keys():
        # Check if the item is a tensor so we can print its shape
        if hasattr(graph_data[key], 'shape'):
            print(f" - {key}: Tensor with shape {graph_data[key].shape}")
        else:
            print(f" - {key}: {type(graph_data[key])} = {graph_data[key]}")