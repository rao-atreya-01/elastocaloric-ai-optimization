import traceback
import numpy as np
from elastocaloric_env import ElastocaloricEnv

def run_debug():
    print("Initializing Environment...")
    env = ElastocaloricEnv()
    
    print("\n--- FORCED VALID GEOMETRY TEST ---")
    # We are bypassing the AI's random guesses and forcing a safe shape:
    # Width = 1.0mm, Radius = 0.25mm, other params = 1.0mm
    action = np.array([1.0, 0.25, 1.0, 1.0, 1.0], dtype=np.float32)
    print(f"Hardcoded Parameters: {action}")
    
    try:
        print("Sending to Gmsh and FEniCSx. Waiting for physics to solve...")
        state, reward, done, truncated, info = env.step(action)
        
        print("\n--- RESULTS ---")
        print("Printable:", info.get('printable'))
        print("FEA Failed:", info.get('fea_failed'))
        print("Max Stress:", info.get('max_von_mises'))
        print("Nodes Extracted:", len(info.get('node_coordinates')))
        
        if info.get('printable') == True and info.get('fea_failed') == False:
            print("\n✅ SUCCESS! The C++ Physics Engine is alive and mathematically solving 3D meshes!")
            
    except Exception as e:
        print("\n🚨 CRITICAL CRASH TRACEBACK:")
        traceback.print_exc()

if __name__ == "__main__":
    run_debug()