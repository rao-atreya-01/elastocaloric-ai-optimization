import numpy as np
import gymnasium as gym
from gymnasium import spaces

class ElastocaloricEnv(gym.Env):
    """
    The Universal Translator between PyTorch (The Brain) and FEniCSx (The Physics).
    """
    def __init__(self):
        super(ElastocaloricEnv, self).__init__()
        
        self.cell_size = 10.0 # Our standard 10x10x10mm BCC unit cell
        self.min_printable_wall = 0.2 # LPBF 3D Printing Constraint (mm)
        
        # ACTION SPACE: What is the AI allowed to do?
        # It can guess a change in the cylinder radius between -0.5mm and +0.5mm per step.
        self.action_space = spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        
        # OBSERVATION SPACE (The State): What does the AI see?
        # [current_radius, max_stress, porosity_percentage]
        self.observation_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.0]), 
            high=np.array([5.0, 10000.0, 100.0]), 
            dtype=np.float32
        )
        
        self.current_radius = 2.0 # Start with a safe, conservative 2mm hole

    def reset(self, seed=None):
        """Wipes the lab bench clean for a new generation."""
        super().reset(seed=seed)
        self.current_radius = 2.0
        
        # Baseline starting stats
        initial_stress = 50.0 
        initial_porosity = 25.0
        
        self.state = np.array([self.current_radius, initial_stress, initial_porosity], dtype=np.float32)
        return self.state, {}

    def step(self, action):
        """The core loop: AI guesses -> Check Rules -> Run Physics -> Reward AI."""
        
        # 1. THE ACTION
        radius_change = action[0]
        new_radius = self.current_radius + radius_change
        
        # 2. THE MANUFACTURABILITY CHECK (0.2mm rule)
        wall_thickness = self.cell_size - (2.0 * new_radius)
        
        if wall_thickness < self.min_printable_wall or new_radius <= 0.1:
            # THE DEATH PENALTY: The AI tried to build an impossible shape.
            reward = -1000.0
            done = True # End the simulation
            print(f"❌ FATAL: AI guessed radius {new_radius:.2f}mm. Wall too thin to 3D print!")
            return self.state, reward, done, False, {"status": "unprintable"}

        # 3. IGNITE THE PHYSICS ENGINE!
        self.current_radius = new_radius
        print(f"⚙️ VALID SHAPE: Radius {new_radius:.2f}mm. Building CAD and crushing in FEniCSx...")
        
        # =====================================================================
        # IN REALITY, YOU CALL YOUR SCRIPTS HERE:
        # e.g., generate_perfect_lattice(self.current_radius)
        # max_stress, porosity = simulate_compression()
        # =====================================================================
        
        # For this blueprint, we simulate the physics response:
        # Bigger holes = more porosity (good) but exponentially more stress (bad)
        simulated_porosity = (new_radius / 5.0) * 100.0 
        simulated_stress = 10.0 * np.exp(new_radius / 1.5)
        
        self.state = np.array([self.current_radius, simulated_stress, simulated_porosity], dtype=np.float32)
        
        # 4. THE REWARD FUNCTION (The actual PhD breakthrough)
        # Reward = (Maximize Cooling) - (Minimize Fatigue)
        reward = (simulated_porosity * 5.0) - (simulated_stress * 0.5)
        
        # If stress exceeds Nitinol's breaking point (~500 MPa), snap the metal
        done = bool(simulated_stress > 500.0)
        if done:
            reward -= 500.0 # Massive penalty for breaking
            print(f"💥 SNAP! Metal fractured at {simulated_stress:.1f} MPa.")
        else:
            print(f"✅ SUCCESS: Survived with {simulated_porosity:.1f}% porosity. Reward: {reward:.1f}")

        return self.state, reward, done, False, {"status": "success"}

# Quick test to prove the environment works!
if __name__ == "__main__":
    env = ElastocaloricEnv()
    state, _ = env.reset()
    
    print("Testing the Environment Wrapper...\n")
    # Force the AI to aggressively increase the hole size by 0.5mm repeatedly
    for step in range(1, 10):
        print(f"--- Generation {step} ---")
        action = np.array([0.5], dtype=np.float32)
        next_state, reward, done, truncated, info = env.step(action)
        
        if done:
            print("\nSimulation Terminated by Environment Rules.")
            break
