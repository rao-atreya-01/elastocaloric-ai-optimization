import gymnasium as gym
from gymnasium import spaces
import numpy as np
import dolfinx
import gmsh
import dolfinx.io
from mpi4py import MPI
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem # <-- THE FIX: Importing the heavy solver
from dolfinx import default_scalar_type
import traceback

# 1. THE GRAPH TRANSLATOR
def extract_graph_from_mesh(mesh):
    node_coords = mesh.geometry.x
    mesh.topology.create_connectivity(1, 0)
    edge_to_node = mesh.topology.connectivity(1, 0)
    edges = []
    for i in range(edge_to_node.num_nodes):
        connected_nodes = edge_to_node.links(i)
        if len(connected_nodes) == 2:
            n1, n2 = connected_nodes[0], connected_nodes[1]
            edges.extend([[n1, n2], [n2, n1]]) 
    if len(edges) == 0:
        return node_coords, np.zeros((2, 0), dtype=np.int64)
    edge_index = np.array(edges, dtype=np.int64).T
    return node_coords, edge_index

# 2. THE MAIN ENVIRONMENT
class ElastocaloricEnv(gym.Env):
    """Custom Environment bridging OpenAI Gym and FEniCSx"""
    
    def __init__(self):
        super(ElastocaloricEnv, self).__init__()
        self.action_space = spaces.Box(low=0.1, high=2.0, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.1, high=2.0, shape=(5,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        initial_state = self.action_space.sample()
        return initial_state, {}

    def step(self, action):
        width = float(action[0]) 
        radius = float(action[1])

        # --- DROP ZONE 1: MANUFACTURING CONSTRAINTS ---
        if np.any(action < 0.2):
            return action, -1000.0, True, False, {"printable": False, "fea_failed": False}
            
        if radius >= (width / 2.0) - 0.1:
            return action, -1000.0, True, False, {"printable": False, "fea_failed": False}

        try:
            # --- 1. GENERATE GEOMETRY (GMSH) ---
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0) 
            gmsh.model.add("Elastocaloric_Unit_Cell")
            
            box = gmsh.model.occ.addBox(0, 0, 0, width, width, width)
            cylinder = gmsh.model.occ.addCylinder(width/2, width/2, 0, 0, 0, width, radius)
            
            cut_out = gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
            gmsh.model.occ.synchronize()
            
            volume_tags = [tag[1] for tag in cut_out[0]]
            if volume_tags: gmsh.model.addPhysicalGroup(3, volume_tags, 1)
                
            gmsh.model.mesh.generate(3)
            mesh, _, _ = dolfinx.io.gmshio.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
            gmsh.clear()
            
            # --- 2. THE SQUISH: SOLVE PHYSICS (FENICSX) ---
            V = fem.functionspace(mesh, ("Lagrange", 1, (3,)))
            u = ufl.TrialFunction(V)
            v = ufl.TestFunction(V)

            # Nitinol Material Properties (Austenite phase)
            E_Niti = 75e9  # 75 GPa (Young's Modulus)
            nu = 0.33      # Poisson's ratio
            mu = E_Niti / (2.0 * (1.0 + nu))
            lmbda = E_Niti * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

            # Define Strain and Stress
            def epsilon(u): return ufl.sym(ufl.grad(u))
            def sigma(u): return lmbda * ufl.tr(epsilon(u)) * ufl.Identity(3) + 2.0 * mu * epsilon(u)

            # Define the Boundaries
            def bottom_boundary(x): return np.isclose(x[2], 0.0)
            def top_boundary(x): return np.isclose(x[2], width)
            
            # Fix Bottom to the floor (0 movement)
            bottom_dofs = fem.locate_dofs_geometrical(V, bottom_boundary)
            u_bottom = fem.Constant(mesh, default_scalar_type((0.0, 0.0, 0.0)))
            bc_bottom = fem.dirichletbc(u_bottom, bottom_dofs, V)

            # Crush Top down by 5% of its height
            top_dofs = fem.locate_dofs_geometrical(V, top_boundary)
            u_top = fem.Constant(mesh, default_scalar_type((0.0, 0.0, -0.05 * width)))
            bc_top = fem.dirichletbc(u_top, top_dofs, V)

            # THE FIX: Directly call LinearProblem
            a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
            L = ufl.dot(fem.Constant(mesh, default_scalar_type((0.0, 0.0, 0.0))), v) * ufl.dx
            problem = LinearProblem(a, L, bcs=[bc_bottom, bc_top], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
            uh = problem.solve()

            # Calculate Max Von Mises Stress across the shape
            s = sigma(uh) - (1./3)*ufl.tr(sigma(uh))*ufl.Identity(3)
            von_Mises = ufl.sqrt(3./2 * ufl.inner(s, s))
            
            V_von_mises = fem.functionspace(mesh, ("DG", 0))
            expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
            von_mises_func = fem.Function(V_von_mises)
            von_mises_func.interpolate(expr)
            
            # Convert Pascals to Megapascals (MPa)
            max_von_mises = float(np.max(von_mises_func.x.array)) / 1e6 
            
            # --- 3. RETURN DATA TO AI ---
            nodes, edges = extract_graph_from_mesh(mesh)
            
            info = {
                "printable": True,
                "fea_failed": False,
                "max_von_mises": max_von_mises,
                "node_coordinates": nodes,
                "edge_connectivity": edges
            }
            
            # The AI wants to minimize stress, so reward is negative max stress
            return action, -max_von_mises, True, False, info

        except Exception as e:
            print(f"\n[PHYSICS ENGINE CRASHED]: {e}")
            try: gmsh.clear() 
            except: pass
            return action, -1000.0, True, False, {"printable": True, "fea_failed": True}