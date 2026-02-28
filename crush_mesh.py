import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem  # THE FIX: Explicitly import the solver
from dolfinx.io import gmshio

def simulate_compression():
    print("1. Loading Metamaterial Mesh into FEniCSx...")
    domain, _, _ = gmshio.read_from_msh("perfect_physics_lattice.msh", MPI.COMM_WORLD, 0, gdim=3)
    
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

    print("2. Setting up the Virtual Hydraulic Press (Boundary Conditions)...")
    def bottom_surface(x): return np.isclose(x[2], 0.0)
    def top_surface(x): return np.isclose(x[2], 10.0)

    fdim = domain.topology.dim - 1
    bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom_surface)
    top_facets = mesh.locate_entities_boundary(domain, fdim, top_surface)

    # Clamp the bottom (0 movement)
    u_bottom = np.array([0.0, 0.0, 0.0], dtype=default_scalar_type)
    bc_bottom = fem.dirichletbc(u_bottom, fem.locate_dofs_topological(V, fdim, bottom_facets), V)

    # Push the top down by 0.5 units in the Z direction
    u_top = np.array([0.0, 0.0, -0.5], dtype=default_scalar_type)
    bc_top = fem.dirichletbc(u_top, fem.locate_dofs_topological(V, fdim, top_facets), V)
    
    bcs = [bc_bottom, bc_top]

    print("3. Defining Nitinol Material Properties...")
    E = default_scalar_type(50e3) 
    nu = default_scalar_type(0.3)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    def epsilon(u): return ufl.sym(ufl.grad(u))
    def sigma(u): return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

    a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
    L = ufl.dot(fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0))), v) * ufl.dx

    print("4. FIRING PHYSICS ENGINE! Solving for structural deformation...")
    # THE FIX: Call the newly imported LinearProblem directly
    problem = LinearProblem(a, L, bcs=bcs)
    uh = problem.solve()

    print("5. Exporting Results...")
    with io.XDMFFile(domain.comm, "deformation_results.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        uh.name = "Deformation"
        xdmf.write_function(uh)
        
    print("\nSUCCESS! Mesh crushed. Results saved as: deformation_results.xdmf")

if __name__ == "__main__":
    simulate_compression()
