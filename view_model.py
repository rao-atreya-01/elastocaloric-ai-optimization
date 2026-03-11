import gmsh
import sys

print("Booting up the CAD Engine...")

# 1. Initialize Gmsh (Without the silent command!)
gmsh.initialize(sys.argv)
gmsh.model.add("Elastocaloric_Unit_Cell")

# 2. Hardcode the exact same valid shape from our debug test
width = 1.0
radius = 0.25

print(f"Building a {width}mm box with a {radius}mm hole...")
box = gmsh.model.occ.addBox(0, 0, 0, width, width, width)
cylinder = gmsh.model.occ.addCylinder(width/2, width/2, 0, 0, 0, width, radius)

# Cut the hole out of the box
gmsh.model.occ.cut([(3, box)], [(3, cylinder)])
gmsh.model.occ.synchronize()

# 3. Generate the actual 3D Tetrahedral Mesh
print("Generating Finite Element Mesh...")
gmsh.model.mesh.generate(3)

# 4. THE REVEAL: Open the interactive window
print("Launching 3D Viewer! You can click and drag to rotate.")
print("(Close the pop-up window when you are done to end the script)")
gmsh.fltk.run()

gmsh.finalize()