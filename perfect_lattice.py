import gmsh
import sys

def generate_perfect_lattice():
    gmsh.initialize()
    gmsh.model.add("Elastocaloric_Metamaterial")
    factory = gmsh.model.occ
    
    print("Building mathematically perfect CAD geometry...")
    box = factory.addBox(0, 0, 0, 10, 10, 10)
    
    cyl_x = factory.addCylinder(0, 5, 5, 10, 0, 0, 3)
    cyl_y = factory.addCylinder(5, 0, 5, 0, 10, 0, 3)
    cyl_z = factory.addCylinder(5, 5, 0, 0, 0, 10, 3)
    
    factory.cut([(3, box)], [(3, cyl_x), (3, cyl_y), (3, cyl_z)])
    factory.synchronize()
    
    # =======================================================
    # THE FIX: Tell FEniCSx which part is the actual metal!
    # =======================================================
    volumes = gmsh.model.getEntities(3)
    volume_tags = [v[1] for v in volumes]
    # We group the 3D volume and tag it so FEniCSx can see it
    gmsh.model.addPhysicalGroup(3, volume_tags, tag=1)
    gmsh.model.setPhysicalName(3, 1, "Metal_Volume")
    # =======================================================
    
    print("Packing volume with 3D Tetrahedrons...")
    gmsh.model.mesh.generate(3)
    
    filename = "perfect_physics_lattice.msh"
    gmsh.write(filename)
    
    print(f"\nSUCCESS! 3D Physics Mesh saved as: {filename}")
    gmsh.finalize()

if __name__ == "__main__":
    generate_perfect_lattice()
