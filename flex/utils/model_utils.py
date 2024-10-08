# utils about URDF, MJCF and XML stuff



import os
import sys
from shutil import copyfile

import mujoco

import trimesh
import numpy as np
import argparse
# from dm_control import mjcf 
import struct 
import xml.etree.ElementTree as ET


def print_usage():
    print("""python compile_mjcf_model.py input_file output_file""")


def compile_mjcf_model(input_file, output_file):
    """Loads a raw mjcf file and saves a compiled mjcf file.

    This avoids mujoco-py from complaining about .urdf extension.
    Also allows assets to be compiled properly.

    Example:
        $ python compile_mjcf_model.py source_mjcf.xml target_mjcf.xml
    """
    input_folder = os.path.dirname(input_file)

    tempfile = os.path.join(input_folder, ".robosuite_temp_model.xml")
    copyfile(input_file, tempfile)

    model = mujoco.MjModel.from_xml_path(tempfile)
    xml_string = model.get_xml()
    with open(output_file, "w") as f:
        f.write(xml_string)

    os.remove(tempfile)

def convert_obj_to_stl(file_dir, output_dir):
    # find all obj files in the directory
    obj_files = [f for f in os.listdir(file_dir) if f.endswith('.obj')]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for obj_file in obj_files:
        obj_path = os.path.join(file_dir, obj_file)
        mesh = trimesh.load(obj_path)
        mesh.export(file_obj=os.path.join(output_dir, obj_file.replace('.obj', '.stl')))
        print(f"Converted {obj_file} to {obj_file.replace('.obj', '.stl')}")
    print(f"Converted {len(obj_files)} obj files to stl files")


def ascii_to_binary_stl(file_dir, outpur_dir):
    # convert all ascii stl files to binary stl files for mujoco
    stl_files = [f for f in os.listdir(file_dir) if f.endswith('.stl')]
    # create output directory if it does not exist
    if not os.path.exists(outpur_dir):
        os.makedirs(outpur_dir)
    
    for stl_file in stl_files:
        stl_path = os.path.join(file_dir, stl_file)
        mesh = trimesh.load(stl_path)
        mesh.export(file_obj=os.path.join(outpur_dir, stl_file), file_type='stl')
        print(f"Converted {stl_file} to binary stl")
    print(f"Converted {len(stl_files)} stl files to binary stl files")
    
def extrude_2d_meshes(file_dir, output_dir):
    '''
    check for all 2D meshes (stl) in the directory and extrude them to 3D meshes
    '''
    # find all stl files in the directory
    stl_files = [f for f in os.listdir(file_dir) if f.endswith('.stl')]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for stl_file in stl_files:
        stl_path = os.path.join(file_dir, stl_file)
        mesh = trimesh.load(stl_path)
        # check if the mesh is 2D using watertight
        if is_2d_plane(mesh):
            print(f"Extruding {stl_file} to 3D mesh")
            # get the outline of the mesh, move it to 2D, save the transform
            # on_plane, to_3D = mesh.outline().to_planar()
            # # extrude the outline into a solid    
            # extrude = on_plane.extrude(0.01)                                                          
            # extrude = on_plane.extrude(0.01).to_mesh().apply_transform(to_3D)
            # assert extrude.is_watertight, "Extruded mesh is not watertight"
            # trimesh.Scene([extrude, mesh]).show()

            # extrude the mesh
            extrude = extrude_single_2d_mesh(mesh, 0.001)
            # show the extruded mesh
            trimesh.Scene([extrude, mesh]).show()

            # save the extruded mesh
            extrude.export(file_obj=os.path.join(output_dir, stl_file), file_type='stl')
        else:
            # copy the mesh to the output directory
            copyfile(stl_path, os.path.join(output_dir, stl_file))

def preprocess_meshes(file_dir, output_dir):
    # read obj, extrude them, and save them as binary stl

    # find all obj files in the directory
    obj_files = [f for f in os.listdir(file_dir) if f.endswith('.obj')]
    # create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for obj_file in obj_files:
        obj_path = os.path.join(file_dir, obj_file)
        print("path:", obj_path)
        mesh = trimesh.load(obj_path, force='mesh')
        # check if the mesh is 2D using watertight
        if is_2d_plane(mesh):
            print(f"Extruding {obj_file} to 3D mesh")
            # get the outline of the mesh, move it to 2D, save the transform
            # on_plane, to_3D = mesh.outline().to_planar()
            # # extrude the outline into a solid    
            # extrude = on_plane.extrude(0.01)                                                          
            # extrude = on_plane.extrude(0.01).to_mesh().apply_transform(to_3D)
            # assert extrude.is_watertight, "Extruded mesh is not watertight"
            # trimesh.Scene([extrude, mesh]).show()

            # extrude the mesh
            extrude = extrude_single_2d_mesh(mesh, 0.01)
            # show the extruded mesh
            trimesh.Scene([extrude, mesh]).show()

            # save the extruded mesh
            extrude.export(file_obj=os.path.join(output_dir, obj_file.replace('.obj', '.stl')), file_type='stl')
        else:
            # export the mesh as stl
            print(f"Converting {obj_file} to stl")
            mesh.export(file_obj=os.path.join(output_dir, obj_file).replace('.obj', '.stl'), file_type='stl')


# def extrude_single_2d_mesh(mesh, extrusion_height):
#     """
#     Extrudes a 2D mesh along the z-axis by a specified height.

#     Args:
#         mesh (trimesh.Trimesh): The 2D mesh to be extruded.
#         extrusion_height (float): The height of the extrusion.

#     Returns:
#         trimesh.Trimesh: The extruded 3D mesh.
#     """
#     # Get vertices and faces of the 2D mesh
#     vertices = mesh.vertices
#     faces = mesh.faces

#     # Create a copy of the vertices and move them along the z-axis
#     extruded_vertices = np.copy(vertices)
#     extruded_vertices[:, 2] += extrusion_height

#     # Create new faces for the extruded mesh
#     extruded_faces = []
#     for face in faces:
#         # Original face
#         original_face = list(face)
#         # Extruded face
#         extruded_face = [v + len(vertices) for v in face]
#         # Faces connecting original vertices to extruded vertices
#         for i in range(len(face)):
#             next_i = (i + 1) % len(face)
#             extruded_faces.append([original_face[i], original_face[next_i], extruded_face[next_i]])
#             extruded_faces.append([original_face[i], extruded_face[next_i], extruded_face[i]])
#         # Add the faces of the original mesh (optional, if you want to keep the base)
#         extruded_faces.append([original_face[0], original_face[1], original_face[2]])
    
#     # Combine original vertices and extruded vertices
#     combined_vertices = np.vstack([vertices, extruded_vertices])
    
#     # Create the extruded mesh
#     extruded_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=extruded_faces, process=False)

#     return extruded_mesh



def extrude_single_2d_mesh(mesh, extrusion_height):
    """
    Extrudes a 2D mesh along its normal direction by a specified height.

    Args:
        mesh (trimesh.Trimesh): The 2D mesh to be extruded.
        extrusion_height (float): The height of the extrusion.

    Returns:
        trimesh.Trimesh: The extruded 3D mesh.
    """
    # Ensure the mesh is 2D
    if mesh.is_watertight:
        raise ValueError("Mesh is watertight. Please provide a valid 2D mesh.")
    
    # Get vertices and faces of the 2D mesh
    vertices = mesh.vertices
    faces = mesh.faces

    # Compute the normal vector of the mesh (assumed to be uniform)
    normal = mesh.face_normals[0]

    # Determine extrusion direction
    extrusion_direction = normal / np.linalg.norm(normal) * extrusion_height

    # Create a copy of the vertices and move them along the extrusion direction
    extruded_vertices = np.copy(vertices)
    extruded_vertices += extrusion_direction

    # Create new faces for the extruded mesh
    extruded_faces = []
    for face in faces:
        # Original face
        original_face = list(face)
        # Extruded face
        extruded_face = [v + len(vertices) for v in face]
        # Faces connecting original vertices to extruded vertices
        for i in range(len(face)):
            next_i = (i + 1) % len(face)
            extruded_faces.append([original_face[i], original_face[next_i], extruded_face[next_i]])
            extruded_faces.append([original_face[i], extruded_face[next_i], extruded_face[i]])
        # Add the faces of the original mesh (optional, if you want to keep the base)
        extruded_faces.append([original_face[0], original_face[1], original_face[2]])
    
    # Combine original vertices and extruded vertices
    combined_vertices = np.vstack([vertices, extruded_vertices])
    
    # Create the extruded mesh
    extruded_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=extruded_faces, process=False)

    return extruded_mesh


def is_2d_plane(mesh):
    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces
    
    # Extract indices of the first face
    face_indices = faces[0]
    if len(face_indices) < 3:
        # A face with less than 3 vertices cannot define a plane
        return False

    # Get the vertices of the first face
    p1, p2, p3 = vertices[face_indices[:3]]

    # Compute the normal vector of the plane formed by these points
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)
    
    if norm == 0:
        # The points are collinear, so they do not define a plane
        return False
    
    # Normalize the normal vector
    normal /= norm
    
    # Compute the plane constant d
    d = -np.dot(normal, p1)
    
    # Check if all vertices lie on the same plane
    for v in vertices:
        if not np.isclose(np.dot(normal, v) + d, 0, atol=1e-6):
            return False
    
    return True

def read_stl(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(80) 
        n_tri = struct.unpack('<I', f.read(4))[0] 
        b_str = f.read()
    return b_str



def scale_object_new(original_xml_path, output_name = None, scale=[0.3, 0.3, 0.3]):

    original_xml_dir = os.path.dirname(original_xml_path)
    tree = ET.parse(original_xml_path) 
    root = tree.getroot() 
    # print(root.__dir__())
    for mesh in root.findall('.//mesh'):
        mesh.set('scale', f'{scale[0]} {scale[1]} {scale[2]}') 
        # print(mesh) 
    
    for geom in root.findall('.//geom'):
        # print(geom.set(''))
        if 'pos' in geom.attrib:
            pos_str = geom.get('pos')
            pos_strs = pos_str.split(' ') 
            pos = [scale[i] * float(p) for i, p in enumerate(pos_strs)]
            geom.set('pos', f'{pos[0]} {pos[1]} {pos[2]}')

    for body in root.findall('.//body'): 
        if 'pos' in body.attrib:
            pos_str = body.get('pos')
            pos_strs = pos_str.split(' ') 
            pos = [scale[i] * float(p) for i, p in enumerate(pos_strs)]
            body.set('pos', f'{pos[0]} {pos[1]} {pos[2]}') 

    for joint in root.findall('.//joint'): 
        if 'pos' in joint.attrib:
            pos_str = joint.get('pos')
            pos_strs = pos_str.split(' ') 
            pos = [scale[i] * float(p) for i, p in enumerate(pos_strs)]
            joint.set('pos', f'{pos[0]} {pos[1]} {pos[2]}')
        # scale the joint limit if it is slide
        if 'type' in joint.attrib and joint.get('type') == 'slide':
            # print('scaling joint limit')
            if 'range' in joint.attrib:
                range_str = joint.get('range')
                range_strs = range_str.split(' ')
                range_vals = [scale[i] * float(r) for i, r in enumerate(range_strs)]
                joint.set('range', f'{range_vals[0]} {range_vals[1]}')
        
    for inertial in root.findall('.//inertial'):
        if 'pos' in inertial.attrib:
            pos_str = inertial.get('pos')
            pos_strs = pos_str.split(' ') 
            pos = [scale[i] * float(p) for i, p in enumerate(pos_strs)]
            inertial.set('pos', f'{pos[0]} {pos[1]} {pos[2]}')
    # TODO: also need to scale the joint limit if it is prismatic


    if output_name:
        output_path = os.path.join(original_xml_dir, output_name)
    else:
        # set the output_name to be input_name + '-scaled.xml'
        output_path = os.path.join(original_xml_dir, os.path.basename(original_xml_path).replace('.xml', '-scaled.xml'))
    tree.write(output_path)

        

if __name__ == "__main__":

    # if len(sys.argv) != 4:
    #     print_usage()
    #     exit(0)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', '--input', type=str, help='input directory')
    parser.add_argument('-o', '--output', type=str, help='output directory')
    parser.add_argument('-f', '--func', type=str, help='function to execute', default="preprocess_meshes")

    args = parser.parse_args()

    if args.func == "preprocess_meshes":
        preprocess_meshes(args.input, args.output)
    else:
        raise NotImplementedError(f"Function {args.func} not implemented")

    # object_path = "./sim_models/microwave-2/microwave-2-original.xml"
    # scale_object_new(object_path, 'sim_models/microwave-2/microwave-2-test.xml')


    # func_name = sys.argv[1]
    # input_dir = sys.argv[2]
    # output_dir = sys.argv[3]
    # # execute the function
    # if func_name == "convert_obj_to_stl":
    #     convert_obj_to_stl(input_dir, output_dir)
    # elif func_name == "ascii_to_binary_stl":
    #     ascii_to_binary_stl(input_dir, output_dir)
    # elif func_name == "compile_mjcf_model":
    #     compile_mjcf_model(input_dir, output_dir)
    # elif func_name == "extrude_2d_meshes":
    #     extrude_2d_meshes(input_dir, output_dir)
    # elif func_name == "convert_mtl_to_stl":
    #     convert_mtl_to_stl(input_dir, output_dir)
    # else:
    #     print(f"Function {func_name} not found")
    #     print_usage()
    #     exit(0)
