import trimesh
import numpy as np
from trimesh.creation import box, extrude_polygon
from trimesh.boolean import difference
from trimesh.path.creation import text as svg_text

def create_number_mesh(number, height=1.0, depth=0.1):
    # Genera un path 2D SVG
    path = svg_text(text=str(number), font='Arial', font_size=20)
    
    # Estrudi il path in un solido 3D
    mesh = extrude_polygon(path.polygons_full[0], height=depth)
    mesh.apply_translation(-mesh.center_mass)
    return mesh

def get_face_translation(face, size, offset):
    half = size / 2.0 - offset
    return {
        '+X': [ half, 0, 0],
        '-X': [-half, 0, 0],
        '+Y': [0,  half, 0],
        '-Y': [0, -half, 0],
        '+Z': [0, 0,  half],
        '-Z': [0, 0, -half],
    }[face]

def get_rotation_matrix(face):
    from trimesh.transformations import rotation_matrix
    axis_angle = {
        '+Z': ([0, 0, 1], 0),
        '-Z': ([1, 0, 0], np.pi),
        '+Y': ([1, 0, 0], -np.pi / 2),
        '-Y': ([1, 0, 0], np.pi / 2),
        '+X': ([0, 1, 0], -np.pi / 2),
        '-X': ([0, 1, 0], np.pi / 2),
    }
    axis, angle = axis_angle[face]
    return rotation_matrix(angle, axis)

def create_dice(size=1.0, depth=0.1):
    cube = box(extents=[size, size, size])
    faces = ['-Z', '+Z', '-Y', '+Y', '-X', '+X']
    numbers = [1, 6, 2, 5, 3, 4]  # Opposte sommano a 7

    number_meshes = []
    for face, num in zip(faces, numbers):
        number = create_number_mesh(num, depth=depth)
        rot = get_rotation_matrix(face)
        number.apply_transform(rot)
        trans = get_face_translation(face, size, offset=depth/2)
        number.apply_translation(trans)
        number_meshes.append(number)

    # Unione dei numeri
    combined = trimesh.util.concatenate(number_meshes)
    result = difference([cube, combined])
    return result

# Crea e salva il dado
dice = create_dice()
dice.export('dice.stl')
print("Dado salvato come dice.stl")
