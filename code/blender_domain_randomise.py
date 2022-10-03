'''
Domain Randomise Blender Script
Author: cap37
Version: 2.0 Final
Date: 14/09/2022
'''

import bpy
import random
import math
from mathutils import Matrix, Vector
import os


# Randomly size the object
def object_random_size(obj):
    max_dimension = 0.5
    # multiplier on object scale
    scales = [0.8, 1.0, 1.2, 1.4, 1.6]
    scale_factor = max_dimension / max(artefact.dimensions) * random.choice(scales)
    obj.scale = (scale_factor, scale_factor, scale_factor)


# force the camera to look at the object at an offset
def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    points = Vector(tuple([x + random.uniform(-0.2, 0.2) for x in point]))
    direction = points - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')

    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


# random background for mesh
def add_random_background(obj):
    # blender uses 0-1 for colours
    r = random.random()
    g = random.random()
    b = random.random()
    alpha = 1
    color = (r, g, b, alpha)
    mat = bpy.data.materials.new("random_mat")
    mat.diffuse_color = color
    obj.data.materials.append(mat)


# randomly rotate the object for pose
def random_rotation(obj):
    obj.rotation_euler[0] = random.randrange(0, 360)
    obj.rotation_euler[1] = random.randrange(0, 360)
    obj.rotation_euler[2] = random.randrange(0, 360)


# randomly spawn the distractor objects at an offset
def random_spawn_location(obj):
    # set an offset to avoid distractors colliding with main object ASSUMING OBJECT @ 0,0,0
    offset = [-1.5, -1.0, -0.8, 0.8, 1.0, 1.5]
    obj.location = (random.random() + random.choice(offset),
                    random.random() + random.choice(offset),
                    0)


# spawn in distractor objects
def distractor_object():
    for elem in range(random.randint(1, 2)):
        bpy.ops.mesh.primitive_cube_add(size=0.4)
        distractor = bpy.context.active_object
        random_spawn_location(distractor)
        add_random_background(distractor)
        object_random_size(distractor)

    for elem in range(random.randint(1, 2)):
        bpy.ops.mesh.primitive_cone_add(radius1=0.4, depth=0.5, rotation=(0, 3.15, 0))
        distractor = bpy.context.active_object
        random_spawn_location(distractor)
        add_random_background(distractor)
        object_random_size(distractor)

    for elem in range(random.randint(1, 2)):
        bpy.ops.mesh.primitive_cylinder_add(radius=0.2, depth=0.3)
        distractor = bpy.context.active_object
        random_spawn_location(distractor)
        add_random_background(distractor)
        object_random_size(distractor)


# create chequered background
def create_checkered_plane_background():
    x_loc = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    y_loc = [-2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5]
    z_loc = 0.5
    combo = [(x, y) for x in x_loc for y in y_loc]
    for val in combo:
        bpy.ops.mesh.primitive_plane_add(size=0.5)
        plane = bpy.context.active_object

        plane.location = (val[0], val[1], z_loc)
        add_random_background(plane)


# create solid plane
def create_solid_plane():
    bpy.ops.mesh.primitive_plane_add(size=0.5)
    plane = bpy.context.active_object

    plane.location = (0, 0, 0.5)
    add_random_background(plane)


# add random lighting in the scene
def add_lighting(min_n=1, max_n=5):
    number_of_lights = random.randint(min_n, max_n)
    for i in range(0, number_of_lights):
        bpy.ops.object.light_add(type='POINT')
        light_ob = bpy.context.object
        light = light_ob.data
        light.energy = random.randint(1000, 10000)
        light.color = (random.uniform(0.0, 1.0), random.uniform(0.0, 1.0),
                       random.uniform(0.0, 1.0)
                       )
        light_ob.location = (random.randint(-10, 10),
                             random.randint(-10, 10),
                             random.randint(-10, -4))

# reset local and global origins
def origin_to_bottom(obj, matrix=Matrix()):
    me = obj.data
    mw = obj.matrix_world
    local_verts = [matrix @ Vector(v[:]) for v in obj.bound_box]
    o = sum(local_verts, Vector()) / 8
    o.z = min(v.z for v in local_verts)
    o = matrix.inverted() @ o
    me.transform(Matrix.Translation(-o))

    mw.translation = mw @ o

# define shading mode of render
def set_shading_mode(mode="SOLID", screens=[]):
    """
    Performs an action analogous to clicking on the display/shade button of
    the 3D view. Mode is one of "RENDERED", "MATERIAL", "SOLID", "WIREFRAME".
    The change is applied to the given collection of bpy.data.screens.
    If none is given, the function is applied to bpy.context.screen (the
    active screen) only. E.g. set all screens to rendered mode:
      set_shading_mode("RENDERED", bpy.data.screens)
    """
    screens = screens if screens else [bpy.context.screen]
    for s in screens:
        for spc in s.areas:
            if spc.type == "VIEW_3D":
                spc.spaces[0].shading.type = mode
                break  # expect at most 1 VIEW_3D space

# input and output directories
input_model_directory = "/models_five"
output_render_directory = ""

for filename in os.listdir(input_model_directory):
    if filename.endswith('.stl'):
        for n in range(0, 1000):
            for o in bpy.context.scene.objects:
                # camera is considered a light, this works better
                if o.type != 'CAMERA':
                    o.select_set(True)
                else:
                    o.select_set(False)

            bpy.ops.object.delete()

            renderer = bpy.context.scene.render
            renderer.resolution_x = 512  # 256
            renderer.resolution_y = 512  # 256
            # bpy.data.scenes["Scene"].render.engine = 'BLENDER_EEVEE' # better resolution, slightly slower
            bpy.data.scenes['Scene'].eevee.taa_render_samples = 1  # lower resoultion, slightly faster

            bpy.ops.import_mesh.stl(filepath=input_model_directory + "/" + filename)
            artefact = bpy.context.active_object
            origin_to_bottom(artefact)

            if max(artefact.dimensions) == 0:
                print("error on filename: ", filename)
                break

            object_random_size(artefact)
            random_rotation(artefact)
            add_random_background(artefact)
            # all images will have the object in the middle
            artefact.location = (0, 0, 0)

            add_lighting()
            # 80/20 chequered/solid
            if random.random() >= 0.8:
                create_solid_plane()
            else:
                create_checkered_plane_background()
            # 50/50 distractor objects
            if random.random() >= 0.5:
                distractor_object()
            # spawn camera location
            camera = bpy.data.objects['Camera']
            camera.location.x = 0
            camera.location.y = 0
            camera.location.z = -3
            # force camera to look at object
            look_at(camera, artefact.matrix_world.to_translation())

            bpy.data.cameras['Camera'].lens = random.randint(20, 60)

            set_shading_mode(random.choice(["RENDERED", "MATERIAL", "SOLID"]))

            folder_name = output_render_directory + '/' + filename.replace('.stl', '')

            if not os.path.exists(os.path.dirname(folder_name)):
                os.makedirs(os.path.dirname(folder_name))

            bpy.data.scenes['Scene'].render.filepath = (
                    folder_name
                    + '/'
                    + filename.replace('.stl', '')
                    + '_'
                    + str(n)
                    + '.png'
            )

            bpy.ops.render.render(write_still=True)

            bpy.ops.render.render(True)

            for o in bpy.context.scene.objects:
                if o.type == 'MESH':
                    o.select_set(True)
                else:
                    o.select_set(False)

            bpy.ops.object.delete()

    else:
        continue
