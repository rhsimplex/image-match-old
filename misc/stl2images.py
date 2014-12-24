from functools import reduce
from operator import add
from mathutils import Matrix
from numpy import identity
from numpy.linalg import eig
from os import curdir
import bpy

# construct a matrix [B] such that [B]y = b x y
def bracketB(v):
    return Matrix([[    0,-v[2], v[1]],
                    [v[2],    0,-v[0]],
                    [-v[1],v[0],   0]])

# dot product is defined for blender Matrix class, but not power
def matrix_square(M):
    return M * M

# see http://en.wikipedia.org/wiki/Moment_of_inertia#Principal_axes
def inertia_matrix(vertices):
    return reduce(add, map(lambda v: -1 * matrix_square(bracketB(v)), vertices))

# clear scene
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_by_type(type = 'MESH')
bpy.ops.object.delete(use_global=False)

# load stl
bpy.ops.import_mesh.stl(filepath="C:\\Users\\ryan\\Downloads\\brain-gear\\brain-gear.stl")
bpy.ops.object.select_by_type(type = 'MESH')

# move object origin to center of mass, move object to global origin
obj = bpy.context.active_object
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
obj.location = [0,0,0]

# scale by max
factor = 3./max([x.co.magnitude for x in obj.data.vertices.values()])
bpy.ops.transform.resize(value=(factor, factor, factor))

# get inertia matrix
me = bpy.context.object
vertices = [v.co for v in me.data.vertices.values()]
Ic = inertia_matrix(vertices)

# get principal axes
evals, evecs = eig(Ic)
evecs = evecs.T

# rotate to principal axes
obj.data.transform(Matrix(evecs).to_4x4())

# swicth camera to orthorhombic perspective
scene = bpy.data.scenes["Scene"]
scene.camera.data.type = 'ORTHO'

# set camera to always point to object
cns = scene.camera.constraints.new('TRACK_TO')
cns.target = me
cns.track_axis = 'TRACK_NEGATIVE_Z'
cns.up_axis = 'UP_Y'

# set render resolution
scene.render.resolution_x = 1024
scene.render.resolution_y = 1024

# set camera and lighting along each of x, y, and z axes
for i, v in enumerate(identity(3)):
    # position along each cardinal axis
    scene.camera.location = 5 * v
    scene.objects['Lamp'].location = 5 * v
    
    # render front
    bpy.data.scenes["Scene"].render.filepath = 'c:\\tmp\\front_%i' % i
    bpy.ops.render.render(write_still=True)
    
    # flip
    scene.camera.location = -5 * v
    scene.objects['Lamp'].location = -5 * v
    
    # render back
    bpy.data.scenes["Scene"].render.filepath = 'c:\\tmp\\back_%i' % i
    bpy.ops.render.render(write_still=True)