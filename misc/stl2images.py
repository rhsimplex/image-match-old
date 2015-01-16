"""Blender script for generating reproducible images from stl files.

Needs two arguments from command line: full path of stl, and full path of directory to store results.
Requires blender 2.7+ (previous versions do not have numpy builtin)
$sudo add-apt-repository ppa:irie/blender
$sudo apt-get update
$sudo apt-get install blender

Usage example:

$blender -b -P stl2images.py -- ~/stl_files/example.stl ~/tmp

Explanation:
-b flag runs blender in background (no GUI)
-P flag tells blender to execute a python script (stl2images.py, this file)
-- tells blender to send these arguments to the script, not to blender
~/stl_files/examples.stl is the target STL file
~/tmp the target output directory
"""
from functools import reduce
from operator import add
from mathutils import Matrix, Vector    # blender-specific classes
import numpy
from numpy import identity, pi
from numpy.linalg import eig
from os.path import split, join
import bpy                          # blender-specific module
import sys


# construct a matrix [B] such that [B]v = b x v (cross-product)
def bracketB(v):
    return Matrix([[    0, -v[2], v[1]],
                    [v[2],    0, -v[0]],
                    [-v[1], v[0],   0]])


# dot product is defined for blender Matrix class, but not power
def matrix_square(M):
    return M * M


# see http://en.wikipedia.org/wiki/Moment_of_inertia#Principal_axes
def inertia_matrix(vertices):
    return reduce(add, map(lambda v: -1 * matrix_square(bracketB(v)), vertices))

# get path and output directory
path = sys.argv[-2]
output_dir = sys.argv[-1]

# clear scene
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete(use_global=False)

# load stl
bpy.ops.import_mesh.stl(filepath=path)
bpy.ops.object.select_by_type(type='MESH')

# move object origin to center of mass, move object to global origin
obj = bpy.context.active_object
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

# # uncomment for testing -- apply a random transformation (scale, rotation, translation)
# print('Randomizing orientation...')
# obj.data.transform(Matrix.Rotation(2*pi*numpy.random.rand(), 4, numpy.random.rand(3)) *
#                    Matrix.Translation(10*numpy.random.rand(3)) *
#                    Matrix.Scale(2*(numpy.random.rand() - 1), 4)
#                     )
#
# bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
# # end testing lines

obj.location = [0, 0, 0]
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

# scale by max
factor = 3./max([x.co.magnitude for x in obj.data.vertices.values()])
bpy.ops.transform.resize(value=(factor, factor, factor))
bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')

# get inertia matrix
me = bpy.context.object
vertices = [v.co for v in me.data.vertices.values()]
Ic = inertia_matrix(vertices)

# get principal axes for first pass
evals, evecs = eig(Ic)
e_order = numpy.argsort(evals)[::-1]
evecs = evecs.T
print('\nFirst pass:')
print('Eigenvalue order:')
print(e_order)
print('Eigenvalues:')
print(evals)
print('Eigenvectors:')
print(evecs)

# rotate to principal axes
obj.data.transform(Matrix(evecs).to_4x4())

# switch camera to orthographic perspective
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

# compute inertia tensor again, after alignment, final pass -- principal axes will point along cardinal axes
me = bpy.context.object
vertices = [v.co for v in me.data.vertices.values()]
Ic = inertia_matrix(vertices)

# get principal axes after transform -- the principal axes will point in cardinal directions.
evals, evecs = eig(Ic)
e_order = numpy.argsort(evals)[::-1]
evecs = evecs.T
print('\nFinal pass:')
print('Eigenvalue order:')
print(e_order)
print('Eigenvalues:')
print(evals)
print('Eigenvectors:')
print(evecs)

# rotate to principal axes
obj.data.transform(Matrix(evecs).to_4x4())

# rotation principal axes so that largest eigenvalue points along x, second along y, third along z
alpha = numpy.arccos(Vector([1, 0, 0]) * Vector(evecs[e_order][0]))
beta = numpy.arccos(Vector([0, 1, 0]) * Vector(evecs[e_order][1]))
gamma = numpy.arccos(Vector([0, 0, 1]) * Vector(evecs[e_order][2]))
obj.data.transform(Matrix.Rotation(gamma, 4, [0, 0, 1]))
obj.data.transform(Matrix.Rotation(beta, 4, [0, 1, 0]))
obj.data.transform(Matrix.Rotation(alpha, 4, [1, 0, 0]))

# rotate object among three principal axes and render
for i, v in enumerate(identity(3)):
    # position along the X axis
    scene.camera.location = 5 * Vector([1,0,0])
    scene.objects['Lamp'].location = 5 * Vector([1,0,0])

    # rotate object into position
    obj.data.transform(Matrix.Rotation(pi/2, 4, v))
    
    # render front
    bpy.data.scenes["Scene"].render.filepath = join(output_dir, '%s_%i_front' % (split(path)[-1], i))
    bpy.ops.render.render(write_still=True)

    # position along the negative X axis
    scene.camera.location = -5 * Vector([1,0,0])
    scene.objects['Lamp'].location = -5 * Vector([1,0,0])

    # render back
    bpy.data.scenes["Scene"].render.filepath = join(output_dir, '%s_%i_back' % (split(path)[-1], i))
    bpy.ops.render.render(write_still=True)

    # undo rotation
    obj.data.transform(Matrix.Rotation(-pi/2, 4, v))
