import sys, os,pdb
sys.path.insert(0, '..')
from lib import models_BN, graph, utils, spherical_vertex

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import time
import pdb
import scipy
import pickle as pkl
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES']='1'

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 6, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
# TODO: change cgcnn for combinatorial Laplacians.
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 2, 'Number of coarsened graphs.')
flags.DEFINE_integer('vertex_number',2562,'Number of vertex of the sphere')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('data', 'mnist'), 'Directory to store data.')

#yq
flags.DEFINE_string('f','','kernel')

#r is the radius of the sphere
def grid_graph(m,r,corners=False):
    z,z_theta = graph.grid_sphere(m,r)  #z is rectangular coordinate system while z_theta is polar
    print('z shape is'+str(np.array(z).shape)+'z_theta shape is'+str(np.array(z_theta).shape)) 
    dist, idx = graph.distance_sklearn_metrics(np.array(z), k=FLAGS.number_edges, metric=FLAGS.metric)
    A = graph.adjacency(dist, idx)

    # Connections are only vertical or horizontal on the grid.
    # Corner vertices are connected to 2 neightbors only.
    if corners:
        import scipy.sparse
        A = A.toarray()
        A[A < A.max()/1.5] = 0
        A = scipy.sparse.csr_matrix(A)
        print('{} edges'.format(A.nnz))

    print("{} > {} edges".format(A.nnz//2, FLAGS.number_edges*m**2//2))
    return z,z_theta,A

t_start = time.process_time()
tile_number = int((np.log2((FLAGS.vertex_number-2)/10))/2)   #2**(2*tile_num)*10+2=vertex_number#
graphs_adjacency=[]
z_xyz = []
vertex_theta = []
for m in range(tile_number,tile_number-FLAGS.coarsening_levels-1,-1):
    z,z_theta,A = grid_graph(m,1,corners=False) #m is the tile_number,r is the radius of the sphere
    z_xyz.append(z)
    vertex_theta.append(z_theta)
    graphs_adjacency.append(A)
L = [graph.laplacian(A, normalized=True) for A in graphs_adjacency]
print(len(L))


index = []    
for m in range(FLAGS.coarsening_levels):
    temp = []
    for j in z_xyz[m+1]:
        temp.append(z_xyz[m].index(j))
    index.append(temp)
print('the shape of index',len(index),len(index[0]),len(index[1]))


project_data_path = 'data/mnist_2562_FFT'  # for mnist
#project_data_path = 'data/cifar10_2562_FFT' # for cifar10
#project_data_path = '/mnt/data/modelnet40_40962_TTT' # for modelnet40

if not os.path.exists(project_data_path):
    print("Please generate the dataset!")


train_data = scipy.io.loadmat('{}/train_data.mat'.format(project_data_path))['train_value']
train_data   /= train_data.max()
val_data   = scipy.io.loadmat('{}/val_data.mat'.format(project_data_path))['val_value']
val_data   /= val_data.max()
test_data  = scipy.io.loadmat('{}/test_data.mat'.format(project_data_path))['test_value']
test_data  /= test_data.max()

train_labels = scipy.io.loadmat('{}/train_labels.mat'.format(project_data_path))['train_labels']
val_labels = scipy.io.loadmat('{}/val_labels.mat'.format(project_data_path))['val_labels']
test_labels = scipy.io.loadmat('{}/test_labels.mat'.format(project_data_path))['test_labels']

print('load data from {} done.'.format(project_data_path))



common = {}
common['dir_name']       = 'MNIST/'
common['num_epochs']     = 50
common['batch_size']     = 10
step_num = common['num_epochs']*train_data.shape[0]/common['batch_size']
common['eval_frequency'] = train_data.shape[0]/common['batch_size']
common['brelu']          = 'b1relu'
common['pool']           = 'multipool'
C = max(train_labels) + 1  # number of classes

model_perf = utils.model_perf()


common['regularization'] = 5e-4
common['dropout']        = 0.9
common['is_training']    = True#batch normalization

common['learning_rate']  = [0.02,0.002]
common['boundaries']     = [int(2/3*step_num)]
print('boundaries is {}'.format(common['boundaries']))
common['momentum']       = 0.9

common['F']              = [32, 64]
common['K']              = [25, 25]
common['p']              = [4, 4]
common['M']              = [512, C]

out_path = 'lr{}{}_bou{}_ep{}_bat{}_reg{}_drp{}_BN_mnist_TTT_logk10_K25'.format(common['learning_rate'][0], common['learning_rate'][1], common['boundaries'][0], common['num_epochs'], common['batch_size'], common['regularization'], common['dropout'])
file_out = open(out_path, 'a')
#file_out = None

if True:
    name = out_path
    params = common.copy()
    params['dir_name'] += name
    params['filter'] = 'chebyshev5'
    model_perf.test(models_BN.cgcnn(L, graphs_adjacency, index, z_xyz[-1], file_out, **params), name, params,
                    train_data, train_labels, test_data, test_labels, test_data, test_labels, file_out)
file_out.close()
