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

flags = tf.app.flags
FLAGS = flags.FLAGS

# Graphs.
flags.DEFINE_integer('number_edges', 6, 'Graph: minimum number of edges per vertex.')
flags.DEFINE_string('metric', 'euclidean', 'Graph: similarity measure (between features).')
flags.DEFINE_bool('normalized_laplacian', True, 'Graph Laplacian: normalized.')
flags.DEFINE_integer('coarsening_levels', 2, 'Number of coarsened graphs.')
flags.DEFINE_integer('vertex_number',2562,'Number of vertex of the sphere')

# Directories.
flags.DEFINE_string('dir_data', os.path.join('data', 'mnist'), 'Directory to store data.')
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
    #A = graph.replace_random_edges(A, 0) retain the hexagonal shape
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


def load_mnist(FLAGS):
    mnist = input_data.read_data_sets(FLAGS.dir_data, one_hot=False)

    train_data = mnist.train.images.astype(np.float32).reshape(-1, 28,28)
    val_data = mnist.validation.images.astype(np.float32).reshape(-1,28,28)
    test_data = mnist.test.images.astype(np.float32).reshape(-1, 28,28)
    train_labels = mnist.train.labels
    val_labels = mnist.validation.labels
    test_labels = mnist.test.labels

    print('Load mnist dataset done.')
    return train_data, train_labels, val_data, val_labels, test_data, test_labels


def _load_cifar10_batch(file_name):
    '''
    Read cifar10 per file by pickle
    Args:
        file_name
    Return:
        X: data with shape [10000, 32, 32, 3]
        Y: label with shape [10000]
    '''
    with open(file_name, 'rb') as f:
        data_dict = pkl.load(f, encoding='latin1')
        X = data_dict['data']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
        Y = data_dict['labels']
        Y = np.array(Y)
    return X,Y
    

def load_cifar10(root_path='./data/cifar10'):
    '''
    Load cifar10 dataset
    Args:
        root_path: path of cifar10 dataset, not contain 'cifar-10-batches-py'
    Return:
        train_data:   shape [50000, 32, 32, 3]
        train_labels: shape [50000]
        test_data:    shape [10000, 32, 32, 3]
        test_labels:  shape [10000]
    '''
    train_batches = ['{}/cifar-10-batches-py/data_batch_{}'.format(root_path, i+1) for i in range(5)]
    test_batch = '{}/cifar-10-batches-py/test_batch'.format(root_path)
    train_data   = []
    train_labels = []
    for train_batch in train_batches:
        tmp_data, tmp_label = _load_cifar10_batch(train_batch)
        train_data.append(tmp_data)
        train_labels.append(tmp_label)
    train_data   = np.vstack(train_data)
    train_labels = np.hstack(train_labels)
    val_data     = train_data[45000:]
    val_labels   = train_labels[45000:]
    train_data   = train_data[:45000]
    train_labels = train_labels[:45000]
    test_data, test_labels = _load_cifar10_batch(test_batch)
    print('Load cifar10 dataset done.')
    return train_data, train_labels, val_data, val_labels, test_data, test_labels
   
def load_modelnet40(root_path='/home/yq/models/PDOs-master'):
    root_path = '{}/experiments/exp2_modelnet40/data'.format(root_path)    
    classes = ['airplane', 'bowl', 'desk', 'keyboard', 'person', 'sofa', 'tv_stand', 'bathtub', 'car', 'door',
                       'lamp', 'piano', 'stairs', 'vase', 'bed', 'chair', 'dresser', 'laptop', 'plant', 'stool',
                       'wardrobe', 'bench', 'cone', 'flower_pot', 'mantel', 'radio', 'table', 'xbox', 'bookshelf', 'cup',
                       'glass_box', 'monitor', 'range_hood', 'tent', 'bottle', 'curtain', 'guitar', 'night_stand', 'sink', 'toilet']
    class2index = dict(zip(classes, range(len(classes))))
    files_train = ['{}/modelnet40_train/{}'.format(root_path, f) for f in os.listdir('{}/modelnet40_train'.format(root_path)) if 'npy' in f]
    files_test  = ['{}/modelnet40_test/{}'.format(root_path, f) for f in os.listdir('{}/modelnet40_test'.format(root_path)) if 'npy' in f]
    train_data = [np.load(f) for f in files_train]
    train_data = np.stack(train_data).transpose([0,2,1])
    test_data  = [np.load(f) for f in files_test]
    test_data  = np.stack(test_data).transpose([0,2,1])
    train_labels = []
    for f in files_train:
        train_labels.append(class2index[f.split('/')[-1].strip('.npy')[4:-5]])
    train_labels = np.array(train_labels)
    test_labels = []
    for f in files_test:
        test_labels.append(class2index[f.split('/')[-1].strip('.npy')[4:-5]])
    test_labels = np.array(test_labels)
    print('train_data shape ', train_data.shape, 'train_labels shape', train_labels.shape)
    print('test_data shape ', test_data.shape, 'test_labels shape', test_labels.shape)
    val_data   = test_data[:1000]
    val_labels = test_labels[:1000]

    return train_data, train_labels, val_data, val_labels, test_data, test_labels#, points_raw

points_raw = None
train_data, train_labels, val_data, val_labels, test_data, test_labels = load_mnist(FLAGS)
#train_data, train_labels, val_data, val_labels, test_data, test_labels = load_cifar10()
#train_data, train_labels, val_data, val_labels, test_data, test_labels = load_modelnet40()


train_data = train_data / train_data.max(0,keepdims=True).max(1,keepdims=True)
val_data   = val_data   / val_data.max(0,keepdims=True).max(1,keepdims=True)
test_data  = test_data  / test_data.max(0,keepdims=True).max(1,keepdims=True)
print('load dataset done.')



#transform to sphere_data


img_w = train_data.shape[1]               #img_w is the size of img
if img_w>1000:
    img_w = 2
#radius = img_w/0.83   #80*80/180*360
radius = img_w/1.73   #120*120/180*360
#radius = img_w/5.67   #160*160/180*360


project_data_path = 'data/mnist_2562_FFT'  # for mnist
#project_data_path = 'data/cifar10_2562_FFT' # for cifar10 inter
#project_data_path = '/mnt/data/modelnet40_40962_TTT' # for modelnet40


print('start to project dataset')
if not os.path.exists(project_data_path):
    os.mkdir(project_data_path)

train_data=spherical_vertex.project_point(train_data,radius,vertex_theta[0],rotation=False, selfRotation=False, points_raw=points_raw)
scipy.io.savemat('{}/train_data.mat'.format(project_data_path), {'train_value':train_data})

val_data=spherical_vertex.project_point(val_data,radius,vertex_theta[0],rotation=False, selfRotation=False, points_raw=points_raw)
scipy.io.savemat('{}/val_data.mat'.format(project_data_path), {'val_value':val_data})

test_data=spherical_vertex.project_point(test_data,radius,vertex_theta[0],rotation=True, selfRotation=False, points_raw=points_raw)
scipy.io.savemat('{}/test_data.mat'.format(project_data_path), {'test_value':test_data})

scipy.io.savemat('{}/train_labels.mat'.format(project_data_path), {'train_labels':train_labels})
scipy.io.savemat('{}/val_labels.mat'.format(project_data_path), {'val_labels':val_labels})
scipy.io.savemat('{}/test_labels.mat'.format(project_data_path), {'test_labels':test_labels})
print('project dataset done')




