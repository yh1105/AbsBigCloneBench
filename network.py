"""Build a CNN network that learns a convolution over a tree structure as
described in Lili Mou et al. (2015) https://arxiv.org/pdf/1409.5718.pdf"""

import math
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
def init_net_finetune(feature_size, embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None, ), name='tree')
        children1 = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        nodes22 = tf.placeholder(tf.int32, shape=(None, ), name='tree')
        children2 = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes1 = tf.expand_dims(nodes1, 0)
        nodes2 = tf.expand_dims(nodes2, 0)
    with tf.name_scope('network'):
        conv1,conv2 = conv_layer(1,conv_dim, nodes1, children1, nodes2, children2, feature_size)
        pooling1 = pooling_layer(conv1)
        pooling2 = pooling_layer(conv2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11, children1,nodes22,children2,res

def init_net_finetune_oneconv(feature_size, embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None, ), name='tree1')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
    with tf.name_scope('network'):
        std = 1.0 / math.sqrt(feature_size)
        init = tf.truncated_normal([conv_dim, ], stddev=math.sqrt(2.0 / feature_size))
        b_conv = tf.Variable(init, name="b_conv")
        w_conv = (
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wn"),
        )
        conv1=tf.tensordot(nodes1,tf.reshape(w_conv,[feature_size,conv_dim]),axes=1)+b_conv
        pooling1 = pooling_layer(conv1)
        conv2 = tf.tensordot(nodes2, tf.reshape(w_conv,[feature_size,conv_dim]),axes=1) + b_conv
        pooling2 = pooling_layer(conv2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11,nodes22,res
def init_net_finetune_oneconv_nohidden(feature_size, embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None, ), name='tree1')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
    with tf.name_scope('network'):
        std = 1.0 / math.sqrt(feature_size)
        init = tf.truncated_normal([conv_dim, ], stddev=math.sqrt(2.0 / feature_size))
        b_conv = tf.Variable(init, name="b_conv")
        w_conv = (
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wn"),
        )
        conv1=tf.tensordot(nodes1,tf.reshape(w_conv,[feature_size,conv_dim]),axes=1)+b_conv
        pooling1 = pooling_layer(conv1)
        conv2 = tf.tensordot(nodes2, tf.reshape(w_conv,[feature_size,conv_dim]),axes=1) + b_conv
        pooling2 = pooling_layer(conv2)
        res=computCos(pooling1,pooling2)
    return nodes11,nodes22,res

def init_net_finetuneConsiderMoreNoHidden(feature_size,embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None,), name='tree')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        children1 = tf.placeholder(tf.int32, shape=(1, None, None), name='children1')
        parent1=tf.placeholder(tf.int32, shape=(None, 2), name='parent1')
        subl1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl1')
        subr1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr1')
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
        children2 = tf.placeholder(tf.int32, shape=(1, None, None), name='children2')
        parent2 = tf.placeholder(tf.int32, shape=(None, 2), name='parent2')
        subl2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl2')
        subr2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        wn,wp=(
            tf.Variable(tf.truncated_normal([feature_size,conv_dim],stddev=std),name="Wn"),
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wp")
        )
        initnp=tf.truncated_normal([conv_dim,],stddev=math.sqrt(2.0/feature_size))
        b_convp=tf.Variable(initnp,name="b_convp")
        node_p1=tf.gather_nd(nodes1,parent1)
        node_p2=tf.gather_nd(nodes2,parent2)
        convtemp1=tf.nn.tanh(tf.tensordot(nodes1,wn,axes=1)+tf.tensordot(node_p1,wp,axes=1)+b_convp)
        convtemp2=tf.nn.tanh(tf.tensordot(nodes2,wn,axes=1)+tf.tensordot(node_p2,wp,axes=1)+b_convp)
        conv1,conv2,convl1,convl2,convr1,convr2=new_conv_layer(1,conv_dim,nodes1,children1,subl1,subr1,nodes2,children2,subl2,subr2,feature_size)
        conv1o=tf.concat([conv1,convl1,convr1,convtemp1],axis=0)
        conv2o = tf.concat([conv2,convl2,convr2,convtemp2],axis=0)
        w_one_by_one=tf.Variable(tf.truncated_normal([1,1,4],stddev=std),name="Wone")
        b_one_by_one=tf.Variable(0.1,name="Bone")
        convStep2_1=tf.tensordot(w_one_by_one,conv1o,axes=1)+b_one_by_one
        convStep2_2 = tf.tensordot(w_one_by_one, conv2o, axes=1) + b_one_by_one
        convStep2_1=tf.reduce_mean(convStep2_1,axis=0)
        convStep2_2 = tf.reduce_mean(convStep2_2, axis=0)
        pooling1 = pooling_layer(convStep2_1)
        pooling2 = pooling_layer(convStep2_2)
        #h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(pooling1,pooling2)
    return nodes11, children1,parent1,subl1,subr1,nodes22,children2,parent2,subl2,subr2,res
def init_net_finetuneOnlyOneParent(feature_size,embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None,), name='tree1')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        parent1=tf.placeholder(tf.int32, shape=(None, 2), name='parent1')
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
        parent2 = tf.placeholder(tf.int32, shape=(None, 2), name='parent2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        wn,wp=(
            tf.Variable(tf.truncated_normal([feature_size,conv_dim],stddev=std),name="Wn"),
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wp")
        )
        initnp=tf.truncated_normal([conv_dim,],stddev=math.sqrt(2.0/feature_size))
        b_convp=tf.Variable(initnp,name="b_convp")
        node_p1=tf.gather_nd(nodes1,parent1)
        node_p2=tf.gather_nd(nodes2,parent2)
        convtemp1=tf.nn.tanh(tf.tensordot(nodes1,wn,axes=1)+tf.tensordot(node_p1,wp,axes=1)+b_convp)
        convtemp2=tf.nn.tanh(tf.tensordot(nodes2,wn,axes=1)+tf.tensordot(node_p2,wp,axes=1)+b_convp)
        pooling1 = pooling_layer(convtemp1)
        pooling2 = pooling_layer(convtemp2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11, parent1,nodes22,parent2,res
def init_net_finetuneConsiderMore(feature_size,embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None,), name='tree1')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        children1 = tf.placeholder(tf.int32, shape=(1, None, None), name='children1')
        parent1=tf.placeholder(tf.int32, shape=(None, 2), name='parent1')
        subl1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl1')
        subr1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr1')
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
        children2 = tf.placeholder(tf.int32, shape=(1, None, None), name='children2')
        parent2 = tf.placeholder(tf.int32, shape=(None, 2), name='parent2')
        subl2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl2')
        subr2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        wn,wp=(
            tf.Variable(tf.truncated_normal([feature_size,conv_dim],stddev=std),name="Wn"),
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wp")
        )
        initnp=tf.truncated_normal([conv_dim,],stddev=math.sqrt(2.0/feature_size))
        b_convp=tf.Variable(initnp,name="b_convp")
        node_p1=tf.gather_nd(nodes1,parent1)
        node_p2=tf.gather_nd(nodes2,parent2)
        convtemp1=tf.nn.tanh(tf.tensordot(nodes1,wn,axes=1)+tf.tensordot(node_p1,wp,axes=1)+b_convp)
        convtemp2=tf.nn.tanh(tf.tensordot(nodes2,wn,axes=1)+tf.tensordot(node_p2,wp,axes=1)+b_convp)
        conv1,conv2,convl1,convl2,convr1,convr2=new_conv_layer(1,conv_dim,nodes1,children1,subl1,subr1,nodes2,children2,subl2,subr2,feature_size)
        conv1o=tf.concat([conv1,convl1,convr1,convtemp1],axis=0)
        conv2o = tf.concat([conv2,convl2,convr2,convtemp2],axis=0)
        w_one_by_one=tf.Variable(tf.truncated_normal([1,1,4],stddev=std),name="Wone")
        b_one_by_one=tf.Variable(0.1,name="Bone")
        convStep2_1=tf.tensordot(w_one_by_one,conv1o,axes=1)+b_one_by_one
        convStep2_2 = tf.tensordot(w_one_by_one, conv2o, axes=1) + b_one_by_one
        convStep2_1=tf.reduce_mean(convStep2_1,axis=0)
        convStep2_2 = tf.reduce_mean(convStep2_2, axis=0)
        pooling1 = pooling_layer(convStep2_1)
        pooling2 = pooling_layer(convStep2_2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11, children1,parent1,subl1,subr1,nodes22,children2,parent2,subl2,subr2,res
def init_net_finetuneonlySubling(feature_size,embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None,), name='tree1')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        subl1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl1')
        subr1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr1')
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
        subl2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl2')
        subr2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        convl1,convl2,convr1,convr2=new_conv_layer_subling(1,conv_dim,nodes1,subl1,subr1,nodes2,subl2,subr2,feature_size)
        conv1o=tf.concat([convl1,convr1],axis=0)
        conv2o = tf.concat([convl2,convr2],axis=0)
        w_one_by_one=tf.Variable(tf.truncated_normal([1,1,2],stddev=std),name="Wone")
        b_one_by_one=tf.Variable(0.1,name="Bone")
        convStep2_1=tf.tensordot(w_one_by_one,conv1o,axes=1)+b_one_by_one
        convStep2_2 = tf.tensordot(w_one_by_one, conv2o, axes=1) + b_one_by_one
        convStep2_1=tf.reduce_mean(convStep2_1,axis=0)
        convStep2_2 = tf.reduce_mean(convStep2_2, axis=0)
        pooling1 = pooling_layer(convStep2_1)
        pooling2 = pooling_layer(convStep2_2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11, subl1,subr1,nodes22,subl2,subr2,res
def init_net_finetuneConsiderMoreNewParent(feature_size,embeddingg,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes11 = tf.placeholder(tf.int32, shape=(None,), name='tree')
        nodes1 = tf.nn.embedding_lookup(embeddingg, nodes11)
        nodes1 = tf.expand_dims(nodes1, 0)
        children1 = tf.placeholder(tf.int32, shape=(1, None, None), name='children1')
        parent1=tf.placeholder(tf.int32, shape=(1,None, None), name='parent1')
        subl1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl1')
        subr1 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr1')
        nodes22 = tf.placeholder(tf.int32, shape=(None,), name='tree2')
        nodes2 = tf.nn.embedding_lookup(embeddingg, nodes22)
        nodes2 = tf.expand_dims(nodes2, 0)
        children2 = tf.placeholder(tf.int32, shape=(1, None, None), name='children2')
        parent2 = tf.placeholder(tf.int32, shape=(1,None, None), name='parent2')
        subl2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subl2')
        subr2 = tf.placeholder(tf.int32, shape=(1, None, None), name='subr2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        conv1,conv2,convp1,convp2,convl1,convl2,convr1,convr2=new_conv_layer_newp(conv_dim,nodes1,parent1,children1,subl1,subr1,nodes2,parent2,children2,subl2,subr2,feature_size)
        conv1o=tf.concat([conv1,convl1,convr1,convp1],axis=0)
        conv2o = tf.concat([conv2,convl2,convr2,convp2],axis=0)
        w_one_by_one=tf.Variable(tf.truncated_normal([1,1,4],stddev=std),name="Wone")
        b_one_by_one=tf.Variable(0.1,name="Bone")
        convStep2_1=tf.tensordot(w_one_by_one,conv1o,axes=1)+b_one_by_one
        convStep2_2 = tf.tensordot(w_one_by_one, conv2o, axes=1) + b_one_by_one
        convStep2_1=tf.reduce_mean(convStep2_1,axis=0)
        convStep2_2 = tf.reduce_mean(convStep2_2, axis=0)
        pooling1 = pooling_layer(convStep2_1)
        pooling2 = pooling_layer(convStep2_2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes11, children1,parent1,subl1,subr1,nodes22,children2,parent2,subl2,subr2,res

def init_net_nofinetuneConsiderMore(feature_size,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes1 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree1')
        children1 = tf.placeholder(tf.int32, shape=(None, None, None), name='children1')
        parent1=tf.placeholder(tf.int32, shape=(None, 2), name='parent1')
        subl1 = tf.placeholder(tf.int32, shape=(None, None, None), name='subl1')
        subr1 = tf.placeholder(tf.int32, shape=(None, None, None), name='subr1')
        nodes2 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree2')
        children2 = tf.placeholder(tf.int32, shape=(None, None, None), name='children2')
        parent2 = tf.placeholder(tf.int32, shape=(None, 2), name='parent2')
        subl2 = tf.placeholder(tf.int32, shape=(None, None, None), name='subl2')
        subr2 = tf.placeholder(tf.int32, shape=(None, None, None), name='subr2')
    with tf.name_scope('network'):
        std=1.0/math.sqrt(feature_size)
        wn,wp=(
            tf.Variable(tf.truncated_normal([feature_size,conv_dim],stddev=std),name="Wn"),
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wp")
        )
        initnp=tf.truncated_normal([conv_dim,],stddev=math.sqrt(2.0/feature_size))
        b_convp=tf.Variable(initnp,name="b_convp")
        node_p1=tf.gather_nd(nodes1,parent1)
        node_p2=tf.gather_nd(nodes2,parent2)
        convtemp1=tf.nn.tanh(tf.tensordot(nodes1,wn,axes=1)+tf.tensordot(node_p1,wp,axes=1)+b_convp)
        convtemp2=tf.nn.tanh(tf.tensordot(nodes2,wn,axes=1)+tf.tensordot(node_p2,wp,axes=1)+b_convp)
        conv1,conv2,convl1,convl2,convr1,convr2=new_conv_layer(1,conv_dim,nodes1,children1,subl1,subr1,nodes2,children2,subl2,subr2,feature_size)
        conv1o=tf.concat([conv1,convl1,convr1,convtemp1],axis=0)
        conv2o = tf.concat([conv2,convl2,convr2,convtemp2],axis=0)
        w_one_by_one=tf.Variable(tf.truncated_normal([1,1,4],stddev=std),name="Wone")
        b_one_by_one=tf.Variable(0.1,name="Bone")
        convStep2_1=tf.tensordot(w_one_by_one,conv1o,axes=1)+b_one_by_one
        convStep2_2 = tf.tensordot(w_one_by_one, conv2o, axes=1) + b_one_by_one
        convStep2_1=tf.reduce_mean(convStep2_1,axis=0)
        convStep2_2 = tf.reduce_mean(convStep2_2, axis=0)
        pooling1 = pooling_layer(convStep2_1)
        pooling2 = pooling_layer(convStep2_2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes1, children1,parent1,subl1,subr1,nodes2,children2,parent2,subl2,subr2,res
def init_net_nofinetune_one(feature_size,conv_dim):

    with tf.name_scope('inputs'):
        nodes1 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        nodes2 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
    with tf.name_scope('network'):
        std = 1.0 / math.sqrt(feature_size)
        init = tf.truncated_normal([conv_dim, ], stddev=math.sqrt(2.0 / feature_size))
        b_conv = tf.Variable(init, name="b_conv")
        w_conv = (
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wn"),
        )
        conv1 = tf.tensordot(nodes1, tf.reshape(w_conv, [feature_size, conv_dim]), axes=1) + b_conv
        pooling1 = pooling_layer(conv1)
        conv2 = tf.tensordot(nodes2, tf.reshape(w_conv, [feature_size, conv_dim]), axes=1) + b_conv
        pooling2 = pooling_layer(conv2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes1,nodes2,res
def init_net_nofinetune_one_nohidden(feature_size,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes1 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        nodes2 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
    with tf.name_scope('network'):
        std = 1.0 / math.sqrt(feature_size)
        init = tf.truncated_normal([conv_dim, ], stddev=math.sqrt(2.0 / feature_size))
        b_conv = tf.Variable(init, name="b_conv")
        w_conv = (
            tf.Variable(tf.truncated_normal([feature_size, conv_dim], stddev=std), name="Wn"),
        )
        conv1 = tf.tensordot(nodes1, tf.reshape(w_conv, [feature_size, conv_dim]), axes=1) + b_conv
        pooling1 = pooling_layer(conv1)
        conv2 = tf.tensordot(nodes2, tf.reshape(w_conv, [feature_size, conv_dim]), axes=1) + b_conv
        pooling2 = pooling_layer(conv2)
        #h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(pooling1,pooling2)
    return nodes1,nodes2,res
def init_net_nofinetune(feature_size,conv_dim):
    """Initialize an empty network."""

    with tf.name_scope('inputs'):
        nodes1 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        children1 = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
        nodes2 = tf.placeholder(tf.float32, shape=(None, None, feature_size), name='tree')
        children2 = tf.placeholder(tf.int32, shape=(None, None, None), name='children')
    with tf.name_scope('network'):
        conv1,conv2 = conv_layer(1,conv_dim, nodes1, children1, nodes2, children2, feature_size)
        pooling1 = pooling_layer(conv1)
        pooling2 = pooling_layer(conv2)
        h1,h2=hidden_layer(pooling1,pooling2,conv_dim,50)
        res=computCos(h1,h2)
    return nodes1, children1,nodes2,children2,res
def conv_layer(num_conv, output_size, nodes1, children1, nodes2, children2,feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes11,nodes22=conv_node(nodes1, children1, nodes2, children2,feature_size, output_size)
        nodes1 = [
            nodes11
            for _ in range(num_conv)
        ]
        nodes2 = [
            nodes22
            for _ in range(num_conv)
        ]
        return tf.concat(nodes1, axis=2),tf.concat(nodes2, axis=2)
def new_conv_layer_subling(num_conv, output_size, nodes1, subl1,subr1, nodes2, subl2,subr2,feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodessubl1,nodessubl2,nodessubr1,nodessubr2=new_conv_node_subling(nodes1,subl1,subr1, nodes2,subl2,subr2,feature_size, output_size)
        # nodes1 = [
        #     nodes11
        #     for _ in range(num_conv)
        # ]
        # nodes2 = [
        #     nodes22
        #     for _ in range(num_conv)
        # ]
        return nodessubl1,nodessubl2,nodessubr1,nodessubr2
def new_conv_layer(num_conv, output_size, nodes1, children1,subl1,subr1, nodes2, children2,subl2,subr2,feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes1,nodes2,nodessubl1,nodessubl2,nodessubr1,nodessubr2=new_conv_node(nodes1, children1,subl1,subr1, nodes2, children2,subl2,subr2,feature_size, output_size)
        # nodes1 = [
        #     nodes11
        #     for _ in range(num_conv)
        # ]
        # nodes2 = [
        #     nodes22
        #     for _ in range(num_conv)
        # ]
        return nodes1,nodes2,nodessubl1,nodessubl2,nodessubr1,nodessubr2
        #return tf.concat(nodes1, axis=2),tf.concat(nodes2, axis=2)
def new_conv_layer_newp( output_size, nodes1, parent1,children1,subl1,subr1, nodes2, parent2,children2,subl2,subr2,feature_size):
    """Creates a convolution layer with num_conv convolutions merged together at
    the output. Final output will be a tensor with shape
    [batch_size, num_nodes, output_size * num_conv]"""

    with tf.name_scope('conv_layer'):
        nodes1,nodes2,nodesp1,nodesp2,nodessubl1,nodessubl2,nodessubr1,nodessubr2=new_conv_node_newp(nodes1, parent1,children1,subl1,subr1, nodes2, parent2,children2,subl2,subr2,feature_size, output_size)
        # nodes1 = [
        #     nodes11
        #     for _ in range(num_conv)
        # ]
        # nodes2 = [
        #     nodes22
        #     for _ in range(num_conv)
        # ]
        return nodes1,nodes2,nodesp1,nodesp2,nodessubl1,nodessubl2,nodessubr1,nodessubr2
        #return tf.concat(nodes1, axis=2),tf.concat(nodes2, axis=2)
def conv_node(nodes1, children1, nodes2, children2,feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/feature_size))
        #init = tf.zeros([output_size,])
        b_conv = tf.Variable(init, name='b_conv')
        return conv_step(nodes1, children1, feature_size, w_t, w_r, w_l, b_conv),conv_step(nodes2, children2, feature_size, w_t, w_r, w_l, b_conv)
def new_conv_node(nodes1, children1,subl1,subr1, nodes2, children2,subl2,subr2,feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/feature_size))
        #init = tf.zeros([output_size,])
        b_conv = tf.Variable(init, name='b_conv')


        w_tsubl, w_subll, w_sublr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subll'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_sublr'),
        )
        initsubl = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        b_convsubl = tf.Variable(initsubl, name='b_convsubl')

        w_tsubr, w_subrl, w_subrr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubr'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrr'),
        )
        initsubr = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        # init = tf.zeros([output_size,])
        b_convsubr = tf.Variable(initsubr, name='b_convsubr')
        return conv_step(nodes1, children1, feature_size, w_t, w_r, w_l, b_conv),conv_step(nodes2, children2, feature_size, w_t, w_r, w_l, b_conv),\
               conv_step(nodes1, subl1, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),conv_step(nodes2, subl2, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),\
               conv_step(nodes1, subr1, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr),conv_step(nodes2, subr2, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr)
def new_conv_node_subling(nodes1,subl1,subr1, nodes2,subl2,subr2,feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)


        w_tsubl, w_subll, w_sublr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subll'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_sublr'),
        )
        initsubl = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        b_convsubl = tf.Variable(initsubl, name='b_convsubl')

        w_tsubr, w_subrl, w_subrr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubr'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrr'),
        )
        initsubr = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        # init = tf.zeros([output_size,])
        b_convsubr = tf.Variable(initsubr, name='b_convsubr')
        return       conv_step(nodes1, subl1, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),conv_step(nodes2, subl2, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),\
               conv_step(nodes1, subr1, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr),conv_step(nodes2, subr2, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr)

def new_conv_node_newp(nodes1,parent1, children1,subl1,subr1, nodes2,parent2, children2,subl2,subr2,feature_size, output_size):
    """Perform convolutions over every batch sample."""
    with tf.name_scope('conv_node'):
        std = 1.0 / math.sqrt(feature_size)
        w_t, w_l, w_r = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr'),
        )
        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/feature_size))
        #init = tf.zeros([output_size,])
        b_conv = tf.Variable(init, name='b_conv')
        w_t_p, w_l_p, w_r_p = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wt_p'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wl_p'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='Wr_p'),
        )
        init_p = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        # init = tf.zeros([output_size,])
        b_conv_p = tf.Variable(init_p, name='b_conv_p')

        w_tsubl, w_subll, w_sublr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subll'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_sublr'),
        )
        initsubl = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        b_convsubl = tf.Variable(initsubl, name='b_convsubl')

        w_tsubr, w_subrl, w_subrr = (
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_tsubr'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrl'),
            tf.Variable(tf.truncated_normal([feature_size, output_size], stddev=std), name='w_subrr'),
        )
        initsubr = tf.truncated_normal([output_size, ], stddev=math.sqrt(2.0 / feature_size))
        # init = tf.zeros([output_size,])
        b_convsubr = tf.Variable(initsubr, name='b_convsubr')
        return conv_step(nodes1, children1, feature_size, w_t, w_r, w_l, b_conv),conv_step(nodes2, children2, feature_size, w_t, w_r, w_l, b_conv), \
               conv_step(nodes1, parent1, feature_size, w_t_p, w_r_p, w_l_p, b_conv_p), conv_step(nodes2, parent2,feature_size, w_t_p, w_r_p, w_l_p, b_conv_p), \
               conv_step(nodes1, subl1, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),conv_step(nodes2, subl2, feature_size, w_tsubl, w_subll, w_sublr, b_convsubl),\
               conv_step(nodes1, subr1, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr),conv_step(nodes2, subr2, feature_size, w_tsubr, w_subrl, w_subrr, b_convsubr)



def children_tensor(nodes, children, feature_size):
    """Build the children tensor from the input nodes and child lookup."""
    with tf.name_scope('children_tensor'):
        max_children = tf.shape(children)[2]
        batch_size = tf.shape(nodes)[0]
        num_nodes = tf.shape(nodes)[1]
        # replace the root node with the zero vector so lookups for the 0th
        # vector return 0 instead of the root vector
        # zero_vecs is (batch_size, num_nodes, 1)
        zero_vecs = tf.zeros((batch_size, 1, feature_size))
        # vector_lookup is (batch_size x num_nodes x feature_size)
        vector_lookup = tf.concat([zero_vecs, nodes[:, 1:, :]], axis=1)
        # children is (batch_size x num_nodes x num_children x 1)
        children = tf.expand_dims(children, axis=3)
        # prepend the batch indices to the 4th dimension of children
        # batch_indices is (batch_size x 1 x 1 x 1)
        batch_indices = tf.reshape(tf.range(0, batch_size), (batch_size, 1, 1, 1))
        # batch_indices is (batch_size x num_nodes x num_children x 1)
        batch_indices = tf.tile(batch_indices, [1, num_nodes, max_children, 1])
        # children is (batch_size x num_nodes x num_children x 2)
        children = tf.concat([batch_indices, children], axis=3)
        # output will have shape (batch_size x num_nodes x num_children x feature_size)
        # NOTE: tf < 1.1 contains a bug that makes backprop not work for this!
        return tf.gather_nd(vector_lookup, children, name='children')

def eta_t(children):
    """Compute weight matrix for how much each vector belongs to the 'top'"""
    with tf.name_scope('coef_t'):
        # children is shape (batch_size x max_tree_size x max_children)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]
        # eta_t is shape (batch_size x max_tree_size x max_children + 1)
        return tf.tile(tf.expand_dims(tf.concat(
            [tf.ones((max_tree_size, 1)), tf.zeros((max_tree_size, max_children))],
            axis=1), axis=0,
        ), [batch_size, 1, 1], name='coef_t')

def eta_r(children, t_coef):
    """Compute weight matrix for how much each vector belogs to the 'right'"""
    with tf.name_scope('coef_r'):
        # children is shape (batch_size x max_tree_size x max_children)
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        max_children = tf.shape(children)[2]

        # num_siblings is shape (batch_size x max_tree_size x 1)
        num_siblings = tf.cast(
            tf.count_nonzero(children, axis=2, keep_dims=True),
            dtype=tf.float32
        )
        # num_siblings is shape (batch_size x max_tree_size x max_children + 1)
        num_siblings = tf.tile(
            num_siblings, [1, 1, max_children + 1], name='num_siblings'
        )
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2, name='mask'
        )

        # child indices for every tree (batch_size x max_tree_size x max_children + 1)
        child_indices = tf.multiply(tf.tile(
            tf.expand_dims(
                tf.expand_dims(
                    tf.range(-1.0, tf.cast(max_children, tf.float32), 1.0, dtype=tf.float32),
                    axis=0
                ),
                axis=0
            ),
            [batch_size, max_tree_size, 1]
        ), mask, name='child_indices')

        # weights for every tree node in the case that num_siblings = 0
        # shape is (batch_size x max_tree_size x max_children + 1)
        singles = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
             tf.fill((batch_size, max_tree_size, 1), 0.5),
             tf.zeros((batch_size, max_tree_size, max_children - 1))],
            axis=2, name='singles')

        # eta_r is shape (batch_size x max_tree_size x max_children + 1)
        return tf.where(
            tf.equal(num_siblings, 1.0),
            # avoid division by 0 when num_siblings == 1
            singles,
            # the normal case where num_siblings != 1
            tf.multiply((1.0 - t_coef), tf.divide(child_indices, num_siblings - 1.0)),
            name='coef_r'
        )

def eta_l(children, coef_t, coef_r):
    """Compute weight matrix for how much each vector belongs to the 'left'"""
    with tf.name_scope('coef_l'):
        children = tf.cast(children, tf.float32)
        batch_size = tf.shape(children)[0]
        max_tree_size = tf.shape(children)[1]
        # creates a mask of 1's and 0's where 1 means there is a child there
        # has shape (batch_size x max_tree_size x max_children + 1)
        mask = tf.concat(
            [tf.zeros((batch_size, max_tree_size, 1)),
                tf.minimum(children, tf.ones(tf.shape(children)))],
            axis=2,
            name='mask'
        )

        # eta_l is shape (batch_size x max_tree_size x max_children + 1)
        return tf.multiply(
            tf.multiply((1.0 - coef_t), (1.0 - coef_r)), mask, name='coef_l'
        )

def conv_step(nodes, children, feature_size, w_t, w_r, w_l, b_conv):
    """Convolve a batch of nodes and children.
    Lots of high dimensional tensors in this function. Intuitively it makes
    more sense if we did this work with while loops, but computationally this
    is more efficient. Don't try to wrap your head around all the tensor dot
    products, just follow the trail of dimensions.
    """
    with tf.name_scope('conv_step'):
        # nodes is shape (batch_size x max_tree_size x feature_size)
        # children is shape (batch_size x max_tree_size x max_children)

        with tf.name_scope('trees'):
            # children_vectors will have shape
            # (batch_size x max_tree_size x max_children x feature_size)
            children_vectors = children_tensor(nodes, children, feature_size)

            # add a 4th dimension to the nodes tensor
            nodes = tf.expand_dims(nodes, axis=2)
            # tree_tensor is shape
            # (batch_size x max_tree_size x max_children + 1 x feature_size)
            tree_tensor = tf.concat([nodes, children_vectors], axis=2, name='trees')

        with tf.name_scope('coefficients'):
            # coefficient tensors are shape (batch_size x max_tree_size x max_children + 1)
            c_t = eta_t(children)
            c_r = eta_r(children, c_t)
            c_l = eta_l(children, c_t, c_r)

            # concatenate the position coefficients into a tensor
            # (batch_size x max_tree_size x max_children + 1 x 3)
            coef = tf.stack([c_t, c_r, c_l], axis=3, name='coef')

        with tf.name_scope('weights'):
            # stack weight matrices on top to make a weight tensor
            # (3, feature_size, output_size)
            weights = tf.stack([w_t, w_r, w_l], axis=0)

        with tf.name_scope('combine'):
            batch_size = tf.shape(children)[0]
            max_tree_size = tf.shape(children)[1]
            max_children = tf.shape(children)[2]

            # reshape for matrix multiplication
            x = batch_size * max_tree_size
            y = max_children + 1
            result = tf.reshape(tree_tensor, (x, y, feature_size))
            coef = tf.reshape(coef, (x, y, 3))
            result = tf.matmul(result, coef, transpose_a=True)
            result = tf.reshape(result, (batch_size, max_tree_size, 3, feature_size))

            result = tf.tensordot(result, weights, [[2, 3], [0, 1]])

            # output is (batch_size, max_tree_size, output_size)
            return tf.nn.tanh(result + b_conv, name='conv')

def pooling_layer(nodes):
    """Creates a max dynamic pooling layer from the nodes."""
    with tf.name_scope("pooling"):
        pooled = tf.reduce_max(nodes, axis=1)
        return pooled
def computCos(pool1,pool2):
    pool1_norm = tf.sqrt(tf.reduce_sum(tf.square(pool1), axis=1))
    pool2_norm = tf.sqrt(tf.reduce_sum(tf.square(pool2), axis=1))
    pool1_pool2 = tf.reduce_sum(tf.multiply(pool1, pool2), axis=1)
    cosin = pool1_pool2 / (pool1_norm * pool2_norm)
    return cosin
def hidden_layer(pooled1, pooled2,input_size, output_size):
    """Create a hidden feedforward layer."""
    with tf.name_scope("hidden"):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/input_size))
        #init = tf.zeros([output_size,])
        biases = tf.Variable(init, name='biases')
        return tf.nn.tanh(tf.matmul(pooled1, weights) + biases),tf.nn.tanh(tf.matmul(pooled2, weights) + biases)

def hidden_layer1(pooled1,input_size, output_size):
    """Create a hidden feedforward layer."""
    with tf.name_scope("hidden"):
        weights = tf.Variable(
            tf.truncated_normal(
                [input_size, output_size], stddev=1.0 / math.sqrt(input_size)
            ),
            name='weights'
        )

        init = tf.truncated_normal([output_size,], stddev=math.sqrt(2.0/input_size))
        #init = tf.zeros([output_size,])
        biases = tf.Variable(init, name='biases')
        return tf.nn.tanh(tf.matmul(pooled1, weights) + biases)

def loss_layer(logits_node):
    """Create a loss layer for training."""

    labels = tf.placeholder(tf.float32, (1,))
    with tf.name_scope('loss_layer'):
        loss=tf.square(logits_node[0]-labels[0])
        return labels, loss

def out_layer(logits_node):
    """Apply softmax to the output layer."""
    with tf.name_scope('output'):
        return tf.nn.softmax(logits_node)