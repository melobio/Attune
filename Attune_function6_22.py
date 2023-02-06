import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import os
import scanpy as sc
import argparse
import pandas as pd
import copy
from collections import Counter
from scipy.sparse import issparse
import scipy
import sys
from bgi.utils.data_utils import *
from bgi.models.DeepSingleCell_Attune import *
from bgi.metrics.clustering_metrics import *
from bgi.losses.contrastive_loss import simclr_loss, simclr_loss_1,simcse_loss
from bgi.losses.MISA_loss import *
import re
import h5py
import time
from sklearn.metrics import mean_squared_error
import yaml

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    #return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(value)))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(value)))

def set_seeds(seed=10):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def preprocessing_rna(
        adata,
        min_features: int = 600,
        min_cells: int = 3,
        target_sum: int = 10000,
        n_top_features=2000,  # or gene list
        chunk_size: int = 20000,
        is_hvg = False,
        batch_key = 'batch',
        log=True
):
    if min_features is None: min_features = 600
    if n_top_features is None: n_top_features = 40000

    if not issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)

    # adata = adata[:, [gene for gene in adata.var_names
    #                   if not str(gene).startswith(tuple(['ERCC', 'MT-', 'mt-']))]]

    #sc.pp.filter_cells(adata, min_genes=min_features)

    #sc.pp.filter_genes(adata, min_cells=min_cells)

    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)
    if is_hvg == True:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_features, batch_key=batch_key, inplace=False, subset=True)

    print('Processed dataset shape: {}'.format(adata.shape))
    return adata

def serialize_example_batch(x_feature, x_weight, y_batch,x_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def create_tfrecord(source_file,  batch_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    counter = 0
    batch_examples = {}
    for x, batch,k in zip(x_data, batch_number,obs_name_list):
        if zero_filter is False:
            #x = x + 10e-6
            indexes = np.where(x >= 0.0)
        else:
            indexes = np.where(x > 0.0)
        values = x[indexes]

        features = np.array(indexes)
        features = np.reshape(features, (features.shape[1]))
        values = np.array(values, dtype=np.float)
        # values = values / np.linalg.norm(values)

        if batch not in batch_examples:
            batch_examples[batch] = []

        example = serialize_example_batch(features, values, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 1000 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            #print(x)
            #print(values)
            #print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    save_dict = {'vocab size': len(features)}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)
#     np.savez_compressed('vocab_size.npz', **save_dict)

#  输入模块
def concerto_input(input_file_list):
    query_adata = sc.read(input_file_list)
    return query_adata

# 预处理模块
def concerto_preprocess(query_adata):
    sc.pp.normalize_total(query_adata, target_sum=10000)
    sc.pp.log1p(query_adata)
    return query_adata

# 交集基因
def concerto_intersect_gene(ref_adata, query_adata, parameters=None):
    ref_var_list = ref_adata.var_names.tolist()
    query_var_list = query_adata.var_names.tolist()
    intersect_gene_list = list(set(ref_var_list).intersection(set(query_var_list)))
    intersect_stats_A_B = len(list(set(ref_var_list).difference(set(query_var_list))))# ref中有query中无的个数
    intersect_stats_B_A = len(list(set(query_var_list).difference(set(ref_var_list))))  # ref中有query中无的个数
    intersect_stats = [intersect_stats_A_B,len(intersect_gene_list),intersect_stats_B_A]
    return intersect_gene_list, intersect_stats # list, [int, int, int]([A-B, A交B, B-A])

# HVG
def concerto_HVG(ref_adata,query_adata,n_top_genes=None, min_disp=0.5, min_mean=0.0125, max_mean=3,
	parameters=None):

    sc.pp.highly_variable_genes(query_adata, n_top_genes=n_top_genes, min_disp=0.5,min_mean=0.0125, max_mean=3)
    sc.pp.highly_variable_genes(ref_adata, n_top_genes=n_top_genes, min_disp=0.5, min_mean=0.0125, max_mean=3)
    ref_adata = ref_adata[:,ref_adata.var.highly_variable]
    query_adata = query_adata[:,query_adata.var.highly_variable]
    HVG_list = list(set(ref_adata.var_names.tolist()).intersection(set(query_adata.var_names.tolist())))
    processed_query_adata = query_adata[:,HVG_list]
    processed_ref_adata = ref_adata[:, HVG_list]
    return processed_ref_adata, processed_query_adata, HVG_list

# 如果不训新模型，补全到Ref的基因个数
def concerto_padding(ref_gene_list_path:str, ref_weight_path:str, query_adata):
    # 检验 ref gene list和 weight size 一致
    f = h5py.File(ref_weight_path, 'r')  # 打开h5文件
    if 'RNA-Embedding/embeddings:0' in f['RNA-Embedding']:
        weight_size = f['RNA-Embedding']['RNA-Embedding/embeddings:0'].value.shape[0]
        print('unsup model')
    else:
        weight_size = f['RNA-Embedding']['RNA-Embedding_1/embeddings:0'].value.shape[0]
        print('sup model')
    gene_names = list(pd.read_csv(ref_gene_list_path)['0'].values)
    if weight_size == len(gene_names):
        query_gene_list = query_adata.var_names.tolist()
        gene_inter_list = list(set(gene_names).intersection(set(query_gene_list)))
        empty_matrix = np.zeros([len(query_adata.obs_names),len(gene_names)])
        inter_index = []
        inter_index_query = []
        for i in gene_inter_list:
            inter_index.append(gene_names.index(i))
            inter_index_query.append(query_gene_list.index(i))
        query_X = query_adata.X.toarray()
        query_X_inter = query_X[:, inter_index_query]
        for j in range(query_X_inter.shape[1]):
            empty_matrix[:, inter_index[j]] = query_X_inter[:, j]
        q = sc.AnnData(empty_matrix)
        q.obs = query_adata.obs.copy()
        q.var_names = gene_names
        return q
    else:
        return print('weight size is different from ref gene list')
def concerto_padding2(ref_gene_list_path:str, ref_weight_path:str, query_adata):
    gene_names = list(pd.read_csv(ref_gene_list_path)['0'].values)
    query_gene_list = query_adata.var_names.tolist()
    gene_inter_list = list(set(gene_names).intersection(set(query_gene_list)))
    empty_matrix = np.zeros([len(query_adata.obs_names),len(gene_names)])
    inter_index = []
    inter_index_query = []
    for i in gene_inter_list:
        inter_index.append(gene_names.index(i))
        inter_index_query.append(query_gene_list.index(i))
    query_X = query_adata.X.toarray()
    query_X_inter = query_X[:, inter_index_query]
    for j in range(query_X_inter.shape[1]):
        empty_matrix[:, inter_index[j]] = query_X_inter[:, j]
    q = sc.AnnData(empty_matrix)
    q.obs = query_adata.obs.copy()
    q.var_names = gene_names
    return q


# 造tfrecords
def concerto_make_tfrecord(processed_ref_adata, tf_path, batch_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    if batch_col_name is None:
        batch_col_name = 'batch_'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name]  = ['0']*sample_num
    print(processed_ref_adata)
    batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
    cc = dict(Counter(batch_list))
    cc = list(cc.keys())
    tfrecord_file = tf_path + '/tf.tfrecord'
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    create_tfrecord(processed_ref_adata, cc, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name)

    return tf_path


# ---------- make supervised tfr --------------
def serialize_example_batch_supervised(x_feature, x_weight, y_label,y_batch,x_id):

    feature = {
        'feature': _int64_feature(x_feature),
        'value': _float_feature(x_weight),
        'label': _int64_feature(y_label),
        'batch': _int64_feature(y_batch),
        'id': _bytes_feature(x_id)
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def create_tfrecord_supervised(source_file,  batch_dict,label_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch',label_key = 'label'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    label_data = source_file.obs[label_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_number = []
    label_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    for j in range(len(label_data)):
        cell_type = label_data[j]
        place = label_dict.index(cell_type)
        label_number.append(place)

    counter = 0
    batch_examples = {}
    for x, y,batch,k in zip(x_data, label_number,batch_number,obs_name_list):
        if zero_filter is False:
            #x = x + 10e-6
            indexes = np.where(x >= 0.0)
        else:
            indexes = np.where(x > 0.0)
        values = x[indexes]

        features = np.array(indexes)
        features = np.reshape(features, (features.shape[1]))
        values = np.array(values, dtype=np.float)
        # values = values / np.linalg.norm(values)

        if batch not in batch_examples:
            batch_examples[batch] = []
        y = np.array([int(y)])
        example = serialize_example_batch_supervised(features, values, y, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 100 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            print(x)
            print(values)
            print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    #save_dict = {'vocab size': len(features)}
    save_dict = {'vocab size': len(features),'classes number':len(label_dict),'label_dict':label_dict,'batch_dict':batch_dict}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)

# 造tfrecords
def concerto_make_tfrecord_supervised(processed_ref_adata, tf_path,save_dict = None, batch_col_name=None,label_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    tfrecord_file = os.path.join(tf_path, 'tf.tfrecord')
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    if batch_col_name is None:
        batch_col_name = 'batch'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name] = ['0'] * sample_num
    if label_col_name is None:
        label_col_name = 'label'
    if save_dict is not None:
        f = np.load(os.path.join(save_dict,'vocab_size.npz')) # load saved dict path
        cc_ = list(f['label_dict'])
        cc = list(f['batch_dict'])
    else:
        batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
        cc = dict(Counter(batch_list))
        cc = list(cc.keys())
        label_list = processed_ref_adata.obs[label_col_name].unique().tolist()
        cc_ = dict(Counter(label_list))
        cc_ = list(cc_.keys())
    create_tfrecord_supervised(processed_ref_adata, cc,cc_, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name,label_key=label_col_name)

    return tf_path



def create_tfrecord_supervised_1batch(source_file, batch_dict,label_dict, tfrecord_file, zero_filter=False, norm=False, batch_key = 'batch',label_key = 'label'):
    if type(source_file.X) != np.ndarray:
        x_data = source_file.X.toarray()
    else:
        x_data = source_file.X
    batch_data = source_file.obs[batch_key].tolist()
    label_data = source_file.obs[label_key].tolist()
    obs_name_list = source_file.obs_names.tolist()
    batch_name = batch_dict[0]
    batch_number = []
    label_number = []
    for j in range(len(batch_data)):
        batch = batch_data[j]
        place = batch_dict.index(batch)
        batch_number.append(place)

    for j in range(len(label_data)):
        cell_type = label_data[j]
        place = label_dict.index(cell_type)
        label_number.append(place)

    counter = 0
    batch_examples = {}
    for x, y,batch,k in zip(x_data, label_number,batch_number,obs_name_list):
        if zero_filter is False:
            x = x + 10e-6
            indexes = np.where(x >= 0.0)
        else:
            indexes = np.where(x > 0.0)
        values = x[indexes]

        features = np.array(indexes)
        features = np.reshape(features, (features.shape[1]))
        values = np.array(values, dtype=np.float)
        # values = values / np.linalg.norm(values)

        if batch not in batch_examples:
            batch_examples[batch] = []
        y = np.array([int(y)])
        example = serialize_example_batch_supervised(features, values, y, np.array([int(batch)]),k)
        batch_examples[batch].append(example)

        counter += 1
        if counter % 100 == 0:
            print('counter: {} shape: {}, batch: {}'.format(counter, features.shape, batch))

            print(x)
            print(values)
            print("batchs: ", batch_dict)

    for item in batch_examples.items():
        batch = item[0]
        examples = item[1]
        if zero_filter is False:
            file = tfrecord_file.replace('.tfrecord', '_{}.tfrecord'.format(batch_name))
        else:
            if norm is False:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero_no_norm.tfrecord'.format(batch_name))
            else:
                file = tfrecord_file.replace('.tfrecord', '_{}_no_zero.tfrecord'.format(batch_name))
        with tf.io.TFRecordWriter(file) as writer:
            for example in examples:
                writer.write(example)
    #save_dict = {'vocab size': len(features)}
    save_dict = {'vocab size': len(features),'classes number':len(label_dict),'label_dict':label_dict,'batch_dict':batch_dict}
    file = tfrecord_file.replace('tf.tfrecord','vocab_size.npz')
    np.savez_compressed(file, **save_dict)



# 造tfrecords
def concerto_make_tfrecord_supervised_1batch(processed_ref_adata, tf_path, save_dict = None, batch_col_name=None,label_col_name=None):
    # 有输入batch_col_name的时候，用这列作为batchid， 若无假设所有是一个batch
	# 不做乱序,
    tfrecord_file = os.path.join(tf_path, 'tf.tfrecord')
    if not os.path.exists(tf_path):
        os.makedirs(tf_path)
    if batch_col_name is None:
        batch_col_name = 'batch'
        sample_num = len(processed_ref_adata.obs_names.tolist())
        processed_ref_adata.obs[batch_col_name] = ['0'] * sample_num
    if label_col_name is None:
        label_col_name = 'label'
    if save_dict is not None:
        f = np.load(os.path.join(save_dict,'vocab_size.npz')) # load saved dict path
        cc_ = list(f['label_dict'])
        cc = list(f['batch_dict'])

    else:
        batch_list = processed_ref_adata.obs[batch_col_name].unique().tolist()
        cc = dict(Counter(batch_list))
        cc = list(cc.keys())
        label_list = processed_ref_adata.obs[label_col_name].unique().tolist()
        cc_ = dict(Counter(label_list))
        cc_ = list(cc_.keys())

    create_tfrecord_supervised_1batch(processed_ref_adata, cc,cc_, tfrecord_file, zero_filter=False, norm=True, batch_key =batch_col_name,label_key=label_col_name)

    return tf_path


# train unsupervised
def concerto_train_ref(ref_tf_path:str, weight_path:str, super_parameters=None):
    set_seeds(0)
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch':3,'lr':1e-5}
#     dirname = os.getcwd()
#     f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(ref_tf_path,'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
#     tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
        encode_network.save_weights(
            weight_path + 'weight_encoder_epoch{}.h5'.format(str(epoch+1)))
        decode_network.save_weights(
            weight_path + 'weight_decoder_epoch{}.h5'.format(str(epoch+1)))

    return weight_path


def concerto_test_ref(model_path: str, ref_tf_path: str, super_parameters=None, saved_weight_path=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}

    f = np.load(ref_tf_path + 'vocab_size.npz')
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    if saved_weight_path is None:
        weight_id_list = []
        weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]
        for id in weight_list:
            id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
            weight_id_list.append(int(id_1[0]))
            encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                        by_name=True)

    else:
        encode_network.load_weights(saved_weight_path, by_name=True)
        print('load saved weight')

    source_data_batch = []
    source_data_feature = []
    source_data_id = []
    batch_size = super_parameters['batch_size']
    for file in train_source_list:
        print(file)
        train_size = 0
        ref_db = create_classifier_dataset_multi(
            [file],
            batch_size=batch_size,
            is_training=False,
            data_augment=False,
            shuffle_size=10000)
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            train_size += len(target_id)
            if step == 0:
                output = encode_network([target_features, target_values], training=False)

        dim = output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_batch_1 = np.zeros((train_size))
        # source_id_batch_1 = np.zeros((train_size))
        source_id_batch_1 = []
        all_samples = 0
        for step, (target_features, target_values, target_batch, target_id) in enumerate(ref_db):
            output = encode_network([target_features, target_values], training=False)
            output = tf.nn.l2_normalize(output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(target_id), :] = output
            source_data_batch_1[all_samples:all_samples + len(target_id)] = target_batch
            # source_id_batch_1[all_samples:all_samples + len(target_id)] = target_id.numpy().decode("utf-8")
            source_id_batch_1.extend(list(target_id.numpy().astype('U')))

            all_samples += len(target_id)
        source_data_feature.extend(source_data_feature_1)
        source_data_batch.extend(source_data_batch_1)
        source_data_id.extend(source_id_batch_1)

    ref_embedding = np.array(source_data_feature)
    print('reference embedding shape', ref_embedding.shape)
    print('ref id length', len(source_data_id))
    return ref_embedding, source_data_id



# train supervised
# train
def concerto_train_ref_supervised(ref_tf_path:str, weight_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch_pretrain':1,'epoch_classifier':5,'lr':1e-5,}
    # dirname = os.getcwd()
    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    for epoch in range(super_parameters['epoch_classifier']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    outputs = cls_network([source_features, source_values], training=True)
                    classifer_loss = cls_loss_object(source_label, outputs)
                    source_pred = outputs
                    train_cls_accuracy(source_label, source_pred)
                    train_loss(classifer_loss)

                variables = [cls_network.trainable_variables]
                grads = tape.gradient(classifer_loss, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, train cls loss: {:0.4f}, train acc: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          train_cls_accuracy.result(),
                                          ))
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch+1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch+1))))

    return weight_path

# train sup 0112
def concerto_train_ref_supervised_yzs(ref_tf_path:str, weight_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size':32,'epoch_pretrain':1,'epoch_classifier':5,'lr':1e-5, 'drop_rate': 0.1}
#     dirname = os.getcwd()
    f = np.load(ref_tf_path + '/vocab_size.npz')
    
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
#     tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2,temperature = 0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    for epoch in range(super_parameters['epoch_classifier']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi_supervised([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_label, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    outputs = cls_network([source_features, source_values], training=True)
                    classifer_loss = cls_loss_object(source_label, outputs)
                    source_pred = outputs
                    train_cls_accuracy(source_label, source_pred)
                    train_loss(classifer_loss)

                variables = [cls_network.trainable_variables]
                grads = tape.gradient(classifer_loss, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, train cls loss: {:0.4f}, train acc: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          train_cls_accuracy.result(),
                                          ))
        encode_network.save_weights(
            weight_path + 'weight_encoder_epoch{}.h5'.format(str(epoch+1)))
        decode_network.save_weights(
            weight_path + 'weight_decoder_epoch{}.h5'.format(str(epoch+1)))
        cls_network.save_weights(os.path.join(weight_path, 'weight_cls_epoch{}.h5'.format(str(epoch+1))))
        

    return weight_path

# test
def concerto_test_1set_attention_supervised(model_path: str, ref_tf_path: str, super_parameters=None, n_cells_for_ref=5000):
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch': 1, 'lr': 1e-5,'drop_rate': 0.1}

    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    num_classes = int(f['classes number'])
    label_dict = f['label_dict']
    batch_dict = f['batch_dict']
    batch_size = super_parameters['batch_size']
    encode_network = multi_embedding_attention_transfer(
        multi_max_features=[vocab_size],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = [os.path.join(ref_tf_path, i) for i in tf_list_1]
    # choose last epoch as test model
    weight_id_list = []
    # weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and f.startswith('weight') )]
    weight_list = [f for f in os.listdir(model_path) if (f.endswith('h5') and ('cls' in f))]  # yyyx 1214
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h5', id)  # f1
        weight_id_list.append(int(id_1[0]))
    weight_name_ = sorted(list(zip(weight_id_list, weight_list)), key=lambda x: x[0])[-1][1]
    output = encode_network.layers[-1].output
    output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network.input, outputs=output)
    cls_network.load_weights(os.path.join(model_path, weight_name_))

    t1 = time.time()
    ref_db = create_classifier_dataset_multi_supervised(
        train_source_list,
        batch_size=batch_size,  # maybe slow
        is_training=False,
        data_augment=False,
        shuffle_size=10000)

    t2 = time.time()
    print('load all tf in memory time(s)', t2 - t1)  # time consumption is huge this step!!!!

    feature_len = n_cells_for_ref // batch_size * batch_size
    print(feature_len, batch_size)
    t2 = time.time()
    source_data_batch_1 = np.zeros((feature_len))
    source_data_label_1 = np.zeros((feature_len))
    source_data_pred_1 = np.zeros((feature_len))
    source_id_1 = []
    source_id_label_1 = []
    source_id_batch_1 = []
    source_id_pred_1 = []
    for step, (target_features, target_values,target_label, target_batch, target_id) in enumerate(ref_db):
        if step * batch_size >= feature_len:
            break
        preds = cls_network([target_features, target_values], training=False)
        preds_1 = np.argmax(preds, axis=1)
        source_data_pred_1[step * batch_size:(step + 1) * batch_size] = preds_1
        source_data_batch_1[step * batch_size:(step + 1) * batch_size] = target_batch
        source_data_label_1[step * batch_size:(step + 1) * batch_size] = target_label
        source_id_1.extend(list(target_id.numpy().astype('U')))

    t3 = time.time()
    print('test time', t3 - t2)
    print('source_id len', len(source_id_1))
    for j in source_data_label_1:
        source_id_label_1.append(label_dict[int(j)])
    for j in source_data_pred_1:
        source_id_pred_1.append(label_dict[int(j)])
    for j in source_data_batch_1:
        source_id_batch_1.append(batch_dict[int(j)])

    acc = accuracy_score(source_data_label_1, source_data_pred_1)
    f1_scores_median = f1_score(source_data_label_1, source_data_pred_1, average=None)
    f1_scores_median = np.median(f1_scores_median)
    f1_scores_macro = f1_score(source_data_label_1, source_data_pred_1, average='macro')
    f1_scores_micro = f1_score(source_data_label_1, source_data_pred_1, average='micro')
    print('acc:', acc, 'f1_scores_median:', f1_scores_median, 'f1_scores_macro:',
          f1_scores_macro, 'f1_scores_micro:', f1_scores_micro)


    return acc,f1_scores_median



# query
def concerto_train_query(ref_model_path:str,ref_tf_path:str,query_tf_path:str, weight_path:str, super_parameters=None):
    set_seeds(0)
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5}
    #dirname = os.getcwd()
#     f = np.load(ref_tf_path + '/vocab_size.npz')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    f = np.load(os.path.join(ref_tf_path,'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
#     tf_list_2 = os.listdir(os.path.join(query_tf_path))
    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]

    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps, super_parameters['lr']*1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    weight_id_list = []
    weight_list = [f for f in os.listdir(ref_model_path) if f.endswith('h5')]
    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    # encode_network.load_weights(ref_model_path + '/weight_encoder_epoch{}.h5'.format(max(weight_id_list))) 
    encode_network.load_weights(ref_model_path + '/weight_encoder_epoch{}.h5'.format(max(weight_id_list)), by_name=True) # yyyx 0126, 支持多模态模型fine tune
    decode_network.load_weights(ref_model_path + '/weight_decoder_epoch{}.h5'.format(max(weight_id_list)))
    for epoch in range(super_parameters['epoch']):
        for file in train_target_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(
                    train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
        encode_network.save_weights(
            weight_path + '/weight_encoder_epoch{}.h5'.format(str(epoch + 1)))

    return weight_path

# 无监督一起训REF和query，解决读入模型不一致的问题
def concerto_train_ref_query(ref_tf_path: str, query_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'epoch_fineturn': 1, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(ref_tf_path, 'vocab_size.npz'))
    vocab_size = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network = multi_embedding_attention_transfer(multi_max_features=[vocab_size],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(ref_tf_path)) if 'tfrecord' in f]
    train_source_list = []
    for i in tf_list_1:
        train_source_list.append(os.path.join(ref_tf_path, i))

    tf_list_2 = [f for f in os.listdir(os.path.join(query_tf_path)) if 'tfrecord' in f]
    train_target_list = []
    for i in tf_list_2:
        train_target_list.append(os.path.join(query_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    test_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_cls_accuracy')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch_pretrain']):
        np.random.shuffle(train_source_list)
        for file in train_source_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))

    for epoch in range(super_parameters['epoch_fineturn']):
        np.random.shuffle(train_target_list)
        for file in train_target_list:
            print(file)
            train_db = create_classifier_dataset_multi([file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000)

            train_loss.reset_states()
            train_cls_accuracy.reset_states()
            test_cls_accuracy.reset_states()
            for step, (source_features, source_values, source_batch, source_id) in enumerate(train_db):
                # enumerate
                with tf.GradientTape() as tape:
                    z1 = encode_network([source_features, source_values], training=True)
                    z2 = decode_network([source_values], training=True)
                    mu_1 = mu_enc(z1)
                    var_1 = tf.exp(var_enc(z1))
                    ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1) + ssl_loss
                    train_loss(loss)

                variables = [encode_network.trainable_variables,
                             decode_network.trainable_variables,
                             mu_enc.trainable_variables,
                             var_enc.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, simclr loss: {:0.4f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))
        encode_network.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))

    return weight_path



def compute_loss(labels, logits, smoothing, vocab_size, padding_value=0):
  """Computes average (per-token) cross entropy loss.

  1. Applies label smoothing -- all entries in the groundtruth label tensor
     get non-zero probability mass.
  2. Computes per token loss of shape [batch_size, tgt_seq_len], where padded
     positions are masked, and then the sum of per token loss is normalized by
     the total number of non-padding entries.

  Args:
    labels: int tensor of shape [batch_size, tgt_seq_len], the groundtruth
      token ids.
    logits: float tensor of shape [batch_size, tgt_seq_len, vocab_size], the
      predicted logits of tokens over the vocabulary.
    smoothing: float scalar, the amount of label smoothing applied to the
      one-hot class labels.
    vocab_size: int scalar, num of tokens (including SOS and EOS) in the
      vocabulary.
    padding_value: int scalar, the vocabulary index of the PAD token.

  Returns:
    loss: float scalar tensor, the per-token cross entropy
  """
  # effective_vocab = vocab - {SOS_ID}
  effective_vocab_size = vocab_size - 1

  # prob mass allocated to the token that should've been predicted
  on_value = 1.0 - smoothing
  # prob mass allocated to all other tokens
  off_value = smoothing / (effective_vocab_size - 1)

  # [batch_size, tgt_seq_len, vocab_size]
  labels = tf.cast(labels,dtype=tf.int32)
  labels_one_hot = tf.one_hot(
      labels,
      depth=vocab_size,
      on_value=on_value,
      off_value=off_value)

  # compute cross entropy over all tokens in vocabulary but SOS_ID (i.e. 0)
  # because SOS_ID should never appear in the decoded sequence
  # [batch_size, tgt_seq_len]
  # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
  #     labels=labels_one_hot[:, :, 1:], logits=logits[:, :, 1:])
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
      labels=labels_one_hot, logits=logits)

  # this is the entropy when the softmax'ed logits == groundtruth labels
  # so it should be deducted from `cross_entropy` to make sure the minimum
  # possible cross entropy == 0
  normalizing_constant = -(on_value * tf.math.log(on_value) +
      (effective_vocab_size - 1) * off_value * tf.math.log(off_value + 1e-20))
  cross_entropy -= normalizing_constant

  # mask out predictions where the labels == `padding_value`
  # weights = tf.cast(tf.not_equal(labels, padding_value), 'float32')
  # cross_entropy *= weights
  # loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(weights)
  loss = tf.reduce_sum(cross_entropy)
  return loss


def concerto_train_multimodal_CLS_noITM(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}

    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path)
    f = np.load(saved_embed_path)
    peak_embed_ = f['peak_embed']
    print('peak_embed_ shape',peak_embed_.shape)
    peak_embed = np.squeeze(peak_embed_,axis=1)
    print('peak_embed shape', peak_embed.shape)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_cls(vocab_size=vocab_size_ATAC)
    print('cls')
    for i, w in enumerate(model.weights): print(i, w.name)

    Add_enc_1 = EncoderHead_add()
    Add_enc_2 = EncoderHead_add()
    ITM_head = EncoderHead_ITM(hidden_size = 1)
    #final_enc = scbasset_model(units_gene=vocab_size_RNA,flatten=False,globalpool=True)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    rna_atac_ss_loss = tf.keras.metrics.Mean(name='RNA_ATAC_ss_loss')
    rna_regress_loss = tf.keras.metrics.Mean(name='regress_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    #CE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            rna_atac_ss_loss.reset_states()
            rna_regress_loss.reset_states()
            itm_loss.reset_states()
            train_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    # ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    # mu_1 = mu_enc(z1)
                    # var_1 = tf.exp(var_enc(z1))
                    # KL_loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    encoder_output = Add_enc_1([z1,z1_1]) # teacher
                    decoder_output = Add_enc_2([z2, z2_1]) # student
                    #ssl_loss = simclr_loss(encoder_output, decoder_output, temperature=0.1)
                    ssl_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(encoder_output, decoder_output, temperature=0.1)
                    #################################### cross attention ###################################################
                    cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    cls_embed_gene = logits[:, 0, :]
                    ssl_cls_loss = simclr_loss(cls_embed_peak,cls_embed_gene, temperature=0.1)

                    #################################### ITM  #########################################################
                    # weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    # weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    # x = tf.linalg.diag_part(weights_t2t01)
                    # matrix = tf.linalg.diag(x)
                    # weights_t2t01 = weights_t2t01 - matrix
                    # x = tf.linalg.diag_part(weights_t2t10)
                    # matrix = tf.linalg.diag(x)
                    # weights_t2t10 = weights_t2t10 - matrix
                    # # select a negative teacher:0 for each student:1
                    # cell_gene_embed_neg = []
                    # neg_idx = tf.random.categorical(weights_t2t10, 1)
                    # for b in neg_idx:
                    #     cell_gene_embed_neg.append(cell_gene_embed[b[0]])
                    #
                    # cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)
                    #
                    # # select a negative student:1 for each teacher:0
                    # cell_peak_embed_neg = []
                    # neg_idx = tf.random.categorical(weights_t2t01, 1)
                    # for b in neg_idx:
                    #     cell_peak_embed_neg.append(cell_peak_embed[b[0]])
                    #
                    # cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    # cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    # cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    # cell_peak_embed_neg_, cell_gene_embed_neg_ = model([cell_peak_embed_all, cell_gene_embed_all])
                    # vl_embeddings = tf.concat([cell_gene_embed_pos[:,0,:],cell_gene_embed_neg_[:,0,:]],0)
                    # vl_output = ITM_head(vl_embeddings)
                    # ITM_label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    # ITM_loss = CE_loss(ITM_label,vl_output)

                    ##################################### regression task #####################################
                    # cell_gene_embed_flatten = tf.keras.layers.GlobalAveragePooling1D()(cell_gene_embed_1)
                    # output = final_enc(cell_gene_embed_pos)
                    #################################### loss function ###################################################
                    #KL_loss_1 = tf.keras.losses.kullback_leibler_divergence(source_values_RNA,output)
                    #h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
                    #KL_loss_1 = h(source_values_RNA,output)
                    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    #     labels=source_values_RNA, logits=output)

                    # regress_loss = CE_loss(source_values_RNA,output)
                    ################################### total loss #########################################################
                    loss = ssl_loss + ssl_cls_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    rna_atac_ss_loss(ssl_cls_loss)
                    #itm_loss(ITM_loss)
                    #rna_regress_loss(regress_loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             Add_enc_1.trainable_variables,
                             Add_enc_2.trainable_variables,
                             #final_enc.trainable_variables,
                             model.trainable_variables,
                             ITM_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                # train_accuracy(source_values_RNA, output)
                # train_ROC(source_values_RNA, output)
                # train_PR(source_values_RNA, output)
                if step > 0 and step % 50 == 0:
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, CE_loss:{:0.3f}, Accuracy:{},ROC:{},PR:{}'
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, itm loss:{:0.3f}, cls_ss_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          itm_loss.result(),
                                          rna_atac_ss_loss.result()
                                          #rna_regress_loss.result(),
                                          #train_accuracy.result(),
                                          #train_ROC.result(),
                                          #train_PR.result(),
                                          ))
                    # print('pred:',output[:2,:20])
                    # print('gt:',source_values_RNA[:2,:20])

        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        model.save_weights(
            os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))

        #ITM_head.save_weights(os.path.join(weight_path, 'ITM_head_epoch{}.h5'.format(str(epoch + 1))))
        # final_enc.save_weights(
        #     os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_1.save_weights(os.path.join(weight_path, 'weight_encoder_Add_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_2.save_weights(os.path.join(weight_path, 'weight_decoder_Add_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')

def concerto_train_multimodal_noCLS_noITM_MISA(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')

    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path)
    f = np.load(saved_embed_path)
    peak_embed_ = f['peak_embed']
    print('peak_embed_ shape',peak_embed_.shape)
    peak_embed = np.squeeze(peak_embed_,axis=1)
    print('peak_embed shape', peak_embed.shape)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_cls(vocab_size=vocab_size_ATAC)
    print('cls')
    for i, w in enumerate(model.weights): print(i, w.name)

    Add_enc_1 = EncoderHead_add()
    Add_enc_2 = EncoderHead_add()
    ITM_head = EncoderHead_ITM(hidden_size = 1)
    misa = MISA()
    #final_enc = scbasset_model(units_gene=vocab_size_RNA,flatten=False,globalpool=True)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    MISA_loss1 = tf.keras.metrics.Mean(name='MISA_loss')
    MISA_loss2 = tf.keras.metrics.Mean(name='MISA_loss')
    MISA_loss3 = tf.keras.metrics.Mean(name='MISA_loss')
    rna_regress_loss = tf.keras.metrics.Mean(name='regress_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            MISA_loss1.reset_states()
            MISA_loss2.reset_states()
            MISA_loss3.reset_states()
            itm_loss.reset_states()
            train_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    # ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    # mu_1 = mu_enc(z1)
                    # var_1 = tf.exp(var_enc(z1))
                    # KL_loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    # multi simclr
                    # encoder_output = Add_enc_1([z1,z1_1]) # teacher
                    # decoder_output = Add_enc_2([z2, z2_1]) # student
                    # ssl_loss = simclr_loss(encoder_output, decoder_output, temperature=0.1)
                    # ssl_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(encoder_output, decoder_output, temperature=0.1)

                    # rna simclr
                    ssl_loss_rna = simclr_loss(z1, z2, temperature=0.1)
                    # atac simclr
                    ssl_loss_atac = simclr_loss(z1_1, z2_1, temperature=0.1)
                    ssl_loss = (ssl_loss_rna + ssl_loss_atac)/2
                    #################################### MISA ##################################################
                    utterance_t, utt_private_t, utt_shared_t, utt_t, utt_t_recon, \
                    utterance_v, utt_private_v, utt_shared_v, utt_v, utt_v_recon,\
                    = misa(z1,z1_1)
                    # get_cmd_loss, losses between shared states
                    CMD_loss = CMD(utt_shared_t, utt_shared_v, 5)
                    # get_recon_loss
                    recon_loss = MSE_loss(utterance_t, utt_t_recon)
                    recon_loss += MSE_loss(utterance_v, utt_v_recon)
                    recon_loss = recon_loss / 2.0
                    # get_diff_loss
                    diff_loss = get_diff_loss(utt_shared_t,utt_shared_v,utt_private_t,utt_private_v)

                    #################################### cross attention ###################################################
                    # cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    # cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    # cls_embed_gene = cell_gene_embed_pos[:, 0, :]
                    # ssl_cls_loss = simclr_loss(cls_embed_peak,cls_embed_gene, temperature=0.1)

                    #################################### ITM  #########################################################
                    # weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    # weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    # x = tf.linalg.diag_part(weights_t2t01)
                    # matrix = tf.linalg.diag(x)
                    # weights_t2t01 = weights_t2t01 - matrix
                    # x = tf.linalg.diag_part(weights_t2t10)
                    # matrix = tf.linalg.diag(x)
                    # weights_t2t10 = weights_t2t10 - matrix
                    # # select a negative teacher:0 for each student:1
                    # cell_gene_embed_neg = []
                    # neg_idx = tf.random.categorical(weights_t2t10, 1)
                    # for b in neg_idx:
                    #     cell_gene_embed_neg.append(cell_gene_embed[b[0]])
                    #
                    # cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)
                    #
                    # # select a negative student:1 for each teacher:0
                    # cell_peak_embed_neg = []
                    # neg_idx = tf.random.categorical(weights_t2t01, 1)
                    # for b in neg_idx:
                    #     cell_peak_embed_neg.append(cell_peak_embed[b[0]])
                    #
                    # cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    # cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    # cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    # cell_peak_embed_neg_, cell_gene_embed_neg_ = model([cell_peak_embed_all, cell_gene_embed_all])
                    # vl_embeddings = tf.concat([cell_gene_embed_pos[:,0,:],cell_gene_embed_neg_[:,0,:]],0)
                    # vl_output = ITM_head(vl_embeddings)
                    # ITM_label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    # ITM_loss = CE_loss(ITM_label,vl_output)

                    ##################################### regression task #####################################
                    # cell_gene_embed_flatten = tf.keras.layers.GlobalAveragePooling1D()(cell_gene_embed_1)
                    # output = final_enc(cell_gene_embed_pos)
                    #################################### loss function ###################################################
                    #KL_loss_1 = tf.keras.losses.kullback_leibler_divergence(source_values_RNA,output)
                    #h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
                    #KL_loss_1 = h(source_values_RNA,output)
                    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    #     labels=source_values_RNA, logits=output)

                    # regress_loss = CE_loss(source_values_RNA,output)
                    ################################### total loss #########################################################
                    loss = ssl_loss + CMD_loss + recon_loss + diff_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    MISA_loss1(CMD_loss)
                    MISA_loss2(recon_loss)
                    MISA_loss3(diff_loss)
                    #itm_loss(ITM_loss)
                    #rna_regress_loss(regress_loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             misa.trainable_variables,
                             #Add_enc_1.trainable_variables,
                             #Add_enc_2.trainable_variables,
                             #final_enc.trainable_variables,
                             #model.trainable_variables,
                             #ITM_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                # train_accuracy(source_values_RNA, output)
                # train_ROC(source_values_RNA, output)
                # train_PR(source_values_RNA, output)
                if step > 0 and step % 50 == 0:
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, CE_loss:{:0.3f}, Accuracy:{},ROC:{},PR:{}'
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, ss_loss:{:0.3f}, itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          itm_loss.result(),
                                          MISA_loss1.result(),
                                          MISA_loss2.result(),
                                          MISA_loss3.result(),
                                          #train_ROC.result(),
                                          #train_PR.result(),
                                          ))
                    # print('pred:',output[:2,:20])
                    # print('gt:',source_values_RNA[:2,:20])

        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        misa.save_weights(os.path.join(weight_path, 'weight_MISA_epoch{}.h5'.format(str(epoch + 1))))
        # model.save_weights(
        #     os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))

        #ITM_head.save_weights(os.path.join(weight_path, 'ITM_head_epoch{}.h5'.format(str(epoch + 1))))
        # final_enc.save_weights(
        #     os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        # Add_enc_1.save_weights(os.path.join(weight_path, 'weight_encoder_Add_epoch{}.h5'.format(str(epoch + 1))))
        # Add_enc_2.save_weights(os.path.join(weight_path, 'weight_decoder_Add_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')

def concerto_test_multimodal_noCLS_noITM_MISA(task: str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None, saved_weight_path_regress=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'epoch_regress': 50, 'lr': 1e-4, 'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    misa = MISA()
    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    decode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_epoch{epoch}.h5')
    embedding_decoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_embedding_epoch{epoch}.h5',
                                            by_name=True)
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5',
                                            by_name=True)

    print('load saved weight')

    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    project_RNA_all = []
    project_ATAC_all = []
    private_RNA_all = []
    private_ATAC_all = []
    shared_RNA_all = []
    shared_ATAC_all = []
    recon_RNA_all = []
    recon_ATAC_all = []
    RNA_id_all = []

    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):

            if step == 0:
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                utterance_t, utt_private_t, utt_shared_t, utt_t, utt_t_recon, \
                utterance_v, utt_private_v, utt_shared_v, utt_v, utt_v_recon, \
                    = misa(z1, z1_1)
                break

        misa.load_weights(saved_weight_path_pretrain + f'weight_MISA_epoch{epoch}.h5', by_name=True)
        # final_enc.load_weights(saved_weight_path_regress + f'weight_project_epoch{epoch_regress}.h5', by_name=True)

        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        project_RNA = np.zeros((feature_len, dim))
        project_ATAC = np.zeros((feature_len, dim))
        private_RNA = np.zeros((feature_len, dim))
        private_ATAC = np.zeros((feature_len, dim))
        shared_RNA = np.zeros((feature_len, dim))
        shared_ATAC = np.zeros((feature_len, dim))
        recon_RNA = np.zeros((feature_len, dim))
        recon_ATAC = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
                encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)
                utterance_t, utt_private_t, utt_shared_t, utt_t, utt_t_recon, \
                utterance_v, utt_private_v, utt_shared_v, utt_v, utt_v_recon, \
                    = misa(z1, z1_1)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
                cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
                project_RNA[all_samples:all_samples + len(source_id_RNA), :] = utterance_t
                project_ATAC[all_samples:all_samples + len(source_id_RNA), :] = utterance_v
                private_RNA[all_samples:all_samples + len(source_id_RNA), :] = utt_private_t
                private_ATAC[all_samples:all_samples + len(source_id_RNA), :] = utt_private_v
                shared_RNA[all_samples:all_samples + len(source_id_RNA), :] = utt_shared_t
                shared_ATAC[all_samples:all_samples + len(source_id_RNA), :] = utt_shared_v
                recon_RNA[all_samples:all_samples + len(source_id_RNA), :] = utt_t_recon
                recon_ATAC[all_samples:all_samples + len(source_id_RNA), :] = utt_v_recon
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        project_RNA_all.extend(project_RNA[:all_samples])
        project_ATAC_all.extend(project_ATAC[:all_samples])
        private_RNA_all.extend(private_RNA[:all_samples])
        private_ATAC_all.extend(private_ATAC[:all_samples])
        shared_RNA_all.extend(shared_RNA[:all_samples])
        shared_ATAC_all.extend(shared_ATAC[:all_samples])
        recon_RNA_all.extend(recon_RNA[:all_samples])
        recon_ATAC_all.extend(recon_ATAC[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])


    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32')
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32')
    project_RNA_all = np.array(project_RNA_all).astype('float32')
    project_ATAC_all = np.array(project_ATAC_all).astype('float32')
    private_RNA_all = np.array(private_RNA_all).astype('float32')
    private_ATAC_all = np.array(private_ATAC_all).astype('float32')
    shared_RNA_all = np.array(shared_RNA_all).astype('float32')
    shared_ATAC_all = np.array(shared_ATAC_all).astype('float32')
    recon_RNA_all = np.array(recon_RNA_all).astype('float32')
    recon_ATAC_all = np.array(recon_ATAC_all).astype('float32')

    return cell_embed_RNA_all, cell_embed_ATAC_all, project_RNA_all, \
           project_ATAC_all, private_RNA_all, private_ATAC_all,shared_RNA_all,shared_ATAC_all, \
           recon_RNA_all,recon_ATAC_all,RNA_id_all



def attune_train_regulatory(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,mask_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch_pretrain': 20,'epoch_transformer':10, 'lr': 1e-4, 'drop_rate': 0.1}
    epoch_pretrain = super_parameters['epoch_pretrain']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path + f'weight_encoder_epoch{epoch_pretrain}.h5')
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)


    embedding_encoder_pretrain.load_weights(saved_weight_path + f'weight_encoder_embedding_epoch{epoch_pretrain}.h5')
    model = Transformer_model_cls(vocab_size=vocab_size_ATAC, attention_mask_path=mask_path)
    discriminator_head = EncoderHead_ITM(hidden_size = 1)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    match_loss = tf.keras.metrics.Mean(name='Match_loss')
    total_update_steps = 300 * super_parameters['epoch_transformer']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    CE_loss = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(super_parameters['epoch_transformer']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            match_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA],training=False)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC],training=False)
                    #################################### transformer ###################################################
                    cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    cls_embed_gene = cell_gene_embed_pos[:, 0, :]
                    ssl_cls_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(cls_embed_peak,cls_embed_gene)
                    #################################### ITM  #########################################################
                    weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    x = tf.linalg.diag_part(weights_t2t01)
                    matrix = tf.linalg.diag(x)
                    weights_t2t01 = weights_t2t01 - matrix
                    x = tf.linalg.diag_part(weights_t2t10)
                    matrix = tf.linalg.diag(x)
                    weights_t2t10 = weights_t2t10 - matrix
                    # select a negative teacher:0 for each student:1
                    cell_gene_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t10, 1)
                    for b in neg_idx:
                        cell_gene_embed_neg.append(cell_gene_embed[b[0]])

                    cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)

                    # select a negative student:1 for each teacher:0
                    cell_peak_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t01, 1)
                    for b in neg_idx:
                        cell_peak_embed_neg.append(cell_peak_embed[b[0]])

                    cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    cell_peak_embed_neg_, cell_gene_embed_neg_,logits_neg = model([cell_peak_embed_all, cell_gene_embed_all])
                    vl_embeddings = tf.concat([logits[:,0,:],logits_neg[:,0,:]],0)
                    vl_output = discriminator_head(vl_embeddings)
                    label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    discriminator_loss = CE_loss(label,vl_output)
                    ################################### total loss #########################################################
                    loss = ssl_cls_loss + discriminator_loss
                    train_loss(loss)
                    cls_ss_loss(ssl_cls_loss)
                    match_loss(discriminator_loss)

                variables = [model.trainable_variables,
                             discriminator_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 20 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, cls_ss_loss:{:0.3f}, match loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          cls_ss_loss.result(),
                                          match_loss.result(),
                                          ))

        model.save_weights(
            os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))
        discriminator_head.save_weights(os.path.join(weight_path, 'discriminator_head_epoch{}.h5'.format(str(epoch + 1))))


    return print('finished')

def attune_train_regulatory_SHARE(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,mask_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 128, 'epoch_pretrain': 20,'epoch_transformer':10, 'lr': 1e-4, 'drop_rate': 0.1}
    epoch_pretrain = super_parameters['epoch_pretrain']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path + f'weight_encoder_epoch{epoch_pretrain}.h5')
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)


    embedding_encoder_pretrain.load_weights(saved_weight_path + f'weight_encoder_embedding_epoch{epoch_pretrain}.h5')
    model = Transformer_model_cls_1(vocab_size=vocab_size_ATAC, attention_mask_path=mask_path)
    discriminator_head = EncoderHead_ITM(hidden_size = 1)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    match_loss = tf.keras.metrics.Mean(name='Match_loss')
    total_update_steps = 300 * super_parameters['epoch_transformer']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    CE_loss = tf.keras.losses.BinaryCrossentropy()

    for epoch in range(super_parameters['epoch_transformer']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            match_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA],training=False)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC],training=False)
                    #################################### transformer ###################################################
                    cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    cls_embed_gene = cell_gene_embed_pos[:, 0, :]
                    ssl_cls_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(cls_embed_peak,cls_embed_gene)
                    #################################### ITM  #########################################################
                    weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    x = tf.linalg.diag_part(weights_t2t01)
                    matrix = tf.linalg.diag(x)
                    weights_t2t01 = weights_t2t01 - matrix
                    x = tf.linalg.diag_part(weights_t2t10)
                    matrix = tf.linalg.diag(x)
                    weights_t2t10 = weights_t2t10 - matrix
                    # select a negative teacher:0 for each student:1
                    cell_gene_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t10, 1)
                    for b in neg_idx:
                        cell_gene_embed_neg.append(cell_gene_embed[b[0]])

                    cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)

                    # select a negative student:1 for each teacher:0
                    cell_peak_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t01, 1)
                    for b in neg_idx:
                        cell_peak_embed_neg.append(cell_peak_embed[b[0]])

                    cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    cell_peak_embed_neg_, cell_gene_embed_neg_,logits_neg = model([cell_peak_embed_all, cell_gene_embed_all])
                    vl_embeddings = tf.concat([logits[:,0,:],logits_neg[:,0,:]],0)
                    vl_output = discriminator_head(vl_embeddings)
                    label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    discriminator_loss = CE_loss(label,vl_output)
                    ################################### total loss #########################################################
                    loss = ssl_cls_loss + discriminator_loss
                    train_loss(loss)
                    cls_ss_loss(ssl_cls_loss)
                    match_loss(discriminator_loss)

                variables = [model.trainable_variables,
                             discriminator_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 20 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, cls_ss_loss:{:0.3f}, match loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          cls_ss_loss.result(),
                                          match_loss.result(),
                                          ))

        model.save_weights(
            os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))
        discriminator_head.save_weights(os.path.join(weight_path, 'discriminator_head_epoch{}.h5'.format(str(epoch + 1))))


    return print('finished')


def attune_pretrain(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1,'temperature':0.1}

    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    ssl_loss_t = simclr_loss(z1, z1_1, temperature=super_parameters['temperature'])
                    ssl_loss_s = simclr_loss(z2, z2_1, temperature=super_parameters['temperature'])
                    ssl_loss = (ssl_loss_t + ssl_loss_s)/2
                    ################################### total loss #########################################################
                    loss = ssl_loss
                    train_loss(loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))


def concerto_train_multimodal_transformer_mutulsim_pretrain_SHARE(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path,by_name=True)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    #embedding_encoder_pretrain.load_weights('./multi_model/scCAT_seq/label_finetune/seed_0/weight_cls_1_epoch20.h5',by_name=True)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA],training=False) # origin False
                    #z2 = decode_network_pretrain(source_values_RNA)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    #z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    # rna atac simclr
                    ssl_loss_rna_atac = simclr_loss(z1, z1_1, temperature=0.1)
                    #ssl_loss_atac_rna = simclr_loss(z2, z2_1, temperature=0.1)
                    ssl_loss = ssl_loss_rna_atac

                    ################################### total loss #########################################################
                    loss = ssl_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    cls_ss_loss(ssl_loss)
                    #itm_loss(ITM_loss)

                variables = [#encode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}'
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, regress_loss:{:0.3f}'
                    #template = 'Epoch{},step{},total loss:{:0.3f},ss_loss:{:0.3f},itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f},regress_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          #cls_ss_loss.result(),
                                          #itm_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))

def concerto_train_multimodal_transformer_mutulsim_pretrain_SHARE_withstudent(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path,by_name=True)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    #embedding_encoder_pretrain.load_weights('./multi_model/scCAT_seq/label_finetune/seed_0/weight_cls_1_epoch20.h5',by_name=True)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA],training=False) # origin False
                    z2 = decode_network_pretrain(source_values_RNA)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    # rna atac simclr
                    ssl_loss_rna_atac = simclr_loss(z1, z1_1, temperature=0.1)
                    ssl_loss_atac_rna = simclr_loss(z2, z2_1, temperature=0.1)
                    ssl_loss = (ssl_loss_rna_atac + ssl_loss_atac_rna)/2

                    ################################### total loss #########################################################
                    loss = ssl_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    cls_ss_loss(ssl_loss)
                    #itm_loss(ITM_loss)

                variables = [#encode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}'
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, regress_loss:{:0.3f}'
                    #template = 'Epoch{},step{},total loss:{:0.3f},ss_loss:{:0.3f},itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f},regress_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          #cls_ss_loss.result(),
                                          #itm_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))

def concerto_train_multimodal_transformer_mutulsim_pretrain_supervised_center(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}

    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights('./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_epoch20.h5')
    # SHARE:./multi_model/SHARE_all/5_randomseed/seed_0/weight_encoder_epoch20.h5
    # scCAT: ./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_epoch20.h5
    embedding_encoder_pretrain.load_weights(
        './multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5')
    # SHARE: ./multi_model/SHARE_all/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5
    # scCAT: ./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    train_cls_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    num_classes = 5 #SHARE:22 ; scCAT: 5

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output1 = encode_network_pretrain.layers[-1].output
    output2 = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output1)
    cls_network = tf.keras.Model(encode_network_pretrain.input, outputs=output2)
    output_1 = embedding_encoder_pretrain.layers[-1].output
    output_1 = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output_1)
    cls_network_1 = tf.keras.Model(embedding_encoder_pretrain.input, outputs=output_1)
    ##########################################################################################
    aux_input = tf.keras.layers.Input((num_classes,), name='Center_Input')
    center_output = tf.keras.layers.Dense(2, name='Axis_Layer')(output1)
    center_output = CenterLossLayer(alpha=0.5, num_classes=num_classes, name='Center')([center_output, aux_input])
    network = tf.keras.models.Model([encode_network_pretrain.input, aux_input], [output2, center_output])
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi_supervised([RNA_file],
                                                                      batch_size=super_parameters['batch_size'],
                                                                      is_training=True,
                                                                      data_augment=False,
                                                                      shuffle_size=10000,
                                                                      )
            train_db_ATAC = create_classifier_dataset_multi_supervised([ATAC_file],
                                                                       batch_size=super_parameters['batch_size'],
                                                                       is_training=True,
                                                                       data_augment=False,
                                                                       shuffle_size=10000,
                                                                       )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,source_label,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,source_label1,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1
                y_onehot = tf.one_hot(source_label, depth=num_classes)
                with tf.GradientTape() as tape:

                    #outputs = cls_network([source_features_RNA, source_values_RNA], training=True)
                    outputs = network([source_features_RNA, source_values_RNA, y_onehot])
                    #outputs_1 = cls_network_1([source_features_ATAC, source_values_ATAC], training=True)
                    classifer_loss = cls_loss_object(source_label, outputs[0])
                    #classifer_loss_1 = cls_loss_object(source_label, outputs_1)
                    #####################################################################
                    center_loss = 0.5 * tf.reduce_sum(outputs[1], axis=0)
                    classification_loss = tf.reduce_mean(classifer_loss + 0.05 * center_loss)
                    ########################################################################
                    loss_all = classification_loss
                    train_cls_accuracy(source_label, outputs[0])
                    train_cls_accuracy_1(source_label,outputs[0])
                    train_loss(loss_all)
                    rna_ss_loss(loss_all)
                    cls_ss_loss(loss_all)

                variables = [network.trainable_variables,
                             #encode_network_pretrain.trainable_variables,
                             #decode_network_pretrain.trainable_variables,
                             #embedding_encoder_pretrain.trainable_variables,
                             #embedding_decoder_pretrain.trainable_variables,
                             ]
                grads = tape.gradient(loss_all, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, loss all: {:0.4f},loss contrast:{:0.4f},loss cls: {:0.4f},train acc: {:0.4f},train acc_1: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          cls_ss_loss.result(),
                                          train_cls_accuracy.result(),
                                          train_cls_accuracy_1.result(),
                                          ))

        network.save_weights(os.path.join(weight_path, 'weight_cls_epoch{}.h5'.format(str(epoch + 1))))
        # cls_network.save_weights(os.path.join(weight_path, 'weight_cls_epoch{}.h5'.format(str(epoch + 1))))
        # cls_network_1.save_weights(os.path.join(weight_path, 'weight_cls_1_epoch{}.h5'.format(str(epoch + 1))))
        # encode_network_pretrain.save_weights(
        #     os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        # decode_network_pretrain.save_weights(
        #     os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        # embedding_encoder_pretrain.save_weights(
        #     os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        # embedding_decoder_pretrain.save_weights(
        #     os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))

def concerto_train_multimodal_transformer_mutulsim_pretrain_unsupervised_center(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}

    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    # encode_network_pretrain.load_weights('./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_epoch20.h5')
    # SHARE:./multi_model/SHARE_all/5_randomseed/seed_0/weight_encoder_epoch20.h5
    # scCAT: ./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_epoch20.h5
    # embedding_encoder_pretrain.load_weights(
    #     './multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5')
    # SHARE: ./multi_model/SHARE_all/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5
    # scCAT: ./multi_model/scCAT_seq/5_randomseed/seed_0/weight_encoder_embedding_epoch20.h5
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_cls_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    train_cls_accuracy_1 = tf.keras.metrics.SparseCategoricalAccuracy(name='train_cls_accuracy')
    cls_loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    num_classes = 22 #SHARE:22 ; scCAT: 5

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    output1 = encode_network_pretrain.layers[-1].output
    output2 = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output1)
    cls_network = tf.keras.Model(encode_network_pretrain.input, outputs=output2)
    output_1 = embedding_encoder_pretrain.layers[-1].output
    output_1 = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output_1)
    cls_network_1 = tf.keras.Model(embedding_encoder_pretrain.input, outputs=output_1)
    ##########################################################################################
    aux_input = tf.keras.layers.Input((num_classes,), name='Center_Input')
    center_output = tf.keras.layers.Dense(2, name='Axis_Layer')(output1)
    center_output = CenterLossLayer(alpha=0.5, num_classes=num_classes, name='Center')([center_output, aux_input])
    #network = tf.keras.models.Model([encode_network_pretrain.input, aux_input], [output2, center_output])
    network = tf.keras.models.Model([encode_network_pretrain.input, aux_input], [center_output])
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi_supervised([RNA_file],
                                                                      batch_size=super_parameters['batch_size'],
                                                                      is_training=True,
                                                                      data_augment=False,
                                                                      shuffle_size=10000,
                                                                      )
            train_db_ATAC = create_classifier_dataset_multi_supervised([ATAC_file],
                                                                       batch_size=super_parameters['batch_size'],
                                                                       is_training=True,
                                                                       data_augment=False,
                                                                       shuffle_size=10000,
                                                                       )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,source_label,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,source_label1,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1
                y_onehot = tf.one_hot(source_label, depth=num_classes)
                with tf.GradientTape() as tape:

                    #outputs = cls_network([source_features_RNA, source_values_RNA], training=True)
                    outputs = network([source_features_RNA, source_values_RNA, y_onehot])
                    #outputs_1 = cls_network_1([source_features_ATAC, source_values_ATAC], training=True)
                    #classifer_loss = cls_loss_object(source_label, outputs[0])
                    #classifer_loss_1 = cls_loss_object(source_label, outputs_1)
                    #####################################################################
                    center_loss = 0.5 * tf.reduce_sum(outputs, axis=0)
                    classification_loss = tf.reduce_mean(0.05 * center_loss)
                    ########################################################################
                    loss_all = classification_loss
                    #train_cls_accuracy(source_label, outputs[0])
                    #train_cls_accuracy_1(source_label,outputs[0])
                    train_loss(loss_all)
                    rna_ss_loss(loss_all)
                    cls_ss_loss(loss_all)

                variables = [network.trainable_variables,
                             ]
                grads = tape.gradient(loss_all, variables)
                for grad, var in zip(grads, variables):
                    opt.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch {}, step {}, loss all: {:0.4f},loss contrast:{:0.4f},loss cls: {:0.4f}'
                    print(template.format(epoch,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          cls_ss_loss.result(),
                                          ))

        network.save_weights(os.path.join(weight_path, 'weight_cls_epoch{}.h5'.format(str(epoch + 1))))

def concerto_train_multimodal_transformer_mutulsim_pretrain_ablation(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1,'dim':128}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=super_parameters['dim'],
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters['dim'],
                                                        head_2=super_parameters['dim'],
                                                        head_3=super_parameters['dim'])

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=super_parameters['dim'],
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters['dim'],
                                                        head_2=super_parameters['dim'],
                                                        head_3=super_parameters['dim'])

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=super_parameters['dim'],
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters['dim'],
                                                        head_2=super_parameters['dim'],
                                                        head_3=super_parameters['dim'])
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=super_parameters['dim'],
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=super_parameters['dim'],
                                                        head_2=super_parameters['dim'],
                                                        head_3=super_parameters['dim'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    # rna atac simclr
                    ssl_loss_rna_atac = simclr_loss(z1, z1_1, temperature=0.1)
                    ssl_loss_atac_rna = simclr_loss(z2, z2_1, temperature=0.1)
                    ssl_loss = (ssl_loss_rna_atac + ssl_loss_atac_rna)/2

                    ################################### total loss #########################################################
                    loss = ssl_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    cls_ss_loss(ssl_loss)
                    #itm_loss(ITM_loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             #misa.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}'
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, regress_loss:{:0.3f}'
                    #template = 'Epoch{},step{},total loss:{:0.3f},ss_loss:{:0.3f},itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f},regress_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          #cls_ss_loss.result(),
                                          #itm_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        #misa.save_weights(os.path.join(weight_path, 'weight_MISA_epoch{}.h5'.format(str(epoch + 1))))

def concerto_train_multimodal_moco_pretrain(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    temperature = 0.07
    moco = MoCo(vocab_size_RNA,vocab_size_ATAC,temperature,super_parameters['batch_size'],opt_simclr)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1
                loss,encode_network_pretrain,embedding_encoder_pretrain,decode_network_pretrain,embedding_decoder_pretrain = moco([source_features_RNA,source_values_RNA,source_features_ATAC,source_values_ATAC])

                train_loss(loss)

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f},concate_loss:{:0.3f}'
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, regress_loss:{:0.3f}'
                    #template = 'Epoch{},step{},total loss:{:0.3f},ss_loss:{:0.3f},itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f},regress_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          cls_ss_loss.result(),
                                          #itm_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        #misa.save_weights(os.path.join(weight_path, 'weight_MISA_epoch{}.h5'.format(str(epoch + 1))))

class Buffer:

    def __init__(self, train_X, buffer_config):
        self.train_X = train_X
        self.config = buffer_config

        self.train_n = train_X.shape[0]
        self.size = train_X.shape[1]

        self.idx = 0

        # self.counter = torch.ones(size=[self.config['size'],1]) * 5.0
        self.counter = K.ones(shape=(self.config['size'], 1)) * 5.0
        self.init_buffer()

    def __get_sample__(self, n_samples):
        # r_idx = torch.randint(self.train_X.shape[0], size=[n_samples])
        r_idx = K.random_uniform(shape=(n_samples, 1), minval=0, maxval=self.train_X.shape[0], dtype=tf.int32)
        # samples = self.train_X[r_idx]
        samples = tf.gather(self.train_X, r_idx[:, 0])
        # samples = transform_data(samples, self.t, self.config['bs'])
        return samples

    def __get_rand__(self, n_samples):
        if 'CD_ratio' in self.config:
            samples = self.__get_sample__(n_samples)
            samples_1 = K.random_uniform(shape=(n_samples, self.train_X.shape[1]), minval=0, maxval=1, dtype=tf.float32)
            return (self.config['CD_ratio'] * samples + (1.0 - self.config['CD_ratio']) * samples_1) * 2.0 - 1.0
        else:
            return print('')

    def init_buffer(self):
        self.buffer = self.__get_rand__(self.config['size'])

    def sample(self, n_samples):
        #
        self.idx = K.random_uniform(shape=(n_samples, 1), minval=0, maxval=self.config['size'], dtype=tf.int32)
        #print('self.idx', self.idx.shape)
        sample = tf.gather(self.buffer, self.idx[:, 0])

        count = tf.gather(self.counter, self.idx[:, 0])
        #print('sample shape', sample.shape)
        #print('count shape', count.shape)

        sample_2 = self.__get_rand__(n_samples)
        sample = tf.Variable(sample, name='sample')
        count = tf.Variable(count, name='count')
        sample[:int(n_samples * self.config['rho']), :].assign(sample_2[:int(n_samples * self.config['rho']), :])
        count[:int(n_samples * self.config['rho']), :].assign(K.zeros(shape=(int(n_samples * self.config['rho']), 1)))
        self.counter = count + 1.0
        #print('self.counter shape', self.counter.shape)

        return sample, count

    def update(self, samples):
        self.buffer = samples

def similarity_f(x, y, normalize):

    if normalize:

        x = x / (tf.norm(x,axis=1,keepdims=True) + 1e-10)
        y = y / (tf.norm(y,axis=1,keepdims=True) + 1e-10)

    #return -(x - y).square().sum(dim=1)
    return -tf.reduce_sum(tf.square(x[:,None]-y[None]),axis=2)

def logsumexp(log_p):
    m = tf.reduce_max(log_p, axis=1, keepdims=True)
    return tf.reduce_logsumexp(log_p-m,axis=1,keepdims=True)+m


class MSGLD:

    def __init__(self, config):
        self.config = config

    def __get_std__(self, count):
        return self.config['min_std'] + (self.config['max_std'] - self.config['min_std']) * tf.maximum(
            1.0 - count / self.config['threshold'], K.zeros_like(count))

    def __call__(self, init, count):
        out = init
        #lp = tf.reduce_mean(log_pdf(out))
        #lp = tf.reduce_mean(log_pdf)
        # print('lp', lp)
        # lp.backward()
        out_1 = K.random_uniform(shape=(out.shape[0], out.shape[1]), minval=0, maxval=1, dtype=tf.float32)
        out = out + self.config['lr'] * tf.clip_by_value(out, -self.config['tau'],
                                                         self.config['tau']) + self.__get_std__(count) * out_1

        return out

def concerto_train_multimodal_transformer_mutulsim_pretrain_EBCLR(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,config_dir:str,super_parameters=None):

    def load_config(config_path):
        with open(config_path, 'r') as f:
            #config = yaml.load(f, Loader=yaml.FullLoader)
            config = yaml.safe_load(f)
        return config
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    # encode_network_pretrain.load_weights(saved_weight_path)
    # f = np.load(saved_embed_path)
    # peak_embed_ = f['peak_embed']
    # print('peak_embed_ shape',peak_embed_.shape)
    # peak_embed = np.squeeze(peak_embed_,axis=1)
    # print('peak_embed shape', peak_embed.shape)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    #embedding_encoder_pretrain.load_weights('./multi_model/10x_10k_80train_20test/mutul_simloss_transformer/noAdd_CLS_ITM_atmask0_divide/weight_encoder_embedding_epoch20.h5')
    #model = Transformer_model_cls(vocab_size=vocab_size_ATAC)
    #ITM_head = EncoderHead_ITM(hidden_size = 1)

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    cls_ss_loss = tf.keras.metrics.Mean(name='cls_ss_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    ################## EBCLR 配置 #####################################################
    config = load_config(config_dir)
    sgld = MSGLD(config['sgld'])
    logits = lambda x, y: similarity_f(x, y, config['normalize']) / config['temperature']
    log_pdf = lambda x, y: logsumexp(logits(x, y))

    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=True,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            cls_ss_loss.reset_states()
            itm_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### contrastive loss ####################################
                    buffer = Buffer(source_values_RNA, config['buffer'])
                    buffer_1 = Buffer(source_values_ATAC, config['buffer'])
                    #source_values_RNA = source_values_RNA*2-1
                    #source_values_ATAC = source_values_ATAC*2-1
                    # source_values_RNA = source_values_RNA
                    # source_values_ATAC = source_values_ATAC
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    ################################### EBCLR ##################################################
                    X_init, count = buffer.sample(source_values_RNA.shape[0])

                    #z1_init = decode_network_pretrain(X_init)
                    #z1_init,_ = encode_network_pretrain([source_features_RNA,X_init])

                    X_n = sgld(X_init, count)
                    buffer.update(X_n)
                    #Z_n,_ = encode_network_pretrain([source_features_RNA,X_n])
                    Z_n = decode_network_pretrain(X_n)
                    #log_pdf_d = log_pdf(z1, z1_1)
                    #log_pdf_n = log_pdf(Z_n, z2_1)
                    #############################
                    X_init, count = buffer_1.sample(source_values_ATAC.shape[0])

                    #z1_init, _ = embedding_encoder_pretrain([source_features_ATAC, X_init])
                    #z1_init = embedding_decoder_pretrain(X_init)
                    X_n = sgld(X_init, count)
                    buffer_1.update(X_n)
                    #Z_n_1, _ = embedding_encoder_pretrain([source_features_ATAC, X_n])
                    Z_n_1 = embedding_decoder_pretrain(X_n)
                    #log_pdf_n_1 = log_pdf(z1, Z_n_1)

                    # gen_loss = tf.reduce_mean(log_pdf_n) + config['lmda1'] * (
                    #             tf.reduce_sum(tf.square(z1)) + tf.reduce_sum(tf.square(Z_n))) + tf.reduce_mean(log_pdf_n_1) \
                    #            + config['lmda1'] * (tf.reduce_sum(tf.square(Z_n_1)) + tf.reduce_sum(tf.square(z1_1)))
                    gen_loss = config['lmda1'] * (tf.reduce_sum(tf.square(Z_n)))
                    gen_loss_1 = config['lmda1'] * (tf.reduce_sum(tf.square(Z_n_1)))
                    ssl_loss_1 = simclr_loss(z1, Z_n, temperature=0.1)
                    ssl_loss_2 = simclr_loss(z1_1, Z_n_1, temperature=0.1) # add
                    ssl_loss = simclr_loss(z1, z1_1, temperature=0.1)
                    loss = ssl_loss + gen_loss + gen_loss_1 + ssl_loss_1 + ssl_loss_2
                    ################################### total loss #########################################################
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    cls_ss_loss(gen_loss + gen_loss_1)
                    itm_loss(ssl_loss_1+ssl_loss_2)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             #misa.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                if step > 0 and step % 5 == 0:
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f},gen_loss:{:0.3f},rna_ss_loss:{:0.3f}'
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, regress_loss:{:0.3f}'
                    #template = 'Epoch{},step{},total loss:{:0.3f},ss_loss:{:0.3f},itm loss:{:0.3f},CMD_loss:{:0.3f},RECON_loss:{:0.3f},DIFF_loss:{:0.3f},regress_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          cls_ss_loss.result(),
                                          itm_loss.result(),
                                          ))


        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        #misa.save_weights(os.path.join(weight_path, 'weight_MISA_epoch{}.h5'.format(str(epoch + 1))))



def attune_test(RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'lr': 1e-4, 'drop_rate': 0.1}

    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5')
    print('load saved weight')
    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    RNA_id_all = []
    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 50000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):
            if all_samples >= feature_len:
                break

            z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
            z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
            encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
            encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)
            ######################################################################
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
            cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
            all_samples += len(source_id_RNA)
            print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])

    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32') # encoder
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32') # encoder

    return cell_embed_RNA_all, cell_embed_ATAC_all, RNA_id_all

def concerto_test_transformer_mutulsim_pretrain_ablation(task: str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None, saved_weight_path_regress=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'epoch_regress': 50, 'lr': 1e-4, 'drop_rate': 0.1,'dim':128}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=super_parameters['dim'],
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=super_parameters['dim'],
                                                                     head_2=super_parameters['dim'],
                                                                     head_3=super_parameters['dim'])

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=super_parameters['dim'],
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=super_parameters['dim'],
                                                                     head_2=super_parameters['dim'],
                                                                     head_3=super_parameters['dim'])

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=super_parameters['dim'],
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters['dim'],
        head_2=super_parameters['dim'],
        head_3=super_parameters['dim'])
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=super_parameters['dim'],
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=super_parameters['dim'],
        head_2=super_parameters['dim'],
        head_3=super_parameters['dim'])


    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    decode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_epoch{epoch}.h5')
    embedding_decoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_embedding_epoch{epoch}.h5')
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5')

    print('load saved weight')

    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    shared_RNA_all = []
    shared_ATAC_all = []
    RNA_id_all = []

    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        dim = super_parameters['dim']
        if n_cells_for_sample is None:
            feature_len = 20000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        shared_RNA = np.zeros((feature_len, dim))
        shared_ATAC = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                z2 = decode_network_pretrain(source_values_RNA)
                z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
                encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)
                decoder_output_RNA = tf.nn.l2_normalize(z2, axis=-1)
                decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
                cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
                shared_RNA[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_RNA
                shared_ATAC[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        shared_RNA_all.extend(shared_RNA[:all_samples])
        shared_ATAC_all.extend(shared_ATAC[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])


    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32') # encoder
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32') # encoder
    shared_RNA_all = np.array(shared_RNA_all).astype('float32') # decoder
    shared_ATAC_all = np.array(shared_ATAC_all).astype('float32') # decoder


    return cell_embed_RNA_all, cell_embed_ATAC_all,shared_RNA_all,shared_ATAC_all, RNA_id_all

def concerto_test_transformer_mutulsim_pretrain_supervised(task: str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None, saved_weight_path_regress=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'epoch_regress': 50, 'lr': 1e-4, 'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    num_classes = 22
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    output = encode_network_pretrain.layers[-1].output
    #output = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output)
    cls_network = tf.keras.Model(encode_network_pretrain.input, outputs=output)
    output_1 = embedding_encoder_pretrain.layers[-1].output
    #output_1 = tf.keras.layers.Dense(num_classes, activation='softmax', name='CLS')(output_1)
    cls_network_1 = tf.keras.Model(embedding_encoder_pretrain.input, outputs=output_1)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    decode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_epoch{epoch}.h5')
    embedding_decoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_embedding_epoch{epoch}.h5')
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5')
    cls_network.load_weights(saved_weight_path_pretrain + f'weight_cls_epoch{epoch}.h5',by_name=True)
    cls_network_1.load_weights(saved_weight_path_pretrain + f'weight_cls_1_epoch{epoch}.h5',by_name=True)
    print('load saved weight')

    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    shared_RNA_all = []
    shared_ATAC_all = []
    RNA_id_all = []

    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi_supervised([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi_supervised([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 20000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        shared_RNA = np.zeros((feature_len, dim))
        shared_ATAC = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,source_label_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,source_label_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                # z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                # z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                ######################################################################################
                z1 = cls_network([source_features_RNA, source_values_RNA])
                z1_1 = cls_network_1([source_features_ATAC, source_values_ATAC])
                #######################################################################################
                z2 = decode_network_pretrain(source_values_RNA)
                z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
                encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)
                decoder_output_RNA = tf.nn.l2_normalize(z2, axis=-1)
                decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
                cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
                shared_RNA[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_RNA
                shared_ATAC[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        shared_RNA_all.extend(shared_RNA[:all_samples])
        shared_ATAC_all.extend(shared_ATAC[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])


    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32') # encoder
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32') # encoder
    shared_RNA_all = np.array(shared_RNA_all).astype('float32') # decoder
    shared_ATAC_all = np.array(shared_ATAC_all).astype('float32') # decoder


    return cell_embed_RNA_all, cell_embed_ATAC_all,shared_RNA_all,shared_ATAC_all, RNA_id_all

def concerto_test_transformer_mutulsim_pretrain_EBCLR(task: str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'epoch_regress': 50, 'lr': 1e-4, 'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)


    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    #decode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_epoch{epoch}.h5')
    #embedding_decoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_embedding_epoch{epoch}.h5')
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5')

    print('load saved weight')

    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    shared_RNA_all = []
    shared_ATAC_all = []
    RNA_id_all = []

    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        shared_RNA = np.zeros((feature_len, dim))
        shared_ATAC = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                #z1 = decode_network_pretrain(source_values_RNA)
                #z1_1 = embedding_decoder_pretrain(source_values_ATAC)
                encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
                encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)
                #decoder_output_RNA = tf.nn.l2_normalize(z2, axis=-1)
                #decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
                cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
                #shared_RNA[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_RNA
                #shared_ATAC[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        #shared_RNA_all.extend(shared_RNA[:all_samples])
        #shared_ATAC_all.extend(shared_ATAC[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])


    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32') # encoder
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32') # encoder
    #shared_RNA_all = np.array(shared_RNA_all).astype('float32') # decoder
    #shared_ATAC_all = np.array(shared_ATAC_all).astype('float32') # decoder


    return cell_embed_RNA_all, cell_embed_ATAC_all, RNA_id_all

def concerto_train_multimodal_CLS_ITM(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path)
    f = np.load(saved_embed_path)
    peak_embed_ = f['peak_embed']
    print('peak_embed_ shape',peak_embed_.shape)
    peak_embed = np.squeeze(peak_embed_,axis=1)
    print('peak_embed shape', peak_embed.shape)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_cls(vocab_size=vocab_size_ATAC)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    Add_enc_1 = EncoderHead_add()
    Add_enc_2 = EncoderHead_add()
    ITM_head = EncoderHead_ITM(hidden_size = 1)
    #final_enc = scbasset_model(units_gene=vocab_size_RNA,flatten=False,globalpool=True)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    rna_atac_ss_loss = tf.keras.metrics.Mean(name='RNA_ATAC_ss_loss')
    rna_regress_loss = tf.keras.metrics.Mean(name='regress_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    #CE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            rna_atac_ss_loss.reset_states()
            rna_regress_loss.reset_states()
            itm_loss.reset_states()
            train_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    # ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    # mu_1 = mu_enc(z1)
                    # var_1 = tf.exp(var_enc(z1))
                    # KL_loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    encoder_output = Add_enc_1([z1,z1_1]) # teacher
                    decoder_output = Add_enc_2([z2, z2_1]) # student
                    #ssl_loss = simclr_loss(encoder_output, decoder_output, temperature=0.1)
                    ssl_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(encoder_output, decoder_output, temperature=0.1)
                    #################################### cross attention ###################################################
                    cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    cls_embed_gene = logits[:, 0, :]
                    ssl_cls_loss = simclr_loss(cls_embed_peak,cls_embed_gene, temperature=0.1)

                    #################################### ITM  #########################################################
                    weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    x = tf.linalg.diag_part(weights_t2t01)
                    matrix = tf.linalg.diag(x)
                    weights_t2t01 = weights_t2t01 - matrix
                    x = tf.linalg.diag_part(weights_t2t10)
                    matrix = tf.linalg.diag(x)
                    weights_t2t10 = weights_t2t10 - matrix
                    # select a negative teacher:0 for each student:1
                    cell_gene_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t10, 1)
                    for b in neg_idx:
                        cell_gene_embed_neg.append(cell_gene_embed[b[0]])

                    cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)

                    # select a negative student:1 for each teacher:0
                    cell_peak_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t01, 1)
                    for b in neg_idx:
                        cell_peak_embed_neg.append(cell_peak_embed[b[0]])

                    cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    cell_peak_embed_neg_, cell_gene_embed_neg_,logits_neg = model([cell_peak_embed_all, cell_gene_embed_all])
                    vl_embeddings = tf.concat([logits[:,0,:],logits_neg[:,0,:]],0)
                    vl_output = ITM_head(vl_embeddings)
                    ITM_label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    ITM_loss = CE_loss(ITM_label,vl_output)

                    ##################################### regression task #####################################
                    # cell_gene_embed_flatten = tf.keras.layers.GlobalAveragePooling1D()(cell_gene_embed_1)
                    # output = final_enc(cell_gene_embed_pos)
                    #################################### loss function ###################################################
                    #KL_loss_1 = tf.keras.losses.kullback_leibler_divergence(source_values_RNA,output)
                    #h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
                    #KL_loss_1 = h(source_values_RNA,output)
                    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    #     labels=source_values_RNA, logits=output)

                    # regress_loss = CE_loss(source_values_RNA,output)
                    ################################### total loss #########################################################
                    loss = ssl_loss + ssl_cls_loss + ITM_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    rna_atac_ss_loss(ssl_cls_loss)
                    itm_loss(ITM_loss)
                    #rna_regress_loss(regress_loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             Add_enc_1.trainable_variables,
                             Add_enc_2.trainable_variables,
                             #final_enc.trainable_variables,
                             model.trainable_variables,
                             ITM_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                # train_accuracy(source_values_RNA, output)
                # train_ROC(source_values_RNA, output)
                # train_PR(source_values_RNA, output)
                if step > 0 and step % 50 == 0:
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, CE_loss:{:0.3f}, Accuracy:{},ROC:{},PR:{}'
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, itm loss:{:0.3f}, cls_ss_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          itm_loss.result(),
                                          rna_atac_ss_loss.result()
                                          #rna_regress_loss.result(),
                                          #train_accuracy.result(),
                                          #train_ROC.result(),
                                          #train_PR.result(),
                                          ))
                    # print('pred:',output[:2,:20])
                    # print('gt:',source_values_RNA[:2,:20])

        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        model.save_weights(
            os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))
        ITM_head.save_weights(os.path.join(weight_path, 'ITM_head_epoch{}.h5'.format(str(epoch + 1))))
        # final_enc.save_weights(
        #     os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_1.save_weights(os.path.join(weight_path, 'weight_encoder_Add_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_2.save_weights(os.path.join(weight_path, 'weight_decoder_Add_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')

def concerto_train_multimodal_newCLS_ITM(RNA_tf_path: str, ATAC_tf_path: str, weight_path: str,saved_weight_path:str,saved_embed_path:str, super_parameters=None):
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 3, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    encode_network_pretrain.load_weights(saved_weight_path)
    f = np.load(saved_embed_path)
    peak_embed_ = f['peak_embed']
    print('peak_embed_ shape',peak_embed_.shape)
    peak_embed = np.squeeze(peak_embed_,axis=1)
    print('peak_embed shape', peak_embed.shape)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = peak_embed,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_cls(vocab_size=vocab_size_ATAC)
    mu_enc = EncoderHead()
    var_enc = EncoderHead()
    Add_enc_1 = EncoderHead_add()
    Add_enc_2 = EncoderHead_add()
    ITM_head = EncoderHead_ITM(hidden_size = 1)
    #final_enc = scbasset_model(units_gene=vocab_size_RNA,flatten=False,globalpool=True)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_ss_loss = tf.keras.metrics.Mean(name='RNA_ss_loss')
    rna_atac_ss_loss = tf.keras.metrics.Mean(name='RNA_ATAC_ss_loss')
    rna_regress_loss = tf.keras.metrics.Mean(name='regress_loss')
    itm_loss = tf.keras.metrics.Mean(name='ITM_loss')
    total_update_steps = 300 * super_parameters['epoch_pretrain']
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(super_parameters['lr'], total_update_steps,
                                                                super_parameters['lr'] * 1e-2, power=1)
    opt_simclr = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.95, beta_2=0.9995) #scbasset
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    CE_loss = tf.keras.losses.BinaryCrossentropy()
    #CE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                               batch_size=super_parameters['batch_size'],
                                                               is_training=False,
                                                               data_augment=False,
                                                               shuffle_size=10000,
                                                               )
            train_loss.reset_states()
            rna_ss_loss.reset_states()
            rna_atac_ss_loss.reset_states()
            rna_regress_loss.reset_states()
            itm_loss.reset_states()
            train_accuracy.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    #################################### RNA pretrain ####################################################
                    z1,cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                    z2 = decode_network_pretrain(source_values_RNA)
                    # ssl_loss = simclr_loss(z1, z2, temperature=0.1)
                    # mu_1 = mu_enc(z1)
                    # var_1 = tf.exp(var_enc(z1))
                    # KL_loss = tf.keras.losses.kullback_leibler_divergence(mu_1, var_1)
                    #################################### ATAC pretrain ####################################################
                    z1_1,cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                    #################################### contrastive loss ####################################
                    encoder_output = Add_enc_1([z1,z1_1]) # teacher
                    decoder_output = Add_enc_2([z2, z2_1]) # student
                    ssl_loss = simclr_loss(encoder_output, decoder_output, temperature=0.1)
                    #ssl_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(encoder_output, decoder_output, temperature=0.1)
                    #################################### cross attention ###################################################
                    cell_peak_embed_pos, cell_gene_embed_pos,logits = model([cell_peak_embed, cell_gene_embed])
                    cls_embed_peak = cell_peak_embed_pos[:,0,:]
                    cls_embed_gene = cell_gene_embed_pos[:, 0, :]
                    ssl_cls_loss, sim_t2t01,sim_t2t10 = simclr_loss_1(cls_embed_peak,cls_embed_gene, temperature=0.1)

                    #################################### ITM  #########################################################
                    weights_t2t01 = tf.nn.softmax(sim_t2t01)
                    weights_t2t10 = tf.nn.softmax(sim_t2t10)
                    x = tf.linalg.diag_part(weights_t2t01)
                    matrix = tf.linalg.diag(x)
                    weights_t2t01 = weights_t2t01 - matrix
                    x = tf.linalg.diag_part(weights_t2t10)
                    matrix = tf.linalg.diag(x)
                    weights_t2t10 = weights_t2t10 - matrix
                    # select a negative teacher:0 for each student:1
                    cell_gene_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t10, 1)
                    for b in neg_idx:
                        cell_gene_embed_neg.append(cell_gene_embed[b[0]])

                    cell_gene_embed_neg = tf.stack(cell_gene_embed_neg, axis=0)

                    # select a negative student:1 for each teacher:0
                    cell_peak_embed_neg = []
                    neg_idx = tf.random.categorical(weights_t2t01, 1)
                    for b in neg_idx:
                        cell_peak_embed_neg.append(cell_peak_embed[b[0]])

                    cell_peak_embed_neg = tf.stack(cell_peak_embed_neg, axis=0)
                    cell_gene_embed_all = tf.concat([cell_gene_embed, cell_gene_embed_neg],axis=0)
                    cell_peak_embed_all = tf.concat([cell_peak_embed, cell_peak_embed_neg], axis=0)
                    cell_peak_embed_neg_, cell_gene_embed_neg_,logits_neg = model([cell_peak_embed_all, cell_gene_embed_all])
                    vl_embeddings = tf.concat([logits[:,0,:],logits_neg[:,0,:]],0)
                    vl_output = ITM_head(vl_embeddings)
                    ITM_label = tf.concat([tf.ones([super_parameters['batch_size']]),tf.zeros([2*super_parameters['batch_size']])],0)
                    ITM_loss = CE_loss(ITM_label,vl_output)

                    ##################################### regression task #####################################
                    # cell_gene_embed_flatten = tf.keras.layers.GlobalAveragePooling1D()(cell_gene_embed_1)
                    # output = final_enc(cell_gene_embed_pos)
                    #################################### loss function ###################################################
                    #KL_loss_1 = tf.keras.losses.kullback_leibler_divergence(source_values_RNA,output)
                    #h = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
                    #KL_loss_1 = h(source_values_RNA,output)
                    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                    #     labels=source_values_RNA, logits=output)

                    # regress_loss = CE_loss(source_values_RNA,output)
                    ################################### total loss #########################################################
                    loss = ssl_loss + ssl_cls_loss + ITM_loss
                    train_loss(loss)
                    rna_ss_loss(ssl_loss)
                    rna_atac_ss_loss(ssl_cls_loss)
                    itm_loss(ITM_loss)
                    #rna_regress_loss(regress_loss)

                variables = [encode_network_pretrain.trainable_variables,
                             decode_network_pretrain.trainable_variables,
                             embedding_encoder_pretrain.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             Add_enc_1.trainable_variables,
                             Add_enc_2.trainable_variables,
                             #final_enc.trainable_variables,
                             model.trainable_variables,
                             ITM_head.trainable_variables
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    opt_simclr.apply_gradients(zip(grad, var))

                # train_accuracy(source_values_RNA, output)
                # train_ROC(source_values_RNA, output)
                # train_PR(source_values_RNA, output)
                if step > 0 and step % 50 == 0:
                    #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, CE_loss:{:0.3f}, Accuracy:{},ROC:{},PR:{}'
                    template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, itm loss:{:0.3f}, cls_ss_loss:{:0.3f}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          rna_ss_loss.result(),
                                          itm_loss.result(),
                                          rna_atac_ss_loss.result()
                                          #rna_regress_loss.result(),
                                          #train_accuracy.result(),
                                          #train_ROC.result(),
                                          #train_PR.result(),
                                          ))
                    # print('pred:',output[:2,:20])
                    # print('gt:',source_values_RNA[:2,:20])

        encode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_epoch{}.h5'.format(str(epoch + 1))))
        decode_network_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_epoch{}.h5'.format(str(epoch + 1))))
        embedding_encoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_encoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        model.save_weights(
            os.path.join(weight_path, 'weight_transformer_epoch{}.h5'.format(str(epoch + 1))))
        ITM_head.save_weights(os.path.join(weight_path, 'ITM_head_epoch{}.h5'.format(str(epoch + 1))))
        # final_enc.save_weights(
        #     os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_1.save_weights(os.path.join(weight_path, 'weight_encoder_Add_epoch{}.h5'.format(str(epoch + 1))))
        Add_enc_2.save_weights(os.path.join(weight_path, 'weight_decoder_Add_epoch{}.h5'.format(str(epoch + 1))))

    return print('finished')


def concerto_test_multimodal(task: str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,
                             super_parameters=None,
                             saved_weight_path_pretrain=None, saved_weight_path_regress=None):
    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'epoch_regress': 50, 'lr': 1e-4, 'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=True,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    decode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                                     mult_feature_names=['RNA'],
                                                                     embedding_dims=128,
                                                                     include_attention=False,
                                                                     drop_rate=super_parameters['drop_rate'],
                                                                     head_1=128,
                                                                     head_2=128,
                                                                     head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
        embedding_matrix=None,
        multi_max_features=[vocab_size_ATAC],
        mult_feature_names=['ATAC'],
        embedding_dims=128,
        include_attention=False,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)

    # model = Transformer_model(vocab_size=vocab_size_ATAC)
    # mu_enc = EncoderHead()
    # var_enc = EncoderHead()
    Add_enc_1 = EncoderHead_add()
    Add_enc_2 = EncoderHead_add()
    # final_enc = EncoderHead_1(hidden_size = vocab_size_RNA)
    # final_enc = scbasset_model(units_gene=vocab_size_RNA, flatten=False, globalpool=True)
    final_enc = EncoderHead_1(hidden_size1=1000, hidden_size2=2000)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    decode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_epoch{epoch}.h5')
    embedding_decoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_decoder_embedding_epoch{epoch}.h5',
                                            by_name=True)
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5',
                                            by_name=True)

    print('load saved weight')

    cell_embed_RNA_all = []
    cell_embed_ATAC_all = []
    cell_embed_RNA_decoder_all = []
    cell_embed_ATAC_decoder_all = []
    # predict_RNA_count_onlyatac_all = []
    # predict_RNA_count_multi_all = []
    multi_embed_all_encoder = []
    multi_embed_all_decoder = []
    RNA_id_all = []

    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=False,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):

            if step == 0:
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                encoder_output = Add_enc_1([z1, z1_1])
                z2 = decode_network_pretrain(source_values_RNA)
                z2_1 = embedding_decoder_pretrain(source_values_ATAC)
                decoder_output = Add_enc_2([z2, z2_1])

                break

        # model.load_weights(saved_weight_path_pretrain + f'weight_transformer_epoch{epoch}.h5', by_name=True)
        # final_enc.load_weights(saved_weight_path_regress + f'weight_project_epoch{epoch_regress}.h5', by_name=True)
        Add_enc_1.load_weights(saved_weight_path_pretrain + f'weight_encoder_Add_epoch{epoch}.h5', by_name=True)
        Add_enc_2.load_weights(saved_weight_path_pretrain + f'weight_decoder_Add_epoch{epoch}.h5', by_name=True)
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        # predict_RNA_count_onlyatac = np.zeros((feature_len, vocab_size_RNA))
        # predict_RNA_count_multi = np.zeros((feature_len, vocab_size_RNA))
        cell_embed_RNA = np.zeros((feature_len, dim))
        cell_embed_ATAC = np.zeros((feature_len, dim))
        cell_embed_RNA_decoder = np.zeros((feature_len, dim))
        cell_embed_ATAC_decoder = np.zeros((feature_len, dim))
        multi_embed_encoder = np.zeros((feature_len, dim))
        multi_embed_decoder = np.zeros((feature_len, dim))

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                z2 = decode_network_pretrain(source_values_RNA)
                z2_1 = embedding_decoder_pretrain(source_values_ATAC)

                encoder_output_RNA = tf.nn.l2_normalize(z1, axis=-1)
                encoder_output_ATAC = tf.nn.l2_normalize(z1_1, axis=-1)

                decoder_output_RNA = tf.nn.l2_normalize(z2, axis=-1)
                decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)

                encoder_output = Add_enc_1([z1, z1_1])
                encoder_output = tf.nn.l2_normalize(encoder_output, axis=-1)
                decoder_output = Add_enc_2([z2, z2_1])
                decoder_output = tf.nn.l2_normalize(decoder_output, axis=-1)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                # predict_RNA_count_onlyatac[all_samples:all_samples + len(source_id_RNA), :] = output_onlyatac
                # predict_RNA_count_multi[all_samples:all_samples + len(source_id_RNA), :] = output_multi
                cell_embed_ATAC[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_ATAC
                cell_embed_RNA[all_samples:all_samples + len(source_id_RNA), :] = encoder_output_RNA
                cell_embed_RNA_decoder[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_RNA
                cell_embed_ATAC_decoder[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
                multi_embed_encoder[all_samples:all_samples + len(source_id_RNA), :] = encoder_output
                multi_embed_decoder[all_samples:all_samples + len(source_id_RNA), :] = decoder_output

                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))

        cell_embed_RNA_all.extend(cell_embed_RNA[:all_samples])
        cell_embed_ATAC_all.extend(cell_embed_ATAC[:all_samples])
        cell_embed_RNA_decoder_all.extend(cell_embed_RNA_decoder[:all_samples])
        cell_embed_ATAC_decoder_all.extend(cell_embed_ATAC_decoder[:all_samples])
        multi_embed_all_encoder.extend(multi_embed_encoder[:all_samples])
        multi_embed_all_decoder.extend(multi_embed_decoder[:all_samples])
        # predict_RNA_count_onlyatac_all.extend(predict_RNA_count_onlyatac[:all_samples])
        # predict_RNA_count_multi_all.extend(predict_RNA_count_multi[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        # cosine_peak_gene_all.extend(cosine_peak_gene[:all_samples])

    cell_embed_RNA_all = np.array(cell_embed_RNA_all).astype('float32')
    cell_embed_ATAC_all = np.array(cell_embed_ATAC_all).astype('float32')
    cell_embed_RNA_decoder_all = np.array(cell_embed_RNA_decoder_all).astype('float32')
    cell_embed_ATAC_decoder_all = np.array(cell_embed_ATAC_decoder_all).astype('float32')
    multi_embed_all_encoder = np.array(multi_embed_all_encoder).astype('float32')
    multi_embed_all_decoder = np.array(multi_embed_all_decoder).astype('float32')
    # cosine_peak_gene_all = np.array(cosine_peak_gene_all).astype('float32')
    # predict_RNA_count_onlyatac_all = np.array(predict_RNA_count_onlyatac_all).astype('float32')
    # predict_RNA_count_multi_all = np.array(predict_RNA_count_multi_all).astype('float32')
    # attention_weight = {'attention_output_RNA': attention_output_RNA_all,
    #                     'attention_output_Protein': attention_output_Protein_all}
    # np.savez_compressed('./multi_attention.npz', **attention_weight)
    return cell_embed_RNA_all, cell_embed_ATAC_all, cell_embed_RNA_decoder_all, \
           cell_embed_ATAC_decoder_all, multi_embed_all_encoder, multi_embed_all_decoder, RNA_id_all



def attune_test_regulatory(RNA_tf_path: str, ATAC_tf_path: str,mask_path:str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path_pretrain=None, saved_weight_path=None):

    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50,'epoch_transformer':50, 'lr': 1e-4,'drop_rate': 0.1}

    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_transformer = super_parameters['epoch_transformer']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_extract(vocab_size=vocab_size_ATAC,attention_mask_path=mask_path)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5', by_name=True)
    print('load saved weight')

    RNA_id_all = []
    cross_attention_1 = []
    cross_attention_2 = []
    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=True,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):

            if step == 0:
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                cell_gene_embed_1,_ = model([cell_peak_embed, cell_gene_embed])
                break
        model.load_weights(saved_weight_path + f'weight_transformer_epoch{epoch_transformer}.h5', by_name=True)
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):
            if all_samples >= feature_len:
                break

            z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
            z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
            cell_gene_embed_pos,crossattention = model([cell_peak_embed, cell_gene_embed])
            tgt_src_attention = crossattention[0]['tgt_src_attention'].numpy()[:,0]
            print('tgt_src_attention shape',tgt_src_attention.shape)
            tgt_src_attention_1 = tgt_src_attention[:,0,:]
            tgt_src_attention_2 = tgt_src_attention[:, :, 0]
            print('tgt_src_attention_1 shape', tgt_src_attention_1.shape)
            print('tgt_src_attention_2 shape', tgt_src_attention_2.shape)
            ######################################################################
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            cross_attention_1.extend(tgt_src_attention_1)
            cross_attention_2.extend(tgt_src_attention_2)
            all_samples += len(source_id_RNA)
            print('all_samples num:{}'.format(all_samples))

        RNA_id_all.extend(RNA_id[:all_samples])

    return  RNA_id_all,cross_attention_1, cross_attention_2

def concerto_test_multimodal_potential_extract_1(task:str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path_pretrain=None):

    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50,'epoch_regress':50, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress'] # transformer epoch
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)


    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_extract(vocab_size=vocab_size_ATAC)
    #Add_enc_1 = EncoderHead_add()

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    #embedding_decoder_pretrain.load_weights(saved_weight_path_regress + f'weight_decoder_embedding_epoch{epoch_regress}.h5', by_name=True)
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5', by_name=True)

    print('load saved weight')

    f = np.load('./result/SHARE_seurat_TAC_filter_100kpeak_emb128_0921/cls_crossattention/share_top10_aw_index.npz')
    gene_index = list(f['gene'])
    peak_index = list(f['peak'])

    RNA_id_all = []
    cross_attention_1 = []
    cross_attention_2 = []
    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=True,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):

            if step == 0:
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                #encoder_output = Add_enc_1([z1, z1_1])
                cell_gene_embed_1,_ = model([cell_peak_embed, cell_gene_embed])

                break

        model.load_weights(saved_weight_path_pretrain + f'weight_transformer_epoch{epoch_regress}.h5', by_name=True)
        #Add_enc_1.load_weights(saved_weight_path_pretrain + f'weight_encoder_Add_epoch{epoch}.h5', by_name=True)
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break

                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                #z2_1 = embedding_decoder_pretrain(source_values_ATAC)

                cell_gene_embed_pos,crossattention = model([cell_peak_embed, cell_gene_embed])
                #print(crossattention)
                # tgt_src_attention = [
                #     crossattention['layer_%d' % i]['tgt_src_attention'].numpy()[:, 0]
                #     for i in range(stack_size)]
                tgt_src_attention = crossattention[0]['tgt_src_attention'].numpy()[:,0]
                print('tgt_src_attention shape',tgt_src_attention.shape)
                tgt_src_attention = tgt_src_attention[:,1:,1:]
                tgt_src_attention_2 = tgt_src_attention[:,gene_index,peak_index]
                print('tgt_src_attention_2 shape', tgt_src_attention_2.shape)
                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cross_attention_2.extend(tgt_src_attention_2)
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))


        RNA_id_all.extend(RNA_id[:all_samples])


    return  RNA_id_all, cross_attention_2

def concerto_test_multimodal_potential_extract(task:str, RNA_tf_path: str, ATAC_tf_path: str, n_cells_for_sample=None,super_parameters=None,
                             saved_weight_path_pretrain=None):

    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50,'epoch_regress':50, 'lr': 1e-4,'drop_rate': 0.1}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    batch_size = super_parameters['batch_size']
    epoch = super_parameters['epoch_pretrain']
    epoch_regress = super_parameters['epoch_regress'] # transformer epoch
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)


    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    model = Transformer_model_extract(vocab_size=vocab_size_ATAC)
    #Add_enc_1 = EncoderHead_add()

    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    encode_network_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_epoch{epoch}.h5')
    #embedding_decoder_pretrain.load_weights(saved_weight_path_regress + f'weight_decoder_embedding_epoch{epoch_regress}.h5', by_name=True)
    embedding_encoder_pretrain.load_weights(saved_weight_path_pretrain + f'weight_encoder_embedding_epoch{epoch}.h5', by_name=True)

    print('load saved weight')

    RNA_id_all = []
    cross_attention_1 = []
    cross_attention_2 = []
    #cross_attention_onegene = []
    for RNA_file, ATAC_file in zip(train_source_list_RNA, train_source_list_ATAC):
        print(RNA_file)
        print(ATAC_file)
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=super_parameters['batch_size'],
                                                       is_training=True,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                        batch_size=super_parameters['batch_size'],
                                                        is_training=True,
                                                        data_augment=False,
                                                        shuffle_size=10000,
                                                        )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(train_db_RNA, train_db_ATAC)):

            if step == 0:
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                #encoder_output = Add_enc_1([z1, z1_1])
                cell_gene_embed_1,_ = model([cell_peak_embed, cell_gene_embed])

                break

        model.load_weights(saved_weight_path_pretrain + f'weight_transformer_epoch{epoch_regress}.h5', by_name=True)
        #Add_enc_1.load_weights(saved_weight_path_pretrain + f'weight_encoder_Add_epoch{epoch}.h5', by_name=True)
        dim = 128
        if n_cells_for_sample is None:
            feature_len = 10000
        else:
            feature_len = n_cells_for_sample // batch_size * batch_size

        print('feature_len:', feature_len)

        RNA_id = []
        all_samples = 0
        if task == 'integration':
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                if all_samples >= feature_len:
                    break
                print('source_id_RNA:',source_id_RNA)
                print('source_id_ATAC:', source_id_ATAC)
                z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA])
                z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC])
                #z2_1 = embedding_decoder_pretrain(source_values_ATAC)

                cell_gene_embed_pos,crossattention = model([cell_peak_embed, cell_gene_embed])
                #print(crossattention)
                # tgt_src_attention = [
                #     crossattention['layer_%d' % i]['tgt_src_attention'].numpy()[:, 0]
                #     for i in range(stack_size)]
                tgt_src_attention = crossattention[0]['tgt_src_attention'].numpy()[:,0]
                print('tgt_src_attention shape',tgt_src_attention.shape)
                # tgt_src_attention_1 = tgt_src_attention[:,0,:]
                # tgt_src_attention_2 = tgt_src_attention[:, :, 0]
                tgt_src_attention_wocls = tgt_src_attention[:,1:,1:]
                print('tgt_src_attention_wocls shape', tgt_src_attention_wocls.shape)
                tgt_src_attention_wocls_one_gene = tgt_src_attention_wocls[:,0,:]
                print('tgt_src_attention_wocls_one_gene shape', tgt_src_attention_wocls_one_gene.shape)
                tgt_src_attention_wocls_1 = np.mean(tgt_src_attention_wocls,axis=2)
                print('tgt_src_attention_wocls_1 shape',tgt_src_attention_wocls_1.shape)
                tgt_src_attention_wocls_2 = np.mean(tgt_src_attention_wocls_1, axis=0)
                print('tgt_src_attention_wocls_2 shape', tgt_src_attention_wocls_2.shape)
                tgt_src_attention_wocls_sub = tgt_src_attention_wocls_1 - tgt_src_attention_wocls_2
                print('tgt_src_attention_wocls_sub shape', tgt_src_attention_wocls_sub.shape)

                ######################################################################
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                cross_attention_1.extend(tgt_src_attention_wocls_sub)
                cross_attention_2.extend(tgt_src_attention_wocls_1)
                #cross_attention_onegene.extend(tgt_src_attention_wocls_one_gene)
                all_samples += len(source_id_RNA)
                print('all_samples num:{}'.format(all_samples))
                print('cross_attention_1 len:',len(cross_attention_1))


        RNA_id_all.extend(RNA_id[:all_samples])


    return  RNA_id_all,cross_attention_1,cross_attention_2


def attune_predict(RNA_tf_path: str, ATAC_tf_path: str,RNA_tf_path_test: str, ATAC_tf_path_test: str,weight_path:str,saved_result_path:str,super_parameters=None,
                             saved_weight_path=None):

    def plot_cor_pergene(x, y, logscale, normlib):
        """
        return pearson correlation coefficient for each gene: pearson_r_list
        flattened pearson correlation coefficient: pearson_r_flatten
        number of positive values in the true profile for each gene
        """
        assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
        x = np.asarray(x)
        y = np.asarray(y)
        if normlib == 'norm':
            ## compare with normalized true profile
            lib = x.sum(axis=1, keepdims=True)
            x = x / lib
        if logscale:
            x = np.log1p(x)
            y = np.log1p(y)
        pearson_r_flatten, pearson_p_flatten = scipy.stats.pearsonr(x.flatten(), y.flatten())
        pearson_r_list = []
        npos = []
        for i in range(x.shape[1]):
            npos.append(np.sum(x[:, i] > 0))
            if not np.all(x[:, i] == 0) and not np.all(y[:, i] == 0):
                pearson_r, pearson_p = scipy.stats.pearsonr(x[:, i], y[:, i])
                # spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
                pearson_r_list.append(pearson_r)
            else:
                pearson_r_list.append(np.nan)

        return np.array(pearson_r_list), pearson_r_flatten, np.array(npos)

    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch': 50, 'lr': 1e-4,'drop_rate': 0.1,'saved_epoch':3}

    if not os.path.exists(weight_path):
        os.makedirs(weight_path)
    saved_epoch = super_parameters['saved_epoch']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
    final_enc = EncoderHead_1(hidden_size1=1000,hidden_size2=2000)
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))

    tf_list_2 = [f for f in os.listdir(os.path.join(RNA_tf_path_test)) if 'tfrecord' in f]
    test_source_list_RNA = []
    test_source_list_ATAC = []
    for i in tf_list_2:
        test_source_list_RNA.append(os.path.join(RNA_tf_path_test, i))
        test_source_list_ATAC.append(os.path.join(ATAC_tf_path_test, i))

    embedding_decoder_pretrain.load_weights(saved_weight_path + f'weight_decoder_embedding_epoch{saved_epoch}.h5', by_name=True)
    print('load saved weight')
    optimizer = tf.keras.optimizers.Adam(learning_rate=super_parameters['lr'])
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
    CE_loss = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    for epoch in range(super_parameters['epoch']):
        for RNA_file, ATAC_file,RNA_file_test,ATAC_file_test in zip(train_source_list_RNA, train_source_list_ATAC,test_source_list_RNA,test_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                            batch_size=super_parameters['batch_size'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            )
            train_db_RNA_test = create_classifier_dataset_multi([RNA_file_test],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC_test = create_classifier_dataset_multi([ATAC_file_test],
                                                            batch_size=super_parameters['batch_size'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            )
            train_loss.reset_states()
            step = 0
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA, train_db_ATAC)):
                step += 1

                with tf.GradientTape() as tape:
                    z2_1 = embedding_decoder_pretrain(source_values_ATAC,training=True)
                    output = final_enc(z2_1,training=True)
                    #################################### loss function ########################
                    loss = CE_loss(source_values_RNA,output)
                    train_loss(loss)
                variables = [final_enc.trainable_variables,
                             embedding_decoder_pretrain.trainable_variables,
                             ]
                grads = tape.gradient(loss, variables)
                for grad, var in zip(grads, variables):
                    optimizer.apply_gradients(zip(grad, var))

                train_accuracy(source_values_RNA, output)
                if step > 0 and step % 50 == 0:
                    template = 'Epoch{}, step{}, loss:{:0.3f}, mean_squared_error:{}'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result(),
                                          train_accuracy.result(),
                                          ))

                    output_array = np.array(output)
                    source_values_RNA_array = np.array(source_values_RNA)
                    pearson_r_list, pearson_r_flatten, npos = plot_cor_pergene(source_values_RNA_array,
                                                                               output_array,
                                                                               logscale=False, normlib=False)
                    pearson_r_mean = np.mean(pearson_r_list)
                    print('train pearson_r_mean:',pearson_r_mean)
                    print('train pearson_r_flatten:', pearson_r_flatten)
            ######################################### test #######################################################
            feature_len = 50000
            RNA_id = []
            predict_RNA_count_onlyatac_all = []
            RNA_id_all = []
            RNA_count_all = []
            atac_cell_embed_all = []
            all_samples = 0
            predict_RNA_count_onlyatac = np.zeros((feature_len, vocab_size_RNA))
            RNA_count = np.zeros((feature_len, vocab_size_RNA))
            atac_cell_embed = np.zeros((feature_len, 128))
            for (source_features_RNA, source_values_RNA,
                 source_batch_RNA, source_id_RNA), \
                (source_features_ATAC, source_values_ATAC,
                 source_batch_ATAC, source_id_ATAC) \
                    in (zip(train_db_RNA_test, train_db_ATAC_test)):

                if all_samples >= feature_len:
                    break

                z2_1 = embedding_decoder_pretrain(source_values_ATAC,training=False)
                output_onlyatac = final_enc(z2_1,training=False)
                decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)
                RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
                predict_RNA_count_onlyatac[all_samples:all_samples + len(source_id_RNA), :] = output_onlyatac
                RNA_count[all_samples:all_samples + len(source_id_RNA), :] = source_values_RNA
                atac_cell_embed[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
                all_samples += len(source_id_RNA)

            predict_RNA_count_onlyatac_all.extend(predict_RNA_count_onlyatac[:all_samples])
            RNA_id_all.extend(RNA_id[:all_samples])
            RNA_count_all.extend(RNA_count[:all_samples])
            atac_cell_embed_all.extend(atac_cell_embed[:all_samples])
            predict_RNA_count_onlyatac_all = np.array(predict_RNA_count_onlyatac_all).astype('float32')
            RNA_count_all = np.array(RNA_count_all)
            atac_cell_embed_all = np.array(atac_cell_embed_all)
            pearson_r_list, pearson_r_flatten, npos = plot_cor_pergene(RNA_count_all,
                                                                       predict_RNA_count_onlyatac_all,
                                                                       logscale=False, normlib=False)
            pearson_r_mean = np.mean(pearson_r_list)
            mse = np.sqrt(mean_squared_error(RNA_count_all, predict_RNA_count_onlyatac_all))
            print('test pearson_r_mean:', pearson_r_mean)
            print('test pearson_r_flatten:', pearson_r_flatten)
            print('test mse:', mse)
            f = {
                'predict_RNA_count_onlyatac': predict_RNA_count_onlyatac_all,
                'RNA_id': RNA_id_all,
                'atac_cell_embed':atac_cell_embed_all
                }
            np.savez_compressed(saved_result_path + f'/result_ep{epoch + 1}_m{mse}_p{pearson_r_mean}.npz', **f)

        final_enc.save_weights(
                os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))

    return print('predict finished')

def concerto_train_multimodal_regress_1(task:str, RNA_tf_path: str, ATAC_tf_path: str,RNA_tf_path_test: str, ATAC_tf_path_test: str,weight_path:str,saved_result_path:str,super_parameters=None,
                             saved_weight_path=None):

    def plot_cor_pergene(x, y, logscale, normlib):
        """
        return pearson correlation coefficient for each gene: pearson_r_list
        flattened pearson correlation coefficient: pearson_r_flatten
        number of positive values in the true profile for each gene
        """
        assert x.shape == y.shape, f"Mismatched shapes: {x.shape} {y.shape}"
        x = np.asarray(x)
        y = np.asarray(y)
        if normlib == 'norm':
            ## compare with normalized true profile
            lib = x.sum(axis=1, keepdims=True)
            x = x / lib
        if logscale:
            x = np.log1p(x)
            y = np.log1p(y)
        pearson_r_flatten, pearson_p_flatten = scipy.stats.pearsonr(x.flatten(), y.flatten())
        pearson_r_list = []
        npos = []
        for i in range(x.shape[1]):
            npos.append(np.sum(x[:, i] > 0))
            if not np.all(x[:, i] == 0) and not np.all(y[:, i] == 0):
                pearson_r, pearson_p = scipy.stats.pearsonr(x[:, i], y[:, i])
                # spearman_corr, spearman_p = scipy.stats.spearmanr(x, y)
                pearson_r_list.append(pearson_r)
            else:
                pearson_r_list.append(np.nan)

        return np.array(pearson_r_list), pearson_r_flatten, np.array(npos)




    if super_parameters is None:
        super_parameters = {'batch_size': 64, 'epoch_pretrain': 50, 'lr': 1e-4,'drop_rate': 0.1,'saved_epoch':3}
    # dirname = os.getcwd()
    # f = np.load(ref_tf_path + './vocab_size.npz')
    if not os.path.exists(weight_path):
        os.makedirs(weight_path)

    saved_epoch = super_parameters['saved_epoch']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(ATAC_tf_path, 'vocab_size.npz'))
    vocab_size_ATAC = int(f['vocab size'])
    encode_network_pretrain = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    embedding_encoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)


    embedding_decoder_pretrain = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=super_parameters['drop_rate'],
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

    # model = Transformer_model(vocab_size=vocab_size_ATAC)
    # mu_enc = EncoderHead()
    # var_enc = EncoderHead()
    # Add_enc_1 = EncoderHead_add()
    # Add_enc_2 = EncoderHead_add()
    # final_enc = scbasset_model(units_gene=vocab_size_RNA, flatten=False, globalpool=True)
    #final_enc = EncoderHead_1(hidden_size1=1000,hidden_size2=13431) #SHARE HVG+TF: 2591;NIPS openproblem predict:13431 # other:2000 hidden1:1000/NIPS openproblem predict:500
    final_enc = EncoderHead_2(hidden_size1=64, hidden_size2=250,hidden_size3 = 13431,dropout=0.8)

    # tf_list_1 = os.listdir(os.path.join(ref_tf_path))
    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_ATAC = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_ATAC.append(os.path.join(ATAC_tf_path, i))


    tf_list_2 = [f for f in os.listdir(os.path.join(RNA_tf_path_test)) if 'tfrecord' in f]
    test_source_list_RNA = []
    test_source_list_ATAC = []
    for i in tf_list_2:
        test_source_list_RNA.append(os.path.join(RNA_tf_path_test, i))
        test_source_list_ATAC.append(os.path.join(ATAC_tf_path_test, i))


    encode_network_pretrain.load_weights(saved_weight_path + f'weight_encoder_epoch{saved_epoch}.h5')
    embedding_decoder_pretrain.load_weights(saved_weight_path + f'weight_decoder_embedding_epoch{saved_epoch}.h5', by_name=True)
    embedding_encoder_pretrain.load_weights(saved_weight_path + f'weight_encoder_embedding_epoch{saved_epoch}.h5',
                                            by_name=True)
    print('load saved weight')


    optimizer = tf.keras.optimizers.Adam(learning_rate=super_parameters['lr']) #scbasset
    #train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')
    train_ROC = tf.keras.metrics.AUC(curve='ROC', )
    train_PR = tf.keras.metrics.AUC(curve='PR', )
    #CE_loss = tf.keras.losses.BinaryCrossentropy()
    #CE_loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    CE_loss = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    rna_regress_loss = tf.keras.metrics.Mean(name='regress_loss')
    test_db_RNA = create_classifier_dataset_multi(test_source_list_RNA,
                                                   batch_size=super_parameters['batch_size'],
                                                   is_training=True,
                                                   data_augment=False,
                                                   shuffle_size=10000,
                                                   )
    test_db_ATAC = create_classifier_dataset_multi(test_source_list_ATAC,
                                                    batch_size=super_parameters['batch_size'],
                                                    is_training=True,
                                                    data_augment=False,
                                                    shuffle_size=10000,
                                                    )
    for epoch in range(super_parameters['epoch_pretrain']):
        for RNA_file, ATAC_file,RNA_file_test,ATAC_file_test in zip(train_source_list_RNA, train_source_list_ATAC,test_source_list_RNA,test_source_list_ATAC):
            print(RNA_file)
            print(ATAC_file)
            train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                           batch_size=super_parameters['batch_size'],
                                                           is_training=True,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
            train_db_ATAC = create_classifier_dataset_multi([ATAC_file],
                                                            batch_size=super_parameters['batch_size'],
                                                            is_training=True,
                                                            data_augment=False,
                                                            shuffle_size=10000,
                                                            )

            train_loss.reset_states()
            rna_regress_loss.reset_states()


            if task == 'regress':
                step = 0
                for (source_features_RNA, source_values_RNA,
                     source_batch_RNA, source_id_RNA), \
                    (source_features_ATAC, source_values_ATAC,
                     source_batch_ATAC, source_id_ATAC) \
                        in (zip(train_db_RNA, train_db_ATAC)):
                    step += 1

                    with tf.GradientTape() as tape:
                        #################################### RNA pretrain ####################################################
                        #z1, cell_gene_embed = encode_network_pretrain([source_features_RNA, source_values_RNA],training=False)

                        #################################### ATAC pretrain ####################################################
                        #z1_1, cell_peak_embed = embedding_encoder_pretrain([source_features_ATAC, source_values_ATAC],training=True)
                        z2_1 = embedding_decoder_pretrain(source_values_ATAC,training=True)
                        #################################### cross attention ###################################################
                        #cell_gene_embed_pos = model([cell_peak_embed, cell_gene_embed],training=False)
                        ##################################### regression task #####################################
                        #cell_gene_embed_flatten = tf.keras.layers.GlobalAveragePooling1D()(cell_gene_embed_1)
                        #cell_peak = tf.math.l2_normalize(K.sum(cell_peak_embed, axis=-1), axis=-1)
                        output = final_enc(z2_1,training=True)
                        #################################### loss function ###################################################
                        regress_loss = CE_loss(source_values_RNA,output)
                        ################################### total loss #########################################################
                        loss = regress_loss
                        train_loss(loss)
                        #rna_ss_loss(ssl_loss)
                        #itm_loss(ITM_loss)
                        rna_regress_loss(loss)

                    variables = [final_enc.trainable_variables,
                                 embedding_decoder_pretrain.trainable_variables,
                                 ]
                    grads = tape.gradient(loss, variables)
                    for grad, var in zip(grads, variables):
                        optimizer.apply_gradients(zip(grad, var))

                    train_accuracy(source_values_RNA, output)
                    # train_ROC(source_values_RNA, output)
                    # train_PR(source_values_RNA, output)
                    if step > 0 and step % 50 == 0:
                        template = 'Epoch{}, step{}, regress_loss:{:0.3f}, mean_squared_error:{},ROC:{},PR:{}'
                        #template = 'Epoch{}, step{}, total loss:{:0.3f}, rna_atac_ss_loss:{:0.3f}, itm loss:{:0.3f}'
                        print(template.format(epoch + 1,
                                              str(step),
                                              #train_loss.result(),
                                              #rna_ss_loss.result(),
                                              #itm_loss.result()
                                              rna_regress_loss.result(),
                                              train_accuracy.result(),
                                              train_ROC.result(),
                                              train_PR.result(),
                                              ))
                        #print('pred:',output[0,:20])
                        #print('gt:',source_values_RNA[0,:20])
                        output_array = np.array(output)
                        source_values_RNA_array = np.array(source_values_RNA)
                        pearson_r_list, pearson_r_flatten, npos = plot_cor_pergene(source_values_RNA_array,
                                                                                   output_array,
                                                                                   logscale=False, normlib=False)
                        pearson_r_mean = np.mean(pearson_r_list)
                        print('train pearson_r_mean:',pearson_r_mean)
                        print('train pearson_r_flatten:', pearson_r_flatten)

        final_enc.save_weights(
                os.path.join(weight_path, 'weight_project_epoch{}.h5'.format(str(epoch + 1))))
        embedding_decoder_pretrain.save_weights(
            os.path.join(weight_path, 'weight_decoder_embedding_epoch{}.h5'.format(str(epoch + 1))))
        ######################################### test #######################################################
        feature_len = 100000
        RNA_id = []
        predict_RNA_count_onlyatac_all = []
        RNA_id_all = []
        RNA_count_all = []
        atac_cell_embed_all = []
        all_samples = 0
        predict_RNA_count_onlyatac = np.zeros((feature_len, vocab_size_RNA))
        RNA_count = np.zeros((feature_len, vocab_size_RNA))
        atac_cell_embed = np.zeros((feature_len, 128))
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_ATAC, source_values_ATAC,
             source_batch_ATAC, source_id_ATAC) \
                in (zip(test_db_RNA, test_db_ATAC)):

            if all_samples >= feature_len:
                break

            z2_1 = embedding_decoder_pretrain(source_values_ATAC,training=False)
            output_onlyatac = final_enc(z2_1,training=False)
            decoder_output_ATAC = tf.nn.l2_normalize(z2_1, axis=-1)

            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            predict_RNA_count_onlyatac[all_samples:all_samples + len(source_id_RNA), :] = output_onlyatac
            RNA_count[all_samples:all_samples + len(source_id_RNA), :] = source_values_RNA
            atac_cell_embed[all_samples:all_samples + len(source_id_RNA), :] = decoder_output_ATAC
            all_samples += len(source_id_RNA)
            #print('all_samples num:{}'.format(all_samples))

        predict_RNA_count_onlyatac_all.extend(predict_RNA_count_onlyatac[:all_samples])
        RNA_id_all.extend(RNA_id[:all_samples])
        RNA_count_all.extend(RNA_count[:all_samples])
        atac_cell_embed_all.extend(atac_cell_embed[:all_samples])
        predict_RNA_count_onlyatac_all = np.array(predict_RNA_count_onlyatac_all).astype('float32')
        RNA_count_all = np.array(RNA_count_all)
        atac_cell_embed_all = np.array(atac_cell_embed_all)
        print('RNA_count_all shape',RNA_count_all.shape)
        pearson_r_list, pearson_r_flatten, npos = plot_cor_pergene(RNA_count_all,
                                                                   predict_RNA_count_onlyatac_all,
                                                                   logscale=False, normlib=False)
        pearson_r_mean = np.mean(pearson_r_list)
        mse = np.sqrt(mean_squared_error(RNA_count_all, predict_RNA_count_onlyatac_all))
        print('epoch:{}'.format(epoch))
        print('test pearson_r_mean:', pearson_r_mean)
        print('test pearson_r_flatten:', pearson_r_flatten)
        print('test mse:', mse)
        f = {
            'predict_RNA_count_onlyatac': predict_RNA_count_onlyatac_all,
            'RNA_id': RNA_id_all,
            'atac_cell_embed':atac_cell_embed_all
            }
        #np.savez_compressed(saved_result_path + f'/result_ep{epoch + 1}_p{pearson_r_mean}.npz', **f)
        np.savez_compressed(saved_result_path + f'/result_ep{epoch + 1}_m{mse}.npz', **f)



    return print('regress finished')

def concerto_test_multimodal_project(model_path: str, RNA_tf_path: str, Protein_tf_path: str, super_parameters=None,saved_weight_path = None):
    if super_parameters is None:
        super_parameters = {'batch_size': 32, 'epoch': 1, 'lr': 1e-5, 'drop_rate': 0.1}

    batch_size = super_parameters['batch_size']
    f = np.load(os.path.join(RNA_tf_path, 'vocab_size.npz'))
    vocab_size_RNA = int(f['vocab size'])
    f = np.load(os.path.join(Protein_tf_path, 'vocab_size.npz'))
    vocab_size_Protein = int(f['vocab size'])
    encode_network = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA,vocab_size_Protein],
        mult_feature_names=['RNA','Protein'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)
    encode_network_RNA = multi_embedding_attention_transfer_explainability(
        multi_max_features=[vocab_size_RNA],
        mult_feature_names=['RNA'],
        embedding_dims=128,
        include_attention=True,
        drop_rate=super_parameters['drop_rate'],
        head_1=128,
        head_2=128,
        head_3=128)


    tf_list_1 = [f for f in os.listdir(os.path.join(RNA_tf_path)) if 'tfrecord' in f]
    train_source_list_RNA = []
    train_source_list_Protein = []
    for i in tf_list_1:
        train_source_list_RNA.append(os.path.join(RNA_tf_path, i))
        train_source_list_Protein.append(os.path.join(Protein_tf_path, i))

    weight_id_list = []
    weight_list = [f for f in os.listdir(model_path) if f.endswith('h5')]

    for id in weight_list:
        id_1 = re.findall('.*epoch(.*).h.*', id)  # f1
        weight_id_list.append(int(id_1[0]))
    
    if  saved_weight_path is None:
        encode_network.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)
        encode_network_RNA.load_weights(model_path + 'weight_encoder_epoch{}.h5'.format(max(weight_id_list)),
                                    by_name=True)

    else:
        encode_network.load_weights(saved_weight_path,by_name=True)
        encode_network_RNA.load_weights(saved_weight_path, by_name=True)
        

    source_data_batch = []
    source_data_feature = []
    source_data_feature_RNA = []    
    RNA_id_all = []

    for RNA_file, Protein_file in zip(train_source_list_RNA, train_source_list_Protein):
        print(RNA_file)
        print(Protein_file)
        train_size = 0
        train_db_RNA = create_classifier_dataset_multi([RNA_file],
                                                       batch_size=batch_size,
                                                       is_training=False,
                                                       data_augment=False,
                                                       shuffle_size=10000,
                                                       )
        train_db_Protein = create_classifier_dataset_multi([Protein_file],
                                                           batch_size=batch_size,
                                                           is_training=False,
                                                           data_augment=False,
                                                           shuffle_size=10000,
                                                           )
        step = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            train_size += len(source_id_RNA)
            if step == 0:
                encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                                        [source_values_RNA, source_values_protein]],
                                                                       training=False)

        dim = encode_output.shape[1]
        source_data_feature_1 = np.zeros((train_size, dim))
        source_data_feature_RNA_1 = np.zeros((train_size, dim))        
        source_data_batch_1 = np.zeros((train_size))
        RNA_id = []
        all_samples = 0
        for (source_features_RNA, source_values_RNA,
             source_batch_RNA, source_id_RNA), \
            (source_features_protein, source_values_protein,
             source_batch_Protein, source_id_Protein) \
                in (zip(train_db_RNA, train_db_Protein)):
            encode_output, attention_output = encode_network([[source_features_RNA, source_features_protein],
                                                              [source_values_RNA, source_values_protein]],
                                                             training=False)
            encode_output_RNA, attention_output_ = encode_network_RNA([[source_features_RNA],
                                                              [source_values_RNA]],
                                                             training=False)


            encode_output = tf.nn.l2_normalize(encode_output, axis=-1)
            source_data_feature_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output
            source_data_feature_RNA_1[all_samples:all_samples + len(source_id_RNA), :] = encode_output_RNA            
            source_data_batch_1[all_samples:all_samples + len(source_id_RNA)] = source_batch_RNA
            RNA_id.extend(list(source_id_RNA.numpy().astype('U')))
            all_samples += len(source_id_RNA)
            print('all_samples num:{}'.format(all_samples))

        source_data_feature.extend(source_data_feature_1)
        source_data_feature_RNA.extend(source_data_feature_RNA_1)        
        source_data_batch.extend(source_data_batch_1)
        RNA_id_all.extend(RNA_id)

    source_data_feature = np.array(source_data_feature).astype('float32')
    source_data_feature_RNA = np.array(source_data_feature_RNA).astype('float32')    
    source_data_batch = np.array(source_data_batch).astype('int32')

    return source_data_feature,source_data_feature_RNA, source_data_batch, RNA_id_all




def knn_classifier(ref_embedding, query_embedding, ref_anndata, source_data_id, column_name,k, num_chunks=100):
    '''
    return :
        target_neighbor: predicted label
        traget_prob: confidence score
    '''
    train_features = tf.transpose(ref_embedding)
    num_test_images = int(query_embedding.shape[0])
    imgs_per_chunk = num_test_images // num_chunks
    if imgs_per_chunk == 0:
        imgs_per_chunk = 10

    print(num_test_images, imgs_per_chunk)
    ref_anndata = ref_anndata[source_data_id]
    train_labels = ref_anndata.obs[column_name].tolist()
    target_pred_labels = []
    target_pred_prob = []
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the features for test images
        features = query_embedding[
                   idx: min((idx + imgs_per_chunk), num_test_images), :
                   ]
        # targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        similarity = tf.matmul(features, train_features)
        target_distances, target_indices = tf.math.top_k(similarity, k, sorted=True)

        for distances, indices in zip(target_distances, target_indices):
            selected_label = {}
            selected_count = {}
            count = 0
            for distance, index in zip(distances, indices):
                label = train_labels[index]
                weight = distance
                if label not in selected_label:
                    selected_label[label] = 0
                    selected_count[label] = 0
                selected_label[label] += weight
                selected_count[label] += 1
                count += 1

            filter_label_list = sorted(selected_label.items(), key=lambda x: x[1], reverse=True)
            target_pred_labels.append(filter_label_list[0][0])

            prob = selected_label[filter_label_list[0][0]] / selected_count[filter_label_list[0][0]]
            target_pred_prob.append(prob)

    target_neighbor = np.array(target_pred_labels)
    target_prob = np.array(target_pred_prob)

    return target_neighbor, target_prob #返回预测的label和置信度

if __name__ == '__main__':
    print('this is main function')
