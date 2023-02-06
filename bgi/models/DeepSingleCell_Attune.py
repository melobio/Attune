import tensorflow as tf
from tensorflow.keras.models import Model  # layers, Sequential, optimizers, losses, metrics, datasets
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding, Input
from tensorflow.keras.layers import GlobalAveragePooling1D,Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K
from bgi.layers.attention import AttentionWithContext
import numpy as np
import sys
sys.path.append('.../')
from transformer_model_cls import TransformerModel_cls
from transformer_model_cls_1 import TransformerModel_cls_1
from transformer_model_infer import TransformerModel_infer



def multi_embedding_attention_transfer(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=output)

def multi_embedding_attention_pretrain_RNA(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)
        dropout = Dropout(rate=drop_rate)(features[0])
        output = Dense(head_1, name='projection-1', activation='relu')(dropout)

        # inputs = []
        # inputs.append(x_feature_inputs)
        # inputs.append(x_value_inputs)
        return tf.keras.Model(inputs=inputs, outputs=[output, weight_output])

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
        dropout = Dropout(rate=drop_rate)(features[0])
        output = Dense(head_1, name='projection-2', activation='relu')(dropout)

        # inputs = []
        # inputs.append(x_feature_inputs)
        # inputs.append(x_value_inputs)
        return tf.keras.Model(inputs=inputs, outputs=output)



def multi_embedding_attention_pretrain_ATAC(
                                    embedding_matrix,
                                    supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            if not embedding_matrix is None:
                embedding = Embedding(max_length, embedding_dims, input_length=None,
                                      embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                                      name='{}-Embedding'.format(name))(feature_input)
            else:
                embedding = Embedding(max_length, embedding_dims, input_length=None,
                                      name='{}-Embedding'.format(name))(feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            #weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)
        dropout = Dropout(rate=drop_rate)(features[0])
        output = Dense(head_1, name='projection-1', activation='relu')(dropout)

        # inputs = []
        # inputs.append(x_feature_inputs)
        # inputs.append(x_value_inputs)
        return tf.keras.Model(inputs=inputs, outputs=[output, weight_output])

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
        dropout = Dropout(rate=drop_rate)(features[0])
        output = Dense(head_1, name='projection-2', activation='relu')(dropout)

        # inputs = []
        # inputs.append(x_feature_inputs)
        # inputs.append(x_value_inputs)
        return tf.keras.Model(inputs=inputs, outputs=output)


def multi_embedding_attention_transfer_1(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=[output,weight_output_all])

def multi_embedding_attention_transfer_explainability(supvised_train: bool = False,
                                    scan_train: bool = False,
                                    multi_max_features: list = [40000],
                                    mult_feature_names: list = ['Gene'],
                                    embedding_dims=128,
                                    num_classes=5,
                                    activation='softmax',
                                    head_1=128,
                                    head_2=128,
                                    head_3=128,
                                    drop_rate=0.05,
                                    include_attention: bool = False,
                                    use_bias=True,
                                    ):
    assert len(multi_max_features) == len(mult_feature_names)

    # 特征索引
    x_feature_inputs = []
    # 特征值
    x_value_inputs = []
    # 特征向量
    embeddings = []
    features = []
    weight_output_all = []
    if include_attention == True:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入
            feature_input = Input(shape=(None,), name='Input-{}-Feature'.format(name))
            value_input = Input(shape=(None,), name='Input-{}-Value'.format(name), dtype='float')
            x_feature_inputs.append(feature_input)
            x_value_inputs.append(value_input)

            # 向量
            embedding = Embedding(max_length, embedding_dims, input_length=None, name='{}-Embedding'.format(name))(
                feature_input)

            # 向量 * 特征值
            sparse_value = tf.expand_dims(value_input, 2, name='{}-Expend-Dims'.format(name))
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(sparse_value)
            x = tf.multiply(embedding, sparse_value, name='{}-Multiply'.format(name))
            # x = BatchNormalization(name='{}-BN-2'.format(name))(x)

            # # Attention
            weight_output,a = AttentionWithContext()(x)
            x = K.tanh(K.sum(weight_output, axis=1))

            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
            weight_output_all.append(a)
        inputs = []
        inputs.append(x_feature_inputs)
        inputs.append(x_value_inputs)

    else:
        for max_length, name in zip(multi_max_features, mult_feature_names):
            # 输入

            value_input = Input(shape=(max_length,), name='Input-{}-Value'.format(name), dtype='float')

            x_value_inputs.append(value_input)

            # 向量 * 特征值
            sparse_value = BatchNormalization(name='{}-BN-1'.format(name))(value_input)

            x = Dense(head_1, name='{}-projection-0'.format(name), activation='relu')(sparse_value)


            x = BatchNormalization(name='{}-BN-3'.format(name))(x)

            features.append(x)
        inputs = []
        inputs.append(x_value_inputs)
    # Concatenate
    if len(features) > 1:
    #feature = concatenate(features)
        feature = Add()([features[0],features[1]])
    else:
        feature = features[0]
    dropout = Dropout(rate=drop_rate)(feature)
    output = Dense(head_1, name='projection-1', activation='relu')(dropout)

    # inputs = []
    # inputs.append(x_feature_inputs)
    # inputs.append(x_value_inputs)
    return tf.keras.Model(inputs=inputs, outputs=[output,weight_output_all])



class EncoderHead(tf.keras.Model):

    def __init__(self, hidden_size=128, dropout=0.05):
        super(EncoderHead, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_bn1(feature_output)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)

        return feature_output

class EncoderHead_residual(tf.keras.Model):

    def __init__(self, hidden_size1=1000,hidden_size2=2000, dropout=0.05):
        super(EncoderHead_residual, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size1, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size1, activation='relu')
        self.feature_fc3 = tf.keras.layers.Dense(units=hidden_size2, activation=None)
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self,input):
        feature_output = self.feature_fc1(input)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)
        feature_output += input
        feature_output = self.feature_fc3(feature_output)

        return feature_output


class EncoderHead_2(tf.keras.Model):

    def __init__(self, hidden_size1=500,hidden_size2=1000,hidden_size3 = 2000 ,dropout=0.05):
        super(EncoderHead_2, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size1, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size2, activation='relu')
        self.feature_fc3 = tf.keras.layers.Dense(units=hidden_size3, activation=None)

        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc2(feature_output)
        feature_output = self.feature_dropout1(feature_output)
        feature_output = self.feature_fc3(feature_output)
        # feature_output = self.feature_dropout1(feature_output)
        # feature_output = self.feature_fc4(feature_output)
        # feature_output = self.feature_dropout1(feature_output)
        # feature_output = self.feature_fc5(feature_output)



        return feature_output


class EncoderHead_ITM(tf.keras.Model):

    def __init__(self, hidden_size=1000, dropout=0.05):
        super(EncoderHead_ITM, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='sigmoid')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation=None)
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        # feature_output = self.feature_fc2(feature_output)
        # feature_output = self.feature_dropout1(feature_output)
        # feature_output = self.feature_fc2(feature_output)

        return feature_output

class EncoderHead_add(tf.keras.Model):

    def __init__(self, hidden_size=128, dropout=0.05):
        super(EncoderHead_add, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x1 = input[0]
        x2 = input[1]
        feature = Add()([x1, x2])
        feature_output = self.feature_dropout1(feature)
        feature_output = self.feature_fc2(feature_output)

        return feature_output

class EncoderHead_MISA(tf.keras.Model):

    def __init__(self, hidden_size=1000, dropout=0.05):
        super(EncoderHead_MISA, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size, activation=None)
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        return feature_output

class EncoderHead_1(tf.keras.Model):

    def __init__(self, hidden_size1=1000,hidden_size2=2000, dropout=0.05):
        super(EncoderHead_1, self).__init__()
        # self.num_classes = num_classes
        self.feature_fc1 = tf.keras.layers.Dense(units=hidden_size1, activation='relu')
        self.feature_fc2 = tf.keras.layers.Dense(units=hidden_size2, activation=None)
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)

    def call(self, input):
        x = input
        feature_output = self.feature_fc1(x)
        feature_output = self.feature_fc2(feature_output)
        # feature_output = self.feature_dropout1(feature_output)
        # feature_output = self.feature_fc2(feature_output)

        return feature_output

class Transformer_model_cls(tf.keras.Model):

    def __init__(self, vocab_size=128, stack_size=1, num_heads = 1,hidden_size=128,attention_mask_path=''):
        super(Transformer_model_cls, self).__init__()
        # self.num_classes = num_classes
        self.TransformerModel_CLS = TransformerModel_cls(vocab_size=vocab_size,  # vocab_size_ATAC
                                 encoder_stack_size=stack_size,
                                 decoder_stack_size=stack_size,
                                 hidden_size=hidden_size,  # 512
                                 num_heads=num_heads,
                                 filter_size=512,  # 2048
                                 dropout_rate=0.1,
                                 attention_mask_path = attention_mask_path)


    def call(self, input):
        x1 = input[0]
        x2 = input[1]
        encoder_output, decoder_output, logits = self.TransformerModel_CLS(x1,x2)
        return encoder_output, decoder_output, logits

class Transformer_model_cls_1(tf.keras.Model):

    def __init__(self, vocab_size=128, stack_size=1, num_heads = 1,hidden_size=128,attention_mask_path=''):
        super(Transformer_model_cls_1, self).__init__()
        # self.num_classes = num_classes
        self.TransformerModel_CLS = TransformerModel_cls_1(vocab_size=vocab_size,  # vocab_size_ATAC
                                 encoder_stack_size=stack_size,
                                 decoder_stack_size=stack_size,
                                 hidden_size=hidden_size,  # 512
                                 num_heads=num_heads,
                                 filter_size=512,  # 2048
                                 dropout_rate=0,
                                 attention_mask_path = attention_mask_path)


    def call(self, input):
        x1 = input[0]
        x2 = input[1]
        encoder_output, decoder_output, logits = self.TransformerModel_CLS(x1,x2)
        return encoder_output, decoder_output, logits

class Transformer_model_extract(tf.keras.Model):

    def __init__(self, vocab_size=128, stack_size=1, num_heads = 1,hidden_size=128,attention_mask_path=''):
        super(Transformer_model_extract, self).__init__()
        self.TransformerModel = TransformerModel_infer(vocab_size=vocab_size,  # vocab_size_ATAC
                                 encoder_stack_size=stack_size,
                                 decoder_stack_size=stack_size,
                                 hidden_size=hidden_size,  # 512
                                 num_heads=num_heads,
                                 filter_size=512,  # 2048
                                 dropout_rate=0,
                                 attention_mask_path=attention_mask_path)


    def call(self, input):
        x1 = input[0]
        x2 = input[1]
        feature_output,crossattention = self.TransformerModel(x1,x2)

        return feature_output,crossattention



class GELU(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, x):
        # return tf.keras.activations.sigmoid(1.702 * x) * x
        #return tf.keras.activations.sigmoid(tf.constant(1.702) * x) * x
        return tf.keras.activations.relu(x)

class MISA(tf.keras.Model):

    def __init__(self, hidden_size=128, dropout=0.05):
        super(MISA, self).__init__()
        # self.num_classes = num_classes
        self.project_t = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.project_v = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.feature_bn1 = tf.keras.layers.BatchNormalization()
        self.feature_bn2 = tf.keras.layers.BatchNormalization()
        self.private_t = tf.keras.layers.Dense(units=hidden_size, activation='sigmoid')
        self.private_v = tf.keras.layers.Dense(units=hidden_size, activation='sigmoid')
        self.shared = tf.keras.layers.Dense(units=hidden_size, activation='sigmoid')
        self.feature_dropout1 = tf.keras.layers.Dropout(rate=dropout)
        self.feature_dropout2 = tf.keras.layers.Dropout(rate=dropout)
        self.sp_discriminator = tf.keras.layers.Dense(units=3)
        self.recon_t = tf.keras.layers.Dense(units=hidden_size)
        self.recon_v = tf.keras.layers.Dense(units=hidden_size)

    def call(self, x1,x2):
        # Projecting to same sized space
        utterance_t = self.project_t(x1)
        utterance_t = self.feature_bn1(utterance_t)
        utterance_v = self.project_v(x2)
        utterance_v = self.feature_bn2(utterance_v)
        # Private-shared components
        utt_private_t = self.private_t(utterance_t)
        utt_private_v = self.private_v(utterance_v)
        utt_shared_t = self.shared(utterance_t)
        utt_shared_v = self.shared(utterance_v)
        # discriminator
        # shared_or_private_p_t = self.sp_discriminator(utt_private_t)
        # shared_or_private_p_v = self.sp_discriminator(utt_private_v)
        # shared_or_private_s = self.sp_discriminator((utt_shared_t + utt_shared_v)/2)
        # For reconstruction
        utt_t = (utt_private_t + utt_shared_t)
        utt_v = (utt_private_v + utt_shared_v)
        utt_t_recon = self.recon_t(utt_t)
        utt_v_recon = self.recon_v(utt_v)

        return utterance_t,utt_private_t,utt_shared_t,utt_t,utt_t_recon, \
               utterance_v,utt_private_v,utt_shared_v, utt_v, utt_v_recon






def dense_block(
    inputs,
    units=None,
    activation="gelu",
    activation_end=None,
    flatten=False,
    globalpool = False,
    dropout=0,
    l2_scale=0,
    l1_scale=0,
    residual=False,
    batch_norm=True,
    bn_momentum=0.90,
    bn_gamma=None,
    bn_type="standard",
    kernel_initializer="he_normal",
):
    """Construct a single convolution block.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Conv1D filters
        activation:     relu/gelu/etc
        activation_end: Compute activation after the other operations
        flatten:        Flatten across positional axis
        dropout:        Dropout rate probability
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
        residual:       Residual connection boolean
        batch_norm:     Apply batch normalization
        bn_momentum:    BatchNorm momentum
        bn_gamma:       BatchNorm gamma (defaults according to residual)
    Returns:
        [batch_size, seq_length(?), features] output sequence
    """
    current = inputs

    if units is None:
        units = inputs.shape[-1]

    # activation
    current = GELU()(current)

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    if globalpool:
        current = tf.keras.layers.GlobalAveragePooling1D()(current)
        _, seq_len = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len,
            )
        )(current)


    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=(not batch_norm),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)

    # batch norm
    if batch_norm:
        if bn_gamma is None:
            bn_gamma = "zeros" if residual else "ones"
        if bn_type == "sync":
            bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            bn_layer = tf.keras.layers.BatchNormalization
        current = bn_layer(momentum=bn_momentum, gamma_initializer=bn_gamma)(current)

    # dropout
    if dropout > 0:
        current = tf.keras.layers.Dropout(rate=dropout)(current)

    # residual add
    if residual:
        current = tf.keras.layers.Add()([inputs, current])

    return current

def final(
    inputs,
    units,
    activation="linear",
    flatten=False,
    kernel_initializer="he_normal",
    l2_scale=0,
    l1_scale=0,
):
    """Final simple transformation before comparison to targets.
    Args:
        inputs:         [batch_size, seq_length, features] input sequence
        units:          Dense units
        activation:     relu/gelu/etc
        flatten:        Flatten positional axis.
        l2_scale:       L2 regularization weight.
        l1_scale:       L1 regularization weight.
    Returns:
        [batch_size, seq_length(?), units] output sequence
    """
    current = inputs

    # flatten
    if flatten:
        _, seq_len, seq_depth = current.shape
        current = tf.keras.layers.Reshape(
            (
                1,
                seq_len * seq_depth,
            )
        )(current)

    # dense
    current = tf.keras.layers.Dense(
        units=units,
        use_bias=True,
        activation=activation,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),
    )(current)
    return current


class scbasset_model(tf.keras.Model):

    def __init__(self, units=128,units_gene = 2000, batch_norm=True,bn_gamma = None, flatten = True, globalpool = True,residual = False, kernel_initializer = "he_normal",bn_type="standard",l1_scale=0,l2_scale=0,dropout = 0.2):
        super(scbasset_model, self).__init__()
        # self.num_classes = num_classes
        self.GELU = GELU()
        self.flatten = flatten
        self.globalpool = globalpool
        self.batch_norm = batch_norm
        self.bn_gamma = bn_gamma
        self.bn_type = bn_type
        self.residual = residual
        self.Dense = tf.keras.layers.Dense(
        units=units,
        use_bias=(not batch_norm),
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale),)
        self.dropout = tf.keras.layers.Dropout(rate=dropout)
        self.Dense_final = tf.keras.layers.Dense(
        units=units_gene,
        use_bias=True,
        activation='linear',
        kernel_initializer=kernel_initializer,
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1_scale, l2_scale))


    def call(self, inputs):
        current = inputs
        current = self.GELU(current)
        # flatten
        if self.flatten:
            _, seq_len, seq_depth = current.shape
            current = tf.keras.layers.Reshape(
                (
                    1,
                    seq_len * seq_depth,
                )
            )(current)
        if self.globalpool:
            current = tf.keras.layers.GlobalAveragePooling1D()(current)
            _, seq_len = current.shape
            current = tf.keras.layers.Reshape(
                (
                    1,
                    seq_len,
                )
            )(current)
        # Dense
        current = self.Dense(current)
        # batch norm
        if self.batch_norm:
            if self.bn_gamma is None:
                bn_gamma = "zeros" if self.residual else "ones"
            if self.bn_type == "sync":
                bn_layer = tf.keras.layers.experimental.SyncBatchNormalization
            else:
                bn_layer = tf.keras.layers.BatchNormalization
            current = bn_layer(momentum=0.90, gamma_initializer=bn_gamma)(current)

        # dropout
        current = self.dropout(current)
        # residual add
        if self.residual:
            current = tf.keras.layers.Add()([inputs, current])

        current = self.GELU(current)
        current = self.Dense_final(current)
        output = tf.keras.layers.Flatten()(current)

        return output


class MoCo(tf.keras.Model):
    def __init__(self, vocab_size_RNA,vocab_size_ATAC,temperature,batch_size,optimizer, **kwargs):
        super(MoCo, self).__init__(**kwargs)
        self.temperature = temperature
        self._loss = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.encoder_RNA = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
        self.encoder_ATAC = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=True,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)
        self.decoder_RNA = multi_embedding_attention_pretrain_RNA(multi_max_features=[vocab_size_RNA],
                                                        mult_feature_names=['RNA'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

        self.decoder_ATAC = multi_embedding_attention_pretrain_ATAC(
                                                        embedding_matrix = None,
                                                        multi_max_features=[vocab_size_ATAC],
                                                        mult_feature_names=['ATAC'],
                                                        embedding_dims=128,
                                                        include_attention=False,
                                                        drop_rate=0.1,
                                                        head_1=128,
                                                        head_2=128,
                                                        head_3=128)

        _queue = np.random.normal(size=(128, 65536))
        _queue /= np.linalg.norm(_queue, axis=0)
        self.queue = self.add_weight(
            name='queue',
            shape=(128, 65536),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False) # deafult: False

        self.encoder_RNA.get_layer(index=-1).set_weights(
            self.encoder_ATAC.get_layer(index=-1).get_weights())

        # self.decoder_RNA.get_layer(index=-1).set_weights(
        #     self.decoder_ATAC.get_layer(index=-1).get_weights())
        #
        # self.decoder_ATAC.get_layer(index=-1).set_weights(
        #     self.encoder_ATAC.get_layer(index=-1).get_weights())


    def call(self, input):
        feature_RNA = input[0]
        value_RNA = input[1]
        feature_ATAC = input[2]
        value_ATAC = input[3]
        #labels = tf.zeros(shape=(128,65537))
        labels = K.zeros(shape=(self.batch_size,65537))
        ones = K.ones(shape=(self.batch_size,))
        labels = tf.Variable(labels, name='labels')
        labels[:,0].assign(ones)

        RNA_t, cell_gene_embed = self.encoder_RNA([feature_RNA,value_RNA], training=False)
        RNA_t = tf.math.l2_normalize(RNA_t, axis=1)

        # RNA_s = self.decoder_RNA(value_RNA, training=False)
        # RNA_s = tf.math.l2_normalize(RNA_s, axis=1)

        # ATAC_s = self.decoder_ATAC(value_ATAC, training=False)
        # ATAC_s = tf.math.l2_normalize(ATAC_s, axis=1)


        with tf.GradientTape() as tape:
            ATAC_t, cell_peak_embed = self.encoder_ATAC([feature_ATAC,value_ATAC], training=True)
            ATAC_t = tf.math.l2_normalize(ATAC_t, axis=1)

            # ATAC_s = self.decoder_ATAC(value_ATAC, training=True)
            # ATAC_s = tf.math.l2_normalize(ATAC_s, axis=1)

            l_pos = tf.einsum('nc,nc->n', ATAC_t, tf.stop_gradient(RNA_t))[:, None]
            l_neg = tf.einsum('nc,ck->nk', ATAC_t, self.queue)
            #####################################################
            #l_pos1 = tf.einsum('nc,nc->n', ATAC_t, ATAC_s)[:, None]
            # l_pos1 = tf.einsum('nc,nc->n', ATAC_t, tf.stop_gradient(ATAC_s))[:, None]
            # logits1 = tf.concat((l_pos1, l_neg), axis=1)
            # logits1 /= self.temperature
            # logits1 = tf.nn.softmax(logits1,axis=-1)
            # loss_moco3 = self._loss(labels, logits1)
            # loss_moco3 = tf.reduce_mean(loss_moco3)

            logits = tf.concat((l_pos, l_neg), axis=1)
            logits /= self.temperature
            logits = tf.nn.softmax(logits,axis=-1)
            loss_moco1 = self._loss(labels, logits)
            loss_moco1 = tf.reduce_mean(loss_moco1)
            ####################################################
            # l_pos = tf.einsum('nc,nc->n', ATAC_s, tf.stop_gradient(RNA_s))[:, None]
            # l_neg = tf.einsum('nc,ck->nk', ATAC_s, self.queue)
            #
            # logits = tf.concat((l_pos, l_neg), axis=1)
            # logits /= self.temperature
            # logits = tf.nn.softmax(logits,axis=-1)
            # loss_moco2 = self._loss(labels, logits)
            # loss_moco2 = tf.reduce_mean(loss_moco2)

            loss = loss_moco1
            total_loss = loss

        # trainable_vars = self.encoder_ATAC.trainable_variables
        # trainable_vars = self.encoder_RNA.trainable_variables
        trainable_vars = [self.encoder_ATAC.trainable_variables]
        grads = tape.gradient(total_loss, trainable_vars)
        for grad, var in zip(grads, trainable_vars):
            self.optimizer.apply_gradients(zip(grad, var))


        return total_loss,self.encoder_RNA,self.encoder_ATAC,self.decoder_RNA,self.decoder_ATAC


class CenterLossLayer(tf.keras.layers.Layer):

    def __init__(self, alpha=0.5, num_classes=10, num_dim=2, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.num_classes = num_classes
        self.num_dim = num_dim

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_classes, self.num_dim),
                                       initializer='uniform',
                                       trainable=False)
        # self.counter = self.add_weight(name='counter',
        #                                shape=(1,),
        #                                initializer='zeros',
        #                                trainable=False)  # just for debugging
        super().build(input_shape)

    def call(self, x, mask=None):

        # x[0] is Nx2, x[1] is Nx10 onehot, self.centers is 10x2
        delta_centers = K.dot(K.transpose(x[1]), (K.dot(x[1], self.centers) - x[0]))  # 10x2
        center_counts = K.sum(K.transpose(x[1]), axis=1, keepdims=True) + 1  # 10x1
        delta_centers /= center_counts
        new_centers = self.centers - self.alpha * delta_centers
        self.add_update((self.centers, new_centers), x)

        # self.add_update((self.counter, self.counter + 1), x)

        self.result = x[0] - K.dot(x[1], self.centers)
        self.result = K.sum(self.result ** 2, axis=1, keepdims=True) #/ K.dot(x[1], center_counts)
        return self.result # Nx1

    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'alpha': self.alpha,
            'num_classes': self.num_classes,
            'num_dim': self.num_dim,
        })
        return config




