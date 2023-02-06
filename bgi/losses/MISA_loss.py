import tensorflow as tf

def scm(sx1, sx2, k):
    ss1 = tf.reduce_mean(tf.pow(sx1, k), 0)
    ss2 = tf.reduce_mean(tf.pow(sx2, k), 0)
    return tf.sqrt(tf.reduce_sum((ss1-ss2)**2))


def CMD(x1,x2,n_moments):
    mx1 = tf.reduce_mean(x1, axis=0)
    mx2 = tf.reduce_mean(x2, axis=0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = tf.sqrt(tf.reduce_sum((x1-x2)**2))
    scms = dm
    for i in range(n_moments - 1):
        scms += scm(sx1, sx2, i + 2)
    return scms


def loss_diff(input1, input2):

    batch_size = input1.shape[0]
    input1 = tf.reshape(input1,[batch_size, -1])
    input2 = tf.reshape(input2, [batch_size, -1])
    # Zero mean
    input1_mean = tf.reduce_mean(input1, axis=0)
    input2_mean = tf.reduce_mean(input2, axis=0)
    input1 = input1 - input1_mean
    input2 = input2 - input2_mean

    input1_l2_norm = tf.stop_gradient(tf.norm(input1, ord=2, axis=1, keepdims=True))
    input1_l2_norm = tf.reshape(input1_l2_norm,[batch_size, -1]) + 1e-6
    #input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
    input1_l2 = tf.divide(input1,input1_l2_norm)

    input2_l2_norm = tf.stop_gradient(tf.norm(input2, ord=2, axis=1, keepdims=True))
    input2_l2_norm = tf.reshape(input2_l2_norm, [batch_size, -1]) + 1e-6
    #input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)
    input2_l2 = tf.divide(input2, input2_l2_norm)

    #diff_loss = tf.reduce_mean((input1_l2.t().mm(input2_l2)).pow(2))
    a = tf.matmul(tf.transpose(input1_l2),input2_l2)
    diff_loss = tf.reduce_mean(tf.square(a))

    return diff_loss


def get_diff_loss(utt_shared_t,utt_shared_v,utt_private_t,utt_private_v):
    shared_t = utt_shared_t
    shared_v = utt_shared_v
    private_t = utt_private_t
    private_v = utt_private_v

    # Between private and shared
    loss = loss_diff(private_t, shared_t)
    loss += loss_diff(private_v, shared_v)
    # Across privates
    loss += loss_diff(private_t, private_v)

    return loss

