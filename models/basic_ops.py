"""
模块2-基本单元：卷积、池化、ReLU

"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def conv2d(x, W, b):
    """卷积 + 偏置 + ReLU
    参数:
        x: 输入张量, shape=(N, H, W, C_in)
        W: 卷积核常量, shape=(kH, kW, C_in, C_out)
        b: 偏置常量, shape=(C_out,)
    返回:
        ReLU 后张量, shape=(N, H, W, C_out)
    """
    y = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return tf.nn.relu(y)


def conv2d_no_relu(x, W, b):
    """纯卷积（不接 ReLU）"""
    y = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    y = tf.nn.bias_add(y, b)
    return y


def max_pool(x):
    """2x2 最大池化, stride=2"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def avg_pool(x):
    """2x2 平均池化, stride=2 """
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')
