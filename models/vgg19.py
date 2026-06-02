"""
模块3：VGG19 网络结构 + 预训练权重加载
"""
import numpy as np
import scipy.io
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .basic_ops import conv2d, max_pool, avg_pool


# VGG19 卷积部分按顺序排列（共 16 个 conv，5 个 pool）
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2',
    'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2',
    'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2',
    'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5',
)

# 风格迁移使用的层
CONTENT_LAYER = 'conv4_2'
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']


def load_weights(mat_path):
    """从 imagenet-vgg-verydeep-19.mat 加载权重
    返回:
        weights: dict[layer_name -> (W, b)]
            W: ndarray, shape=(kH, kW, in_C, out_C)
            b: ndarray, shape=(out_C,)
        mean_pixel: shape=(3,) RGB 均值
    """
    data = scipy.io.loadmat(mat_path)

    # ---- 提取均值 ----
    mean_pixel = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    try:
        if 'normalization' in data:
            mean = data['normalization'][0][0][0]
            if hasattr(mean, 'dtype') and mean.dtype.kind in ('f', 'u', 'i'):
                mean_pixel = np.mean(mean.astype(np.float32), axis=(0, 1))
        elif 'meta' in data:
            norm_field = data['meta'][0][0]['normalization']
            if hasattr(norm_field, 'dtype') and norm_field.dtype.names:
                mean = norm_field[0][0]['averageImage'][0][0]
            else:
                mean = norm_field[0][0][0]
            if hasattr(mean, 'dtype') and mean.dtype.kind in ('f', 'u', 'i'):
                mean_pixel = np.mean(mean.astype(np.float32), axis=(0, 1))
    except Exception:
        pass 
    print('   ImageNet mean_pixel =', mean_pixel)

    # ---- 提取层数据 ----
    if 'layers' in data:
        layers_data = data['layers'][0]
    else:
        layers_data = data['net'][0][0][0][0]  

    weights = {}
    for i, name in enumerate(VGG19_LAYERS):
        if name.startswith('conv'):
            # 尝试多种常见索引路径
            layer = layers_data[i]
            try:
                kernels = layer[0][0][2][0][0]   # 标准 MatConvNet 格式
                bias = layer[0][0][2][0][1]
            except (IndexError, KeyError):
                kernels = layer[0][0]['weights'][0][0] 
                bias = layer[0][0]['weights'][0][1]
            kernels = np.array(kernels, dtype=np.float32)
            bias = np.array(bias, dtype=np.float32).reshape(-1)
            weights[name] = (kernels, bias)
    return weights, mean_pixel


def build_vgg19(input_tensor, weights, pool_type='max'):
    """构建 VGG19 前向网络（卷积部分）
    参数:
        input_tensor: shape=(N, H, W, 3) 的图像 tensor（已减均值）
        weights: load_weights 返回的 dict
        pool_type: 'max' 或 'avg'
    返回:
        net: dict[layer_name -> tensor]
    """
    pool_fn = max_pool if pool_type == 'max' else avg_pool

    net = {}
    x = input_tensor
    for name in VGG19_LAYERS:
        kind = name[:4]
        if kind == 'conv':
            W, b = weights[name]
            # 转为 TF 常量
            W_const = tf.constant(W, dtype=tf.float32, name=name + '_W')
            b_const = tf.constant(b, dtype=tf.float32, name=name + '_b')
            x = tf.nn.conv2d(x, W_const, strides=[1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(x, b_const)
            net[name] = x
        elif kind == 'relu':
            x = tf.nn.relu(x)
            net[name] = x
        elif kind == 'pool':
            x = pool_fn(x)
            net[name] = x
        else:
            raise ValueError('Unknown layer kind: ' + name)
    return net
