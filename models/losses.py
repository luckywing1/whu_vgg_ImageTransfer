"""
模块2-损失：内容损失、风格损失、Adam 优化器

"""
import numpy as np



# 内容损失
def content_loss_forward(F_x, F_c):
    """前向：返回标量损失值（python float）

    参数:
        F_x: (1, H, W, C) numpy array, 当前生成图在内容层的特征
        F_c: (1, H, W, C) numpy array, 内容图在内容层的特征
    """
    diff = F_x - F_c
    return 0.5 * float(np.sum(diff * diff))


def content_loss_backward(F_x, F_c):
    """反向：返回 dL/dF_x，与 F_x 同 shape"""
    return (F_x - F_c).astype(np.float32)



# Gram 矩阵
def gram_matrix(F):
    """计算 Gram 矩阵
    参数:
        F: (1, H, W, C) numpy array
    返回:
        G: (C, C) numpy array, G = F_flat^T @ F_flat
        F_flat: (M, C) reshape 后的特征，反向用
    """
    _, H, W, C = F.shape
    F_flat = F.reshape(H * W, C)
    G = F_flat.T @ F_flat
    return G, F_flat



# 风格损失（单层）
def style_loss_forward(F_x, G_s):
    """前向：返回 (loss_value, F_flat, G_x) 用于反向复用计算

    参数:
        F_x: (1, H, W, C) 当前生成图特征
        G_s: (C, C) 风格图 Gram 矩阵
    """
    _, H, W, C = F_x.shape
    N = C
    M = H * W
    G_x, F_flat = gram_matrix(F_x)
    diff = G_x - G_s
    loss = float(np.sum(diff * diff)) / (4.0 * (N ** 2) * (M ** 2))
    return loss, F_flat, G_x


def style_loss_backward(F_x_shape, F_flat, G_x, G_s):
    """反向：返回 dL/dF_x, shape 与 F_x 一致
    """
    _, H, W, C = F_x_shape
    N = C
    M = H * W
    diff = G_x - G_s
    dL_dF_flat = F_flat @ diff / ((N ** 2) * (M ** 2))   # (M, N)
    dL_dF = dL_dF_flat.reshape(1, H, W, C).astype(np.float32)
    return dL_dF



# 总损失
def compute_losses_and_grads(content_feat, style_grams,
                             x_content_feat, x_style_feats,
                             style_layer_weights, alpha, beta):
    """一次性计算总损失与各层 dL/dF_x

    参数:
        content_feat:   numpy (1,H,W,C)            内容图在内容层的特征（常量）
        style_grams:    list of (C_l, C_l) numpy   各风格层的 Gram（常量）
        x_content_feat: numpy (1,H,W,C)            生成图在内容层的特征
        x_style_feats:  list of numpy              生成图在各风格层的特征
        style_layer_weights: list of float         各风格层权重 w_l
        alpha, beta:    float                       权重

    返回:
        L_total: float
        L_content: float
        L_style:   float
        grad_content_feat: numpy 与 x_content_feat 同 shape, ∂L_total/∂F_content
        grad_style_feats:  list of numpy, 每层 ∂L_total/∂F_style^l
    """
    # ---- 内容损失 ----
    L_content = content_loss_forward(x_content_feat, content_feat)
    grad_content = content_loss_backward(x_content_feat, content_feat)
    grad_content_feat = (alpha * grad_content).astype(np.float32)

    # ---- 风格损失 ----
    L_style = 0.0
    grad_style_feats = []
    for F_x, G_s, w in zip(x_style_feats, style_grams, style_layer_weights):
        l_l, F_flat, G_x = style_loss_forward(F_x, G_s)
        L_style += w * l_l
        dF = style_loss_backward(F_x.shape, F_flat, G_x, G_s)   # dL_l/dF
        grad_style_feats.append((beta * w * dF).astype(np.float32))

    L_total = alpha * L_content + beta * L_style
    return L_total, alpha * L_content, beta * L_style, grad_content_feat, grad_style_feats



# Adam 优化器
class AdamOptimizer:
    """numpy 实现的 Adam:
        m_t = β1 m_{t-1} + (1-β1) g
        v_t = β2 v_{t-1} + (1-β2) g²
        m_hat = m_t / (1-β1^t)
        v_hat = v_t / (1-β2^t)
        x   = x - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, shape, lr=2.0, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)

    def step(self, x, grad):
        """对 x 做一次 in-place 更新；返回更新后的 x"""
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        x -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return x
