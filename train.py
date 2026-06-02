"""
模块4：训练主流程（非实时风格迁移迭代优化）

训练循环每一步:
  1. sess.run 取出生成图 X 在内容层 + 各风格层的特征 (numpy)
  2. numpy 计算总损失 L、各特征图的梯度 dL/dF
  3. 用 sess.run(grad_X_op, feed_dict={grad_ys 占位符: numpy 梯度})
     一次反向 tf VGG，得到 dL/dX  (numpy)
  4. numpy Adam 更新 X
  5. sess.run(X.assign(X_np)) 写回 tf.Variable
"""
import os
import time
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from utils.data_loader import load_image, preprocess, deprocess
from utils.image_utils import save_image
from models.vgg19 import (
    load_weights, build_vgg19,
    CONTENT_LAYER, STYLE_LAYERS,
)
from models.losses import (
    gram_matrix,
    compute_losses_and_grads,
    AdamOptimizer,
)


# ============== 默认超参数 ==============
DEFAULT_HPARAMS = dict(
    image_size=512,
    alpha=1.0,
    beta=1e3,
    learning_rate=2.0,
    num_iters=1000,
    save_every=100,
    print_every=20,
    init_from='content',
    pool_type='max',
)



def run_vgg_once(weights, img_np, layers_wanted, pool_type='max'):
    """构造一个临时图：输入是常量，跑一次 VGG 取出指定层特征"""
    g = tf.Graph()
    with g.as_default():
        x = tf.constant(img_np, dtype=tf.float32)
        net = build_vgg19(x, weights, pool_type=pool_type)
        ops = {l: net[l] for l in layers_wanted}
        with tf.Session(graph=g) as ss:
            return ss.run(ops)


def stylize(content_path, style_path, weights_path, output_dir, hparams):
    os.makedirs(output_dir, exist_ok=True)

    # ---------- 1. 加载图像 ----------
    print('[1] 加载图像 ...')
    content_raw = load_image(content_path, max_size=hparams['image_size'])
    H, W = content_raw.shape[1], content_raw.shape[2]
    style_raw = load_image(style_path, shape=(H, W))
    print('   content shape:', content_raw.shape,
          ' style shape:', style_raw.shape)
    content_img = preprocess(content_raw)
    style_img = preprocess(style_raw)

    # ---------- 2. 加载权重 ----------
    print('[2] 加载 VGG19 权重 ...')
    weights, mean_pixel = load_weights(weights_path)
    print('   ImageNet mean =', mean_pixel)

    # ---------- 3. 预先计算 style 的各层 Gram ----------
    print('[3] 计算 style Gram 矩阵 (numpy) ...')
    style_feats = run_vgg_once(weights, style_img, STYLE_LAYERS,
                               pool_type=hparams['pool_type'])
    style_grams = []
    for l in STYLE_LAYERS:
        G_s, _ = gram_matrix(style_feats[l])
        style_grams.append(G_s.astype(np.float32))
        print('   {} : Gram {}'.format(l, G_s.shape))

    # ---------- 4. 预先计算 content 在 conv4_2 的特征 ----------
    print('[4] 计算 content 特征 (numpy) ...')
    content_feats = run_vgg_once(weights, content_img, [CONTENT_LAYER],
                                 pool_type=hparams['pool_type'])
    content_feat = content_feats[CONTENT_LAYER].astype(np.float32)
    print('   {} feat shape: {}'.format(CONTENT_LAYER, content_feat.shape))

    # ---------- 5. 训练图：X 为 tf.Variable，构造梯度注入入口 ----------
    print('[5] 构建训练图 ...')
    g_train = tf.Graph()
    with g_train.as_default():
        if hparams['init_from'] == 'noise':
            init_value = np.random.uniform(-20, 20,
                                           content_img.shape).astype(np.float32)
        else:
            init_value = content_img.astype(np.float32)
        X = tf.Variable(init_value, dtype=tf.float32, name='generated_image')

        net = build_vgg19(X, weights, pool_type=hparams['pool_type'])

        # 取出需要的特征 tensor
        feat_content_op = net[CONTENT_LAYER]
        feat_style_ops = [net[l] for l in STYLE_LAYERS]

        # 占位符：接收 numpy 算出来的 dL/dF
        grad_content_ph = tf.placeholder(
            tf.float32, shape=feat_content_op.shape,
            name='grad_content_feat')
        grad_style_phs = [
            tf.placeholder(tf.float32, shape=op.shape,
                           name='grad_style_feat_' + l)
            for op, l in zip(feat_style_ops, STYLE_LAYERS)
        ]

        # tf.gradients 把外部梯度注入 VGG 的反向链 -> 求 dL/dX
        ys = [feat_content_op] + feat_style_ops
        grad_ys = [grad_content_ph] + grad_style_phs
        grad_X_op = tf.gradients(ys, X, grad_ys=grad_ys)[0]

        # 写回 X 的 op
        X_assign_ph = tf.placeholder(tf.float32, shape=X.shape,
                                     name='X_assign_value')
        assign_op = X.assign(X_assign_ph)

        init_op = tf.global_variables_initializer()

        # ---------- 6. 训练 ----------
        print('[6] 开始迭代优化 ...')
        # numpy 端 Adam
        adam = AdamOptimizer(shape=content_img.shape,
                             lr=hparams['learning_rate'])
        # numpy 端 X
        X_np = init_value.copy()
        # 风格层等权
        style_w = [1.0 / len(STYLE_LAYERS)] * len(STYLE_LAYERS)

        with tf.Session(graph=g_train) as sess:
            sess.run(init_op)
            t0 = time.time()

            for step in range(hparams['num_iters'] + 1):
                # 6.1 前向：取生成图各层特征
                fc_np, *fs_np = sess.run([feat_content_op] + feat_style_ops)

                # 6.2 numpy 算损失 + 各特征图的梯度
                L_total, L_c, L_s, dC, dS_list = compute_losses_and_grads(
                    content_feat=content_feat,
                    style_grams=style_grams,
                    x_content_feat=fc_np,
                    x_style_feats=fs_np,
                    style_layer_weights=style_w,
                    alpha=hparams['alpha'],
                    beta=hparams['beta'],
                )

                if step % hparams['print_every'] == 0:
                    print('  step {:5d} | L_total={:.3e}  L_c={:.3e}  L_s={:.3e}  | {:.1f}s'.format(
                        step, L_total, L_c, L_s, time.time() - t0))

                if step % hparams['save_every'] == 0:
                    save_image(deprocess(X_np),
                               os.path.join(output_dir,
                                            'iter_{:05d}.jpg'.format(step)))

                if step >= hparams['num_iters']:
                    break

                # 6.3 把 numpy 梯度注入 tf 反向链，求 dL/dX
                feed = {grad_content_ph: dC}
                for ph, dF in zip(grad_style_phs, dS_list):
                    feed[ph] = dF
                dX_np = sess.run(grad_X_op, feed_dict=feed)

                # 6.4 numpy Adam 更新 X
                X_np = adam.step(X_np, dX_np)

                # 6.5 写回 tf.Variable，供下一轮前向
                sess.run(assign_op, feed_dict={X_assign_ph: X_np})

            # 最终结果
            final_path = os.path.join(output_dir, 'final.jpg')
            save_image(deprocess(X_np), final_path)
            print('done. final image saved to', final_path)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--content', default='data/content.jpg')
    p.add_argument('--style', default='data/style.jpg')
    p.add_argument('--weights', default='data/imagenet-vgg-verydeep-19.mat')
    p.add_argument('--output', default='output')
    p.add_argument('--image_size', type=int, default=DEFAULT_HPARAMS['image_size'])
    p.add_argument('--alpha', type=float, default=DEFAULT_HPARAMS['alpha'])
    p.add_argument('--beta', type=float, default=DEFAULT_HPARAMS['beta'])
    p.add_argument('--lr', type=float, default=DEFAULT_HPARAMS['learning_rate'])
    p.add_argument('--num_iters', type=int, default=DEFAULT_HPARAMS['num_iters'])
    p.add_argument('--save_every', type=int, default=DEFAULT_HPARAMS['save_every'])
    p.add_argument('--print_every', type=int, default=DEFAULT_HPARAMS['print_every'])
    p.add_argument('--init', choices=['content', 'noise'],
                   default=DEFAULT_HPARAMS['init_from'])
    p.add_argument('--pool', choices=['max', 'avg'],
                   default=DEFAULT_HPARAMS['pool_type'])
    return p.parse_args()


def main():
    args = parse_args()
    hparams = dict(DEFAULT_HPARAMS)
    hparams.update(dict(
        image_size=args.image_size,
        alpha=args.alpha,
        beta=args.beta,
        learning_rate=args.lr,
        num_iters=args.num_iters,
        save_every=args.save_every,
        print_every=args.print_every,
        init_from=args.init,
        pool_type=args.pool,
    ))
    print('hparams:', hparams)
    stylize(args.content, args.style, args.weights, args.output, hparams)


if __name__ == '__main__':
    main()
