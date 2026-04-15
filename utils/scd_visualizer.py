import matplotlib.pyplot as plt
import numpy as np

def save_batch_comparison(
        orig_imgs, pred_masks, gt_masks, index2color, out_path,
        class_names=None, overlay_alpha=0.5, epoch=None, metric_str=None,
        softmax_probs=None,    # 新增，list of HW ndarray（单通道最大softmax，或HW）
        dpi=200):
    """
    orig_imgs: list of HWC uint8 ndarray
    pred_masks, gt_masks: list of HW int ndarray
    index2color: callable, mask->RGB
    softmax_probs: list of HW float ndarray, softmax最大通道概率 or (HW, num_classes)
    out_path: 文件保存路径（不带后缀）
    class_names: 类别名列表
    overlay_alpha: overlay透明度
    epoch: epoch号（用于标题）
    metric_str: 指标字符串
    dpi: 分辨率
    """
    from matplotlib.patches import Patch

    n = len(orig_imgs)
    fig, axes = plt.subplots(n, 6, figsize=(22, 3*n), dpi=dpi)
    # 支持单样本特殊情况
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = orig_imgs[i]
        pred = pred_masks[i]
        gt = gt_masks[i]
        pred_color = index2color(pred)
        gt_color = index2color(gt)
        overlay = (img * (1-overlay_alpha) + pred_color * overlay_alpha).astype(np.uint8)

        # Softmax Map（用最大通道概率）
        if softmax_probs is not None and len(softmax_probs) > i:
            prob_map = softmax_probs[i]
            # 若为多通道（HWC），取最大通道
            if prob_map.ndim == 3:
                prob_map = prob_map.max(axis=-1)
            prob_vis = (prob_map * 255).astype(np.uint8)
        else:
            prob_vis = np.zeros_like(pred, dtype=np.uint8)

        # FP-FN Overlay
        fp_mask = ((pred != gt) & (pred > 0)).astype(np.uint8)
        fn_mask = ((pred != gt) & (gt > 0)).astype(np.uint8)
        fp_fn_overlay = img.copy()
        # FP用红色，FN用蓝色，混合像素为紫色
        fp_fn_overlay[fp_mask == 1] = [255, 0, 0]       # False Positive: Red
        fp_fn_overlay[fn_mask == 1] = [0, 0, 255]       # False Negative: Blue
        # 若同一像素同时FP和FN, 叠加为紫色
        both = (fp_mask == 1) & (fn_mask == 1)
        fp_fn_overlay[both] = [128, 0, 128]

        row_axes = axes[i]
        for j, (show, title) in enumerate([
            (img, "Image"),
            (pred_color, "Prediction"),
            (gt_color, "Ground Truth"),
            (overlay, "Overlay"),
            (prob_vis, "Softmax Map"),
            (fp_fn_overlay, "FP/FN Overlay"),
        ]):
            ax = row_axes[j]
            if j == 4:  # Softmax Map显示用热力图
                im = ax.imshow(show, cmap='jet', vmin=0, vmax=255)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                ax.imshow(show)
            if i == 0:
                ax.set_title(title, fontsize=13)
            ax.axis('off')

    # 添加总标题
    main_title = f"Epoch {epoch if epoch is not None else ''}   {metric_str if metric_str else ''}"
    plt.suptitle(main_title, fontsize=16, y=1.02)

    # 添加legend
    if class_names is not None:
        handles = [Patch(color=np.array(index2color(idx))/255, label=name)
                   for idx, name in enumerate(class_names)]
        # 再加上FP/FN图例
        handles.extend([
            Patch(color=np.array([1, 0, 0]), label='FP (red)'),
            Patch(color=np.array([0, 0, 1]), label='FN (blue)'),
            Patch(color=np.array([0.5, 0, 0.5]), label='FP+FN (purple)'),
        ])
        fig.legend(handles=handles, loc='center right', bbox_to_anchor=(1.13, 0.5), fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path + '_batch_vis.png', bbox_inches='tight')
    plt.close(fig)

    # 单独保存legend
    if class_names is not None:
        fig_legend = plt.figure(figsize=(2, (len(class_names)+3)*0.4), dpi=dpi)
        handles = [Patch(color=np.array(index2color(idx))/255, label=name)
                   for idx, name in enumerate(class_names)]
        handles.extend([
            Patch(color=np.array([1, 0, 0]), label='FP (red)'),
            Patch(color=np.array([0, 0, 1]), label='FN (blue)'),
            Patch(color=np.array([0.5, 0, 0.5]), label='FP+FN (purple)'),
        ])
        fig_legend.legend(handles=handles, loc='center', frameon=False)
        plt.axis('off')
        plt.savefig(out_path + '_legend.png', bbox_inches='tight')
        plt.close(fig_legend)
