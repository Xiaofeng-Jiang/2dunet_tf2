import tensorflow as tf


def _dice_coef_func(y_true, y_pred,
                    smooth=1, class_weight=1,
                    beta=1):
    """
    tp: True positive
    fp: False positive
    fn: False negative
    """
    y_true = tf.cast(y_true, tf.float32)
    tp = tf.reduce_sum(
        y_true * y_pred, axis=(0, 1, 2))
    fp = tf.reduce_sum(
        y_pred, axis=(0, 1, 2)) - tp
    fn = tf.reduce_sum(
        y_true, axis=(0, 1, 2)) - tp

    dice = (((1 + beta ** 2) * tp + smooth)
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth))
    dice = tf.reduce_mean(dice * class_weight)
    return dice


def dice_loss_func(y_true, y_pred, smooth=1, class_weight=1, beta=1):
    dice_coef = _dice_coef_func(
        y_true, y_pred,
        smooth=smooth,
        class_weight=class_weight,
        beta=beta)
    dice_loss = 1 - dice_coef
    return dice_loss
