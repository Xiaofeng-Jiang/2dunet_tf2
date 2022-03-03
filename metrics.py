import tensorflow as tf


def dice_coef(y_true, y_pred, smooth=1, threshold=0.5):
    prediction = tf.where(y_pred > threshold, 1, 0)
    prediction = tf.cast(prediction, dtype=y_true.dtype)
    ground_truth_area = tf.reduce_sum(
        y_true, axis=(1, 2, 3))
    prediction_area = tf.reduce_sum(
        prediction, axis=(1, 2, 3))
    intersection_area = tf.reduce_sum(
        y_true * y_pred, axis=(1, 2, 3))
    combined_area = ground_truth_area + prediction_area
    dice = tf.reduce_mean(
        (2 * intersection_area + smooth) / (combined_area + smooth))
    return dice
