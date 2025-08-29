import tensorflow as tf
import keras


@keras.saving.register_keras_serializable()
def masked_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Custom loss function that masks the loss contribution of certain classes.
    Args:
        y_true: Ground truth labels, shape of [batch_size, num_classes].
        y_pred: Predicted logits, shape of [batch_size, num_classes].
    Returns:
        A scalar tensor representing the masked loss.
    """

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    loss = loss_fn(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
