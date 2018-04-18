import tensorflow as tf
import numpy as np

def GroupNormalization(inputs, 
                       group, 
                       N_axis=0, 
                       C_axis=-1, 
                       momentum=0.9,
                       epsilon=1e-3,
                       training=False,
                       name=None):
    """ Group normalization implementation with tensorflow.

    As descriped in Wu's paper(http://arxiv.org/abs/1803.08494), we can implement a 
    group norm with existed batch norm routine.

    The tensorflow code in this paper:
    ```python
    def GroupNorm(x, gamma, beta, G, eps=1e-5):
        # x: input features with shape [N,C,H,W]
        # gamma, beta: scale and offset, with shape [1,C,1,1]
        # G: number of groups for GN
        N, C, H, W = x.shape
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, [N, C, H, W])
        return x * gamma + beta
    ```

    It is easy to know that when `x` is reshaped as [N, G, C//G, H, W], we can implement
    it with batch norm in tensorflow:
    ```python
    tf.layers.batch_normalization(x, axis=[0, 1])
    ```

    Params
    ------
    `inputs`: tensor input
    `group`: number of groups in group norm
    `N_axis`: axis number of batch axis
    `C_axis`: axis number of channel axis
    `momentum`: momentum used in moving average mean and moving average variance
    `epsilon`: a small value to prevent divided by zero
    `training`: either a Python boolean, or a Tensorflow boolean scalar tensor (e.g. a 
    placeholder). Whether to return the output in training mode or in inference mode.
    **Note:** make sure to set this parameter correctly, or else your training/inference
    will not work properly.
    `name`: string, the name of the layer
    
    Returns
    -------
    Output tensor.
    """
    with tf.variable_scope(name, "GroupNorm"):
        input_shape = inputs.get_shape().as_list()
        ndims = len(input_shape)
        if not ndims in [4, 5]:
            raise ValueError("Not supported input dimension. Only 3 or 4")

        if not isinstance(C_axis, int):
            raise ValueError('`C_axis` must be an integer. Now it is {}'.format(C_axis))
        
        # Check axis
        if C_axis < 0:
            C_axis = ndims + C_axis
        if C_axis < 0 or C_axis >= ndims:
            raise ValueError('Invalid axis: %d' % C_axis)
        if N_axis < 0:
            N_axis = ndims + N_axis
        if N_axis < 0 or N_axis >= ndims:
            raise ValueError('Invalid axis: %d' % N_axis)

        # Require C % G == 0
        if input_shape[C_axis] % group != 0 or input_shape[C_axis] < group:
            raise ValueError('`group` should less than C_shape and be dividable '
                             'by C_shape. `group` is %d and C_shape is %d'
                             % (group, input_shape[C_axis]))

        permutation = [N_axis, C_axis] + [i for i in range(ndims) if i != C_axis and i != N_axis]
        inputs = tf.transpose(inputs, perm=permutation)
        
        old_shape = tf.shape(inputs)
        old_shape_val = inputs.get_shape().as_list()
        if ndims == 4:
            new_shape = [old_shape_val[0], group, old_shape_val[1] // group, old_shape[2], old_shape[3]]
        elif ndims == 5:
            new_shape = [old_shape_val[0], group, old_shape_val[1] // group, old_shape[2], old_shape[3], old_shape[4]]
        
        inputs = tf.reshape(inputs, shape=new_shape)

        outputs = tf.layers.batch_normalization(inputs,
                                                axis=[0, 1],
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                training=training)

        outputs = tf.reshape(outputs, shape=old_shape)
        
        reverse_permutation = permutation[:]
        for i, idx in enumerate(permutation):
            reverse_permutation[i] = permutation[idx]
        
        outputs = tf.transpose(outputs, perm=reverse_permutation)

        return outputs


# Alias
group_normalization = GroupNormalization
group_norm = GroupNormalization
batch_norm = tf.layers.batch_normalization
