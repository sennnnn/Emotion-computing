import tensorflow as tf

def MSE(a, b):
    """
    mean squared error.
    Args:
        a,b:items that wait to be input in the calculation.
    Return:
        result:the mean squared error between a and b.
    """
    result = tf.square(a - b)
    result = tf.reduce_mean(result)

    return result

def r_coefficient(a, b, smooth=0.000001):
    """
    Pearson Correlation Coefficient
    This coefficient can measure the intense of the linear relation between a and b.
    Args:
        a,b:items that wait to be input in the calculation.
    Return:
        result:the Pearson Correlation Coefficient between a and b.
    """
    a_avg = tf.reduce_mean(a)
    b_avg = tf.reduce_mean(b)
    numerator = tf.reduce_sum((a-a_avg)*(b-b_avg), keepdims=False)
    denominator = tf.sqrt(tf.reduce_sum(tf.square(a-a_avg))*tf.reduce_sum(tf.square(b-b_avg)))

    return (numerator + smooth)/(denominator + smooth)