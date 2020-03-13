import tensorflow as tf
import numpy as np

def MSE_np(a, b):
    result = np.square(a - b)
    result = np.mean(result)

    return result

def r_coefficient_np(a, b):
    a_avg = np.mean(a)
    b_avg = np.mean(b)
    numerator = np.sum((a-a_avg)*(b-b_avg))
    denominator = np.sqrt(np.sum(np.square(a-a_avg))*np.sum(np.square(b-b_avg)))

    return numerator/denominator

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

if __name__ == "__main__":
    pass