#!/usr/bin/env python

import numpy as np


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    # s -- sigmoid(x)
    """

    s = 1+np.exp(-x)
    s = 1/s
    return s


def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ds = np.multiply(s, (1 - s))

    return ds


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print f
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print g
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print "You should verify these results by hand!\n"


def test_sigmoid():
    import tensorflow as tf
    """
    Use this space to test your sigmoid implementation by running:
        python q2_sigmoid.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    for i in range(10):
        for j in range(10):
            batch = np.random.rand(i+1, j+1)
            x = tf.placeholder(tf.float32, shape=[None, j+1])
            y = tf.nn.sigmoid(x)
            tf.global_variables_initializer()
            sess = tf.Session()
            tf_sigmoid = sess.run(y, feed_dict={x: batch})
            my_sigmoid = sigmoid(batch)
            assert np.allclose(tf_sigmoid, my_sigmoid, rtol=1e-05, atol=1e-06)
            print "##### tf_sigmoid: #####"
            print tf_sigmoid
            print "#### my my_sigmoid: #####"
            print my_sigmoid
            print "-----------------------------"


if __name__ == "__main__":
    test_sigmoid_basic();
    test_sigmoid()
