import numpy as np


def softmax(x):
    """Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code. You might find numpy
    functions np.exp, np.sum, np.reshape, np.max, and numpy
    broadcasting useful for this task.

    Numpy broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html

    You should also make sure that your code works for a single
    N-dimensional vector (treat the vector as a single row) and
    for M x N matrices. This may be useful for testing later. Also,
    make sure that the dimensions of the output match the input.

    You must implement the optimization in problem 1(a) of the
    written assignment!

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        assert len(x.shape) == 2
        s = np.max(x, axis=1)
        s = s[:, np.newaxis]
        e_x = np.exp(x - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis]
        x = e_x / div
    else:
        # Vector
        x -= np.max(x)
        e_x = np.exp(x)
        x = e_x / np.sum(e_x,axis = 0)

    assert x.shape == orig_shape
    return x


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print "You should be able to verify these results by hand!\n"


def test_softmax():
    import tensorflow as tf
    """
    Use this space to test your softmax implementation by running:
        python q1_softmax.py
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    for i in range(10):
        for j in range(10):
            batch = np.random.rand(i+1, j+1)
            x = tf.placeholder(tf.float32, shape=[None, j+1])
            y = tf.nn.softmax(x)
            tf.global_variables_initializer()
            sess = tf.Session()
            tf_softmax = sess.run(y, feed_dict={x: batch})
            my_softmax = softmax(batch)
            assert np.allclose(tf_softmax, my_softmax, rtol=1e-05, atol=1e-06)
            print "##### tf_softmax: #####"
            print tf_softmax
            print "#### my softmax: #####"
            print my_softmax
            print "-----------------------------"


if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
