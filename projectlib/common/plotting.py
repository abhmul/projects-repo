import numpy as np
import matplotlib.pyplot as plt


def plot_labeled_scatter(X, y):
    """
    Plot labeled data in a scatter plot
    :param X: input data (as numpy array N x F)
    :param y: given integer labels (as numpy array N)
    :return:
    """
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.bwr)
    return plt


def plot_decision_boundary(pred_func, X, y, padding=0.5, save_fname="fig.png"):
    """
    plot the decision boundary
    :param pred_func: function used to predict the label (numpy -> numpy)
    :param X: input data (as numpy array)
    :param y: given labels (as numpy array)
    :return:
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, alpha=0.1, cmap=plt.cm.bwr)
    plot_labeled_scatter(X, y)
    plt.savefig(save_fname)
