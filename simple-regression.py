import matplotlib.pyplot as plt
import numpy as np

# Initial function that we will try to approximate
def func(x):
    return 6 * x + 2

# At this point we forget we know the original function
# and return to it only after the calculations have been
# done for comparison

xrng = np.arange(-2, 2, 0.01)
yrng = func(xrng)

# Adding noise to the random linear
# This data will be used while approximating the original function
xxrng = np.random.uniform(-2, 2, 100)
yyrng = func(xxrng) + np.random.uniform(-3, 3, 100)

plt.plot(xrng, yrng)
plt.scatter(xxrng, yyrng, c='r')
plt.show()

# Root Mean Squared Error function
def mse(y, y_train, n):
    return np.sum((y - y_train) ** 2) / n

# Here we use the partial derivatives of the RMSE
def gradient_descent(weight, bias, aplha, x_train, y_train):
    # Partial differential results
    d_w = np.sum(-2 * x_train * (y_train - (weight * x_train + bias))) / n
    d_b = np.sum(-2 * (y_train - (weight * x_train + bias))) / n

    # Gradient descent formulas
    weight = weight - d_w * alpha
    bias = bias - d_b * alpha

    return (weight, bias)

xxrng = np.array(xxrng)
yyrng = np.array(yyrng)

# Some initialization
n = len(xxrng)
weight = 0          # Can be random
bias = 0            # Can be random
alpha = 0.01        # Learning step
epochs = 0
MAX_EPOCHS = 1000

while (epochs < MAX_EPOCHS):
    y = bias + weight * xxrng
    
    mean_sq_er = mse(y, yyrng, n)
    (weight, bias) = gradient_descent(weight, bias, alpha, xxrng, yyrng)
    
    epochs += 1

    # Logging changes in error
    if(epochs % 10 == 0):
        print("Mean squared error on epoch {}: {}".format(epochs, mean_sq_er))

# Final equation estimate
print("Estimated function: y = {} * x + {}".format(weight, bias))