import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

y = np.array([0,1,2,1,1,3])
n_classes = len(np.unique(y))
n_samples = len(y)
y_encoded = np.array(
    [np.zeros(n_classes) for value in range(n_samples)]
)
y_encoded[np.arange(n_samples), y] = 5

print(y_encoded)