import matplotlib

import numpy as np

import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])

A_inverse = np.linalg.inv(A)
identity_1 = np.dot(A, A_inverse)
identity_2 = np.dot(A_inverse, A)

print("Inverse of A:")
print(A_inverse)

print("\nA * A^-1:")
print(identity_1)

print("\nA^-1 * A:")
print(identity_2)
