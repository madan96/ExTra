import numpy as np
from pyemd import emd
from pyemd import emd_with_flow
from pyemd import emd_samples

s1 = 8
s2 = 8
np.random.seed(10)

a = np.random.rand(s1)
b = np.random.rand(s2)
d = np.random.rand(s1, s2)

result1 = emd(a, b, d)
result2 = emd_with_flow(a, b, d)
result3 = emd_samples(a, b)

print (result1)
print ("\n", result2)
print (result3)