import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

v1 = np.random.rand(49, 10).flatten()
v2 = np.random.rand(49, 10).flatten()
v3 = np.random.rand(49, 10).flatten()
arrays = [v1, v2, v3]
data = np.stack(arrays, axis=0).flatten()

sns.displot(data, kde=True)
plt.savefig('keyword_spotting/tmp.png')