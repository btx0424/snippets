import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

def lemniscate(t: float, c: float):
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    sin2p1 = sin_t ** 2 + 1
    x = torch.tensor([cos_t, sin_t * cos_t, c * sin_t]) / sin2p1
    return x

t = torch.linspace(0, 2 * math.pi, 100).tolist()
c = 0.5
x, y, z = torch.stack([lemniscate(i, c) for i in t]).unbind(1)
plt.plot(x, y)
plt.axis('equal')
plt.show()