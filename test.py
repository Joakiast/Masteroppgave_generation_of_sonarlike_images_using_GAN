

import matplotlib.pyplot as plt
import numpy as np

# Generate a random image
random_image = np.random.rand(10, 10)

# Plot the random image
plt.imshow(random_image, cmap='viridis')
plt.colorbar()
plt.show()
