import json
import numpy as np

import matplotlib.pyplot as plt

# Example JSON data
with open('mAP_tests.json', 'r') as f:
    json_data = f.read()
# Parse JSON data
data = json.loads(json_data)

# Convert the 1D list to a 10x10 numpy array
data_array = np.array(data)

# Create a meshgrid
x = np.arange(10) / 10 + 0.1
y = np.arange(10) / 10 + 0.1
X, Y = np.meshgrid(x, y)

# Plot the meshgrid
plt.figure(figsize=(8, 6))
plt.pcolormesh(X, Y, data_array, shading='auto')
plt.colorbar(label='mAP')
plt.title('mAP values for different confidence and NMS thresholds')
plt.ylabel('confidence threshold')
plt.xlabel('NMS threshold')
plt.show()