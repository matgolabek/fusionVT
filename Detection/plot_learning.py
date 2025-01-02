import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int, default=500, help="number of epochs to plot")
parser.add_argument("-d", type=str, default="logs/log_026.txt", help="path to log file")
opt = parser.parse_args()
print(opt)

# File path to your text file
file_path = opt.d

# Define the regular expression pattern to match each line
pattern1 = re.compile(
    r"\[Epoch (\d+)/(\d+), Batch (\d+)/(\d+)\]"
)

pattern2 = re.compile(
    r"x ([\d\.e-]+), y ([\d\.e-]+), w ([\d\.e-]+), h ([\d\.e-]+), "
    r"conf ([\d\.e-]+), cls ([\d\.e-]+), total ([\d\.e-]+), "
    r"recall: ([\d\.e-]+), precision: ([\d\.e-]+)"
)

pattern3 = re.compile(
    r"C0_AP: ([\d\.e-]+), C1_AP: ([\d\.e-]+), C2_AP: ([\d\.e-]+), C3_AP: ([\d\.e-]+), "
    r"mAP: ([\d\.e-]+)]"
)

filee = open(file_path, 'r')
epoch, max_epochs, batch, max_batch = (pattern1.search(filee.readline())).groups()
max_epochs = int(max_epochs)
max_batch = int(max_batch)
filee.close()

things_to_log = 9

# np.ndarray to store values
database = np.zeros(shape=(max_epochs,max_batch,things_to_log))

mAPDatabase = np.zeros(shape=(max_epochs, 5))

# Read the file and parse each line
with open(file_path, 'r') as file:
    for line in file:
        match1 = pattern1.search(line)
        match2 = pattern2.search(line)
        match3 = pattern3.search(line)
        if match1:
            epoch, _, batch, __ =  match1.groups()
        elif match2:
            # Extract values from each group in the regex match
            for ids, log_val in enumerate(match2.groups()):
                database[int(epoch),int(batch),ids] = float(log_val)
        elif match3:
            # c0, c1, c2, c3, c4, mAP = match3.groups()
            mAPDatabase[int(epoch)] = match3.groups()


epo_max = min(opt.e,max_epochs)
epochs = range(epo_max)

x_mean = np.mean(database[:epo_max,:,0],axis=1)
y_mean = np.mean(database[:epo_max,:,1],axis=1)
w_mean = np.mean(database[:epo_max,:,2],axis=1)
h_mean = np.mean(database[:epo_max,:,3],axis=1)
conf_mean = np.mean(database[:epo_max,:,4],axis=1)
cls_mean = np.mean(database[:epo_max,:,5],axis=1)
total_mean = np.mean(database[:epo_max,:,6],axis=1)
recall_mean = np.mean(database[:epo_max,:,7],axis=1)
prec_mean = np.mean(database[:epo_max,:,8],axis=1)


# Plot total loss over batches
plt.figure(figsize=(10, 6))
plt.step(epochs, total_mean, label='Mean Total Loss', color='blue')
plt.xlabel("Epoch")
plt.ylabel("Mean Total Loss")
plt.title("Mean Total Loss per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot individual loss components over batches
plt.figure(figsize=(10, 6))
plt.step(epochs, x_mean, label='x loss', color='red')
plt.step(epochs, y_mean, label='y loss', color='green')
plt.step(epochs, w_mean, label='w loss', color='purple')
plt.step(epochs, h_mean, label='h loss', color='orange')
plt.step(epochs, conf_mean, label='Confidence loss', color='brown')
plt.step(epochs, cls_mean, label='Class loss', color='pink')
plt.xlabel("Epoch")
plt.ylabel("Mean Loss")
plt.title("Mean Loss Components per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot recall and precision over batches
plt.figure(figsize=(10, 6))
plt.step(epochs, recall_mean, label='Recall', color='cyan')
plt.step(epochs, prec_mean, label='Precision', color='magenta')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Mean Recall and Precision per Epoch")
plt.legend()
plt.grid(True)
plt.show()

# Plot class averages and map mean
plt.figure(figsize=(10, 6))
plt.step(epochs, mAPDatabase[:epo_max, 0], label='fire extinguisher', color='#e01616')
plt.step(epochs, mAPDatabase[:epo_max, 1], label='backpack', color="#f2d777")
plt.step(epochs, mAPDatabase[:epo_max, 2], label='human', color='#83db18')
plt.step(epochs, mAPDatabase[:epo_max, 3], label='drill', color='#c714af')
plt.step(epochs, mAPDatabase[:epo_max, 4], label='mAP', color='black')
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("mAP")
plt.legend()
plt.grid(True)
plt.show()
