import matplotlib.pyplot as plt
import numpy as np

labels = ['Jetson Xavier NX', 'Kria KV260', 'Intel Core i7-1165G7']
t_without_YOLO = [2.141, 0.1258, 0.3534]
t_with_YOLO = [2.165, 1.04, 0.3597]
mAP = [0.2004, 0.1154, 0.2579]

plt.figure(figsize=(10, 5))
plt.suptitle('Precision vs Time for different devices')
plt.subplot(1, 2, 1)
plt.scatter(t_without_YOLO[0], mAP[0])
plt.scatter(t_without_YOLO[1], mAP[1])
plt.scatter(t_without_YOLO[2], mAP[2])
plt.legend(labels)
plt.axis([0, 3, 0, 0.3])
plt.grid()
plt.title("Time without YOLO Layer")
plt.xlabel('Time (s)')
plt.ylabel('mAP')

plt.subplot(1, 2, 2)
plt.scatter(t_with_YOLO[0], mAP[0])
plt.scatter(t_with_YOLO[1], mAP[1])
plt.scatter(t_with_YOLO[2], mAP[2])
plt.legend(labels)
plt.axis([0, 3, 0, 0.3])
plt.grid()
plt.title("Time with YOLO Layer")
plt.xlabel('Time (s)')
plt.ylabel('mAP')


plt.show()