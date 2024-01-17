import numpy as np
import matplotlib.pyplot as plt

#methods = ('Grad-CAM', 'ProtoPNet', 'PIP-Net')
#correct_mean, correct_std = (72.4, 73.2, 55.5), (21.5, 24.9, 4.6)
#incorrect_mean, incorrect_std = (32.8, 46.4, 75.4), (24.3, 35.9, 3.4)

#methods = ('Grad-CAM', 'ProtoPNet')
#correct_mean, correct_std = (72.4, 73.2), (21.5, 24.9)
#incorrect_mean, incorrect_std = (32.8, 46.4), (24.3, 35.9)

methods = ('PIP-Net Seg. (ours)', 'ProtoSeg', 'L-CRP')
correct_mean, correct_std = (81.4, 68.3, 41.5), (8.8, 5.6, 7.8)
incorrect_mean, incorrect_std = (84.4, 75.0, 78.0), (12.1, 12.4, 0.8)

ind = np.arange(len(correct_mean))  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind - width/2, correct_mean, width,
                label='Correct', capsize=10, edgecolor="black")
rects2 = ax.bar(ind + width/2, incorrect_mean, width,
                label='Incorrect', capsize=10, edgecolor="black")

ax.set_title("Agreement Task (Segmentation)")
ax.set_xticks(ind)
ax.set_xticklabels(methods)
ax.set_ylabel('Accuracy (%)')
ax.legend()
plt.grid()
plt.ylim(0, 100)
#plt.savefig('TestX.png', dpi=300, bbox_inches='tight')
plt.show()