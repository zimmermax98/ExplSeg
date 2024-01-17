import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from datasets import Cityscapes, VOCSegmentation

dataset = "cityscapes"

if dataset == "cityscapes":
    legend_elements = []
    for c in Cityscapes.classes:
        if c.train_id != 255:
            color = [color_channel / 255 for color_channel in c.color]
            label = f"[{c.train_id}] {c.name}"
            legend_elements.append(Patch(facecolor=color, label=label))

elif dataset == "voc":
    legend_elements = []
    for class_index, class_name in enumerate(VOCSegmentation.class_names):
        
        color = VOCSegmentation.cmap[class_index] / 255
        label = f"[{class_index}] {class_name}"
        legend_elements.append(Patch(facecolor=color, label=label))

fig, ax = plt.subplots()
ax.legend(handles=legend_elements, loc='center')
ax.axis("off")

plt.savefig(f"class_colors_{dataset}.png", dpi=600, bbox_inches='tight')