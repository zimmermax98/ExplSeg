import os
import shutil

path = '/fastdata/MT_ExplSeg/datasets/oxford-iiit-pet/'

path_images = os.path.join(path,'images')
path_train_split = os.path.join(path,'annotations/trainval.txt')
path_test_split = os.path.join(path,'annotations/test.txt')
train_save_path = os.path.join(path,'dataset/train/')
test_save_path = os.path.join(path,'dataset/test/')
 
train_images = []
with open(path_train_split,'r') as f:
    for line in f:
        train_images.append(line.strip('\n').split(' ')[0])
test_images = []
with open(path_test_split,'r') as f:
    for line in f:
        test_images.append(line.strip('\n').split(' ')[0])

for train_image in train_images:
    class_name = train_image.rsplit("_", 1)[0]
    image_name = train_image + ".jpg"
    class_dir_path = train_save_path + class_name
    if not os.path.exists(class_dir_path):
        os.mkdir(class_dir_path)
    shutil.copy(os.path.join(path_images, image_name), os.path.join(class_dir_path, image_name))

for test_image in test_images:
    class_name = test_image.rsplit("_", 1)[0]
    image_name = test_image + ".jpg"
    class_dir_path = test_save_path + class_name
    if not os.path.exists(class_dir_path):
        os.mkdir(class_dir_path)
    shutil.copy(os.path.join(path_images, image_name), os.path.join(class_dir_path, image_name))
