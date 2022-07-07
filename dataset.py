import os
import cv2
import numpy as np
from keras.utils.data_utils import Sequence


def letterbox(im, new_shape=(224, 224), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


# 使用Sequence设置数据集
class DataSequence(Sequence):
    def __init__(self, path, img_size, batch_size=8, shuffle_flag=True):
        self.path = [os.path.join(path, i) for i in os.listdir(path)]
        self.img_size = img_size
        self.batch_size = batch_size
        self.indices = np.arange(len(self.path))
        if shuffle_flag:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        data = []
        label = []
        excerpt = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        for idx in excerpt:
            img_path = self.path[idx]
            img_name = os.path.basename(img_path)
            img_label = img_name.split('.')[0]
            lbl = 0 if img_label == 'cat' else 1
            lbl = np.array([lbl], dtype=np.float32)
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            img, _, _ = letterbox(img, self.img_size, auto=False)
            img = img / 255  # 0-255 -> 0-1
            # img = (img - 0.5) / 0.5     # 0-1 -> -1-1
            img = np.float32(img)
            data.append(img)
            label.append(lbl)
        data = np.array(data)
        label = np.array(label)
        return data, label

    def __len__(self):
        return len(self.path) // self.batch_size


# 使用generator设置数据集
def DataGenerator(path, img_size, batch_size=8, shuffle=True):
    while True:
        img_dir = [os.path.join(path, i) for i in os.listdir(path)]
        indices = np.arange(len(img_dir))
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(len(img_dir) // batch_size):
            data = []
            label = []
            excerpt = indices[start_idx * batch_size:(start_idx + 1) * batch_size]
            for idx in excerpt:
                img_path = img_dir[idx]
                img_name = os.path.basename(img_path)
                img_label = img_name.split('.')[0]
                lbl = 0 if img_label == 'cat' else 1
                lbl = np.array([lbl], dtype=np.float32)
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img, _, _ = letterbox(img, img_size, auto=False)
                img = img / 255  # 0-255 -> 0-1
                # img = (img - 0.5) / 0.5     # 0-1 -> -1-1
                img = np.float32(img)
                data.append(img)
                label.append(lbl)
            data = np.array(data)
            label = np.array(label)
            yield data, label


if __name__ == '__main__':
    test = DataGenerator('E:/DataSources/DogsAndCats/train', 224)
    a = next(test)
    print(a)
