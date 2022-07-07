import os
import argparse
import keras
import keras.losses
import keras.optimizers
import tensorflow as tf
from models.vggnet import VGG11
from dataset import DataSequence, DataGenerator

using_gpu_index = 0  # 使用的 GPU 号码
gpu_list = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_list) > 0:
    try:
        # 设置GPU使用内存根据模型增长
        tf.config.experimental.set_memory_growth(gpu_list[using_gpu_index], True)
        # 限制最大GPU使用内存
        # tf.config.experimental.set_virtual_device_configuration(
        #     gpu_list[using_gpu_index],
        #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)]
        # )
    except RuntimeError as e:
        print(e)
else:
    print("Got no GPUs")


def train(opt):
    path_train, path_val, model_dir, model_name = opt.path_train, opt.path_val, opt.model_dir, opt.model_name
    batch_size, workers, img_size, lr, start_epoch, epochs, eval_epoch = \
        opt.batch_size, opt.workers, opt.img_size, opt.lr, opt.start_epoch, opt.epochs, opt.eval_epoch
    nd = len(gpu_list)  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    if nw == 0:
        worker_flag = False
    else:
        worker_flag = True
    # 定义数据集
    train_data = DataSequence(path_train, img_size, batch_size, shuffle_flag=True)
    # train_data = DataGenerator(path_train, img_size, batch_size, shuffle=True)
    train_steps = len(os.listdir(path_train)) // batch_size
    val_data = DataSequence(path_val, img_size, batch_size, shuffle_flag=False)
    # train_data = DataGenerator(path_val, img_size, batch_size, shuffle=False)
    val_steps = len(os.listdir(path_val)) // batch_size
    # 定义网络
    model = VGG11()
    # 定义优化器与损失函数
    model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    # 是否恢复训练
    model_path = os.path.join(model_dir, model_name)
    if opt.resume and os.path.exists(model_path):
        model.build((None, 224, 224, 3))
        model.load_weights(model_path)
    # 开始训练
    for epoch in range(start_epoch, epochs // eval_epoch):
        history = model.fit(x=train_data, steps_per_epoch=train_steps, epochs=eval_epoch, shuffle=True,
                            use_multiprocessing=worker_flag, workers=nw, validation_data=val_data,
                            validation_steps=val_steps)
        val_acc = history.history['val_accuracy'][-1]
        print(f'acc: {val_acc}')
        # 模型保存
        model_save_name = 'model_{}.h5'.format(epoch * eval_epoch)
        model_save_path = os.path.join(model_dir, model_save_name)
        # model.save(model_save_path)       # 继承keras.Mode模型无法使用model.save()
        model.save_weights(model_save_path)
    model.save_weights(model_path)


def val(opt):
    path_val, model_dir, model_name = opt.path_val, opt.model_dir, opt.model_name
    batch_size, workers, img_size = opt.batch_size, opt.workers, opt.img_size
    model_path = os.path.join(model_dir, model_name)
    model = VGG11()
    # 定义优化器与损失函数
    model.compile(optimizer=keras.optimizers.adam_v2.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999),
                  loss=keras.losses.binary_crossentropy,
                  metrics=['accuracy'])
    model.build((None, 224, 224, 3))
    model.load_weights(model_path)
    nd = len(gpu_list)  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    if nw == 0:
        worker_flag = False
    else:
        worker_flag = True
    val_data = DataSequence(path_val, img_size, batch_size, shuffle_flag=False)
    # train_data = DataGenerator(path_val, img_size, batch_size, shuffle=False)
    val_steps = len(os.listdir(path_val)) // batch_size
    model.evaluate(x=val_data, steps=val_steps, use_multiprocessing=worker_flag, workers=nw)


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_train', type=str, default='E:/DataSources/DogsAndCats/train', help='dataset.yaml path')
    parser.add_argument('--path_val', type=str, default='E:/DataSources/DogsAndCats/val', help='dataset.yaml path')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='model_1.h5')

    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--eval_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--img_size', type=int, default=224, help='train, val image size (pixels)')
    parser.add_argument('--batch_size', type=int, default=4, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


if __name__ == '__main__':
    option = parse_opt(True)
    # train(option)
    val(option)
