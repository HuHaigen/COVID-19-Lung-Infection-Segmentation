import os
import shutil
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets.data3dunet import MyPatientFor3DUNet
from models.RPLFUnet import RPLFUnet
from resources.config import get_configs
from losses.losses import get_loss_criterion, DiceAccuracy

from trainer.trainerunet3d import UNet3DTrainer
from trainer.utils import get_logger

config = get_configs()


def _create_optimizer(config, model):
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _get_loaders(in_channels=1, out_channels=1, label_type='float32'):
    train_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='train',
                                       data_type=label_type)
    val_dataset = MyPatientFor3DUNet(in_channels=in_channels, out_channels=out_channels, step='val',
                                     data_type=label_type)
    return {
        'train': DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=1, shuffle=False)
    }


def _para_():
    print("数据集路径: {}".format(config.dataset_path))

    print("batch_size: {}".format(config.batch_size))
    print("初始学习率: {}".format(config.learning_rate))
    print("weight_decay: {}".format(config.weight_decay))
    print("继续训练: {}".format(config.resume))
    print("模型保存路径: {}".format(config.checkpoint_dir))

    print("损失权重: {}".format(config.loss_weight))
    print("学习率调整间隔epoch: {}".format(config.patience))
    print("最大epoch: {}".format(config.epochs))
    print("模型初始特征图数量: {}".format(config.init_channel_number))

    print("损失函数: {}".format(config.loss))
    print("输出通道: {}".format(config.out_channels))
    print("sigmoid还是softmax: {}".format(config.final_sigmoid))
    print("测试模型路径: {}".format(config.model_path))


def main():
    path = config.dataset_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loss_weight = None
    loss_criterion = get_loss_criterion(
        config.loss, loss_weight, config.ignore_index)

    accuracy_criterion = DiceAccuracy(shrehold=0.5)

    label_dtype = 'float32'

    if config.model == 'UResNet':
        logger = get_logger('UNet3DTrainer')
        model = RPLFUnet(1)
        print(model)
        model = model.to(device)
        loaders = _get_loaders(in_channels=config.in_channels,
                               out_channels=config.out_channels, label_type=label_dtype)

    # Create the optimizer
    optimizer = _create_optimizer(config, model)
    log_and_vaild = len(loaders['train'])
    _para_()
    print("训练集数量: ", len(loaders['train']))
    print("测试集数量: ", len(loaders['val']))
    if config.pertrain_path is not None:
        if os.path.exists("./checkpoint/logs/"):
            shutil.rmtree("./checkpoint/logs/")
        os.makedirs("./checkpoint/logs/")
        logger.info("from pertrain model: {}".format(config.pertrain_path))
        trainer = UNet3DTrainer.from_pertrain(config.pertrain_path, model, optimizer,
                                              loss_criterion,
                                              accuracy_criterion,
                                              device,
                                              loaders,
                                              config.checkpoint_dir,
                                              max_num_epochs=config.epochs,
                                              max_num_iterations=config.iters,
                                              max_patience=config.patience,
                                              patience=config.patience,
                                              validate_after_iters=log_and_vaild,
                                              log_after_iters=log_and_vaild,
                                              logger=logger)
    else:
        if config.resume:
            logger.info("from last model: {}".format(config.checkpoint_dir))
            trainer = UNet3DTrainer.from_checkpoint(config.resume, model, optimizer,
                                                    loss_criterion,
                                                    accuracy_criterion,
                                                    loaders,
                                                    validate_after_iters=log_and_vaild,
                                                    log_after_iters=log_and_vaild,
                                                    logger=logger)
        else:
            if os.path.exists("./checkpoints/logs/"):
                shutil.rmtree("./checkpoints/logs/")
            os.makedirs("./checkpoints/logs/")
            logger.info("new train.")
            trainer = UNet3DTrainer(model, optimizer,
                                    loss_criterion,
                                    accuracy_criterion,
                                    device, loaders, config.checkpoint_dir,
                                    max_num_epochs=config.epochs,
                                    max_num_iterations=config.iters,
                                    max_patience=config.patience,
                                    patience=config.patience,
                                    validate_after_iters=log_and_vaild,
                                    log_after_iters=log_and_vaild,
                                    logger=logger)

    trainer.fit()


if __name__ == '__main__':
    main()
