from __future__ import print_function

import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from loss import FocalLoss
from retinanet import RetinaNet
# from datagen import ListDataset
from dataset import ListDataset
from eval import evaluate
from empty_txt import empty_txt
from scripts.logger import *

from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='PyTorch RetinaNet Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs")###########
parser.add_argument("--image_size", type=int, default=300, help="size of images")
parser.add_argument('--lr_decay_step', dest='lr_decay_step',default=100, type=int, help='step to do learning rate decay')
parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma', default=0.1, type=float, help='learning rate decay ratio')
# parser.add_argument("--resume", default=False)

parser.add_argument("--dataset", type=str, default="composite_18.1_gmy", help="dataset name used for training")###########
parser.add_argument("--save_folder", type=str, default="checkpoints/retinanet_gas", help="folder name used for save weight when training")

args = parser.parse_args()

weight_save_folder=args.save_folder
if not os.path.exists(weight_save_folder):
    os.makedirs(weight_save_folder)
logger = Logger("logs/retinanet_gas")###################################

# assert torch.cuda.is_available(), 'Error: CUDA not found!'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch


# Data
print('==> Preparing data..')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

data_path = r"/home/ecust/txx/project/gmy_2080_copy/pytorch-retinanet-master/pytorch-retinanet-master/data/dataset/composite"


dataset_name=args.dataset
# ext="png"
ext="jpg"######################################
train_path = os.path.join(data_path,'{}/trainval/label_txt'.format(dataset_name))#trainval
test_path = os.path.join(data_path,'{}/test/label_txt'.format(dataset_name))


print("train_path={}".format(train_path))
print("test_path={}".format(test_path))

trainset = ListDataset(root=train_path, train=True, transform=transform, input_size=args.image_size,ext=ext)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=8, collate_fn=trainset.collate_fn)
print("len_train={}".format(len(trainset)))


# Model
net = RetinaNet()
net.load_state_dict(torch.load('./model/net.pth'))#####################

if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ckpt_78.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']

# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
net.to(device)

criterion = FocalLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

def adjust_learning_rate(optimizer, decay=0.1):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * param_group['lr']
# Training
def train():

    best_map = 0
    lr = args.lr
    # net.module.freeze_bn()
    # train_loss = 0
    for epoch in range(args.epochs):
        sum_loss = 0
        sum_loc_loss = 0
        sum_cls_loss = 0

        train_loss = 0
        net.train()


        if (epoch+1) % (args.lr_decay_step ) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr = lr*0.1
            print('Drop LR to {}'.format(lr))

        print('\nEpoch: %d' % epoch)
        for batch_idx, (inputs, loc_targets, cls_targets,_) in enumerate(trainloader):


            # inputs = Variable(inputs.cuda())
            # loc_targets = Variable(loc_targets.cuda())
            # cls_targets = Variable(cls_targets.cuda())
            inputs = inputs.to(device)
            loc_targets = loc_targets.to(device)
            cls_targets = cls_targets.to(device)
            # print(inputs.size(),loc_targets.size(),cls_targets.size())

            optimizer.zero_grad()
            loc_preds, cls_preds = net(inputs)
            # print(loc_preds.size(), cls_preds.size())
            loss,loc_loss,cls_loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # print('train_loss: %.3f | avg_loss: %.3f' % (loss.item(), train_loss/(batch_idx+1)))

            sum_loss += loss.item()
            sum_loc_loss += loc_loss.item()
            sum_cls_loss += cls_loss.item()


        if not os.path.isdir('checkpoints'):
            os.mkdir('checkpoints')

        if epoch % args.evaluation_interval == 0:
            # print("\n---- Evaluating Model ----")
            # map = evaluate(model=net,
            #                transform=transform,
            #                test_path=test_path,#########################
            #                batch_sizes=8,#####################
            #                thresh=0.01,###########0.05
            #                im_size=args.image_size,
            #                eval_classes= ['gas'],############################
            #                dataset_name=dataset_name,
            #                )
            # if map > best_map:
            if True:
                torch.save(net.state_dict(), os.path.join(weight_save_folder,"ckpt_{}.pth".format(epoch)))
                best_map = map


            # logs
            sum_loss/=len(trainloader)
            sum_loc_loss/=len(trainloader)
            sum_cls_loss/=len(trainloader)

            logs_metrics = [
                ("loss", sum_loss),
                ("loss_loc", sum_loc_loss),
                ("loss_cls", sum_cls_loss),
                # ("map_val", map),
            ]
            logger.list_of_scalars_summary(logs_metrics, epoch)


# Test
# def eval(epoch):
#     print('\nTest')
#     net.eval()
#     test_loss = 0
#     for batch_idx, (inputs, loc_targets, cls_targets,_) in enumerate(testloader):
#         # inputs = Variable(inputs.cuda(), volatile=True)
#         # loc_targets = Variable(loc_targets.cuda())
#         # cls_targets = Variable(cls_targets.cuda())
#         inputs = inputs.to(device)
#         loc_targets = loc_targets.to(device)
#         cls_targets = cls_targets.to(device)
#
#         with torch.no_grad():
#             loc_preds, cls_preds = net(inputs)
#             loss = criterion(loc_preds, loc_targets, cls_preds, cls_targets)
#         test_loss += loss.item()
#
#         print('test_loss: %.3f | avg_loss: %.3f' % (loss.item(), test_loss/(batch_idx+1)))
#
#     # Save checkpoint
#     global best_loss
#     # print(len(testloader))
#     test_loss /= len(testloader)
#     if test_loss < best_loss:
#         print('Saving..')
#         # state = {
#         #     # 'net': net.module.state_dict(),
#         #     'net': net.state_dict(),
#         #     'loss': test_loss,
#         #     'epoch': epoch,
#         # }
#         if not os.path.isdir('weights'):
#             os.mkdir('weights')
#         # torch.save(state, './checkpoint/ckpt_%d.pth' % (epoch+1))
#         torch.save(net.state_dict(), f"weights/ckpt_%d.pth" % (epoch + 1))
#         best_loss = test_loss


if __name__=='__main__':
    empty_txt()
    train()
