'''
evaluate tools for voc
'''
#voc for evaluate tool
import torch
from retinanet import RetinaNet
# from datagen import ListDataset
from dataset import ListDataset
from encoder import DataEncoder
from PIL import Image,ImageDraw
import os
import glob
import tqdm
import pickle
import numpy as np
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    size_info = tree.find('size')

    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['height'] = float(size_info.find('height').text)
        obj_struct['width'] = float(size_info.find('width').text)
        obj_struct['bbox'] = [300*float(bbox.find('xmin').text) / obj_struct['width'],
                              300*float(bbox.find('ymin').text) / obj_struct['height'],
                              300*float(bbox.find('xmax').text) / obj_struct['width'],
                              300*float(bbox.find('ymax').text) / obj_struct['height']]
        objects.append(obj_struct)

    return objects

def parse_rec_(filename):
    """ Parse a PASCAL VOC txt file """
    with open (filename) as f:
        lines=f.readlines()
    for line in lines:
        line.strip()
        obj_struct = {}
        obj_struct ['name']=line.split(':')[0]
        obj_struct['pose'] = 'Unspecified'
        obj_struct['truncated'] = int(0)
        obj_struct['difficult'] = int(0)



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]
        #print(i)
        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            dataset_name,
             ovthresh=0.5,
             use_07_metric=False):

    paths='eval/result/{}'.format(dataset_name)######################################################################
    if not os.path.isdir(paths):
        os.makedirs(paths)
    cachefile = os.path.join(paths, 'annots.pkl')

    a = sorted(glob.glob("%s/*.*" % imagesetfile))
    imagenames = [path.split('/')[-1].split('.')[0].strip() for path in a]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec(annopath % (imagename))
            if i % 1000 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                   i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R) # 该图片中该类别对应的所有bbox的是否已被匹配的标志位
        npos = npos + sum(~difficult)#累计所有图片中的该类别目标的总数，不算diffcult,这里计算还是很巧妙的，npos=TP+FN
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets     读取相应类别的检测结果文件，每一行对应一个检测目标
    detfile = detpath%(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence  按置信度由大到小排序
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)# 检测结果文件的行数
        tp = np.zeros(nd) # 用于标记每个检测结果是tp还是fp
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps 计算与该图片中所有ground truth的最大重叠度
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def evaluation(net, batch_sizes, testloader, the_classes, encoder, thresh, w, h, det_folder):
    the_det_file = {}

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for batch_idx, (inputs, loc_targets, cls_targets,fname) in enumerate(tqdm.tqdm(testloader, desc="Detecting objects")):
        # print("*batch_idx={}".format(batch_idx))
        inputs = inputs.cuda()

        _t['im_detect'].tic()
        loc_preds, cls_preds = net(inputs)
        # print(loc_preds, cls_preds)
        # print('Decoding..')
        boxes, labels,scores = encoder.decode(loc_preds.data, cls_preds.data,thresh, (w, h))
        detect_time = _t['im_detect'].toc(average=True)
        # print('current_im_detect: {:d}/{:d} {:.3f}s'.format(batch_idx + 1, len(testloader), detect_time))

        # print( boxes, labels,scores)
        boxes=boxes.unsqueeze(0)
        labels=labels.unsqueeze(0)
        scores=scores.unsqueeze(0)
        for index in range(batch_sizes):
            # print(index,"/",batch_sizes)
            file_name=fname[index]
            # print("**",boxes[index])
            try:
                box=boxes[index]
            except:
                # print('not found target')
                # print(fname[index])
                continue
            label=labels[index]
            score=scores[index]
            # try:
            #     tmp=box[0,:]
            # except:
            #     box=box.unsqueeze(0)
            #     label=label.unsqueeze(0)
            #     score=score.unsqueeze(0)
            for i,bb in enumerate(box):
                if(label[i].item() in the_det_file.keys()):
                    # print(the_classes[int(label[i].item())])
                    the_det_file[label[i].item()].append([file_name,str(score[i].item()),str(bb[0].item()),str(bb[1].item()),str(bb[2].item()),str(bb[3].item())])
                else:
                    the_det_file[label[i].item()]=[[file_name,str(score[i].item()),str(bb[0].item()),str(bb[1].item()),str(bb[2].item()),str(bb[3].item())]]

    # print(the_det_file.items())
    # print("-"*50)
    # print(det_folder)
    # print(len(the_det_file.items()))
    for class_index,info in the_det_file.items():
        # print("***")
        class_name = the_classes[int(class_index)]
        # print(the_classes)
        # print(os.path.join(def_folder, class_name + '.txt'))
        with open(os.path.join(det_folder, class_name + '.txt'), 'w') as f:
            # print(111)
            for det_info in info:
                f.writelines(' '.join(det_info))
                f.write('\n')
            f.close()

    #calculate mAP respectively
def eval(the_classes,dataset_source,dataset_name,det_folder,set_name):
    det_path=os.path.join(det_folder,"%s.txt")
    # annotation_path='/home/ecust/txx/project/gmy_2080_copy/pytorch-retinanet-master/pytorch-retinanet-master/data/IR/valid/label/%s.xml'##########
    # imagesetfile='/home/ecust/txx/project/gmy_2080_copy/pytorch-retinanet-master/pytorch-retinanet-master/data/IR/valid/image'#############
    annotation_path='data/dataset/{}/{}/{}/label/%s.xml'.format(dataset_source,dataset_name,set_name)##########
    imagesetfile='data/dataset/{}/{}/{}/image'.format(dataset_source,dataset_name,set_name)#############
    aps=[]
    for class_name in the_classes:
        rec, prec, ap=voc_eval(det_path,annotation_path,imagesetfile,class_name,dataset_name)
        print('%s AP: %f'%(class_name,ap))
        aps += [ap]
    print('MAP = {:.4f}'.format(np.mean(aps)))
    return np.mean(aps)

def evaluate(model, transform, eval_path, batch_sizes, thresh, im_size, eval_classes, dataset_source,dataset_name, set_name, ext):
    det_folder="eval/{}".format(dataset_name)
    if not os.path.exists(det_folder):
        os.makedirs(det_folder)

    model.eval()
    model.cuda()
    # the_classes = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
    # the_classes = ['smoke']
    the_classes = eval_classes
    # test_path = '/home/ecust/gmy/pytorch-retinanet-master/pytorch-retinanet-master/data/NEU-DET/valid/labels'
    print("evaluate_path={}".format(eval_path))
    evalset = ListDataset(root=eval_path, train=False, transform=transform, input_size=im_size, ext=ext)
    print("len_dataset={}".format(len(evalset)))
    testloader = torch.utils.data.DataLoader(evalset, batch_size=batch_sizes, shuffle=False, num_workers=8,
                                             collate_fn=evalset.collate_fn)
    print("len_test_loader={}".format(len(testloader)))
    encoder = DataEncoder()

    evaluation(model, batch_sizes, testloader, the_classes, encoder,thresh,im_size, im_size,det_folder)
    map=eval(the_classes,dataset_source,dataset_name,det_folder,set_name)

    return map


def test_one_weight():
    print("test_one_weight")
    batch_sizes = 1
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataset_source="composite"###########################
    dataset_source="real"

    # dataset_name="composite_18.1_gmy"####################
    dataset_name = "real_7_gmy"

    # set_name = "test"###############################
    set_name="val"

    eval_path = 'data/dataset/{}/{}/{}/label_txt'.format(dataset_source,dataset_name,set_name)  #############

    if dataset_source=="composite":
        ext = "jpg"
        eval_classes = ["gas"]
    elif dataset_source=="real":
        ext="png"
        eval_classes = ["smoke"]

    print('Loading model..')
    net = RetinaNet()
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    net.load_state_dict(torch.load('weights/weight_retinanet_composite18.1_epoch4.pth'))#############
    evaluate(net,transform,eval_path,batch_sizes,thresh=0.01,im_size = 300,eval_classes=eval_classes,dataset_source=dataset_source,dataset_name=dataset_name,set_name=set_name,ext=ext)



def test_weights_in_folder():
    print("test_weights_in_folder")
    batch_sizes = 1
    print("batch_sizes={}".format(batch_sizes))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # dataset_source="composite"###########################
    dataset_source="real"

    # dataset_name="composite_18.1_gmy"####################
    dataset_name = "real_7_gmy"

    # set_name = "test"###############################
    set_name="val"

    if dataset_source=="composite":
        ext = "jpg"
        eval_classes = ["gas"]
    elif dataset_source=="real":
        ext="png"
        eval_classes = ["smoke"]

    eval_path = 'data/dataset/{}/{}/{}/label_txt'.format(dataset_source,dataset_name,set_name)  #############


    print('Loading model..')
    net = RetinaNet()
    # net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    weights_folder="checkpoints/retinanet_gas_composite18.1"################################################################
    weight_list=os.listdir(weights_folder)
    weight_list.sort(key=lambda x:int(x[x.index("_")+1:x.index(".pth")]))
    for weight in weight_list:
        print("-"*50)
        weight_path=os.path.join(weights_folder,weight)
        print(weight_path)
        net.load_state_dict(torch.load(weight_path))
        evaluate(net,transform,eval_path,batch_sizes,thresh=0.01,im_size = 300,eval_classes=eval_classes,dataset_source=dataset_source,dataset_name=dataset_name,set_name=set_name,ext=ext)



if __name__ == '__main__':
    # test_one_weight()

    test_weights_in_folder()


