
import glob
import os

def empty_txt():
    dir = '/home/ecust/txx/project/gmy_2080_copy/pytorch-retinanet-master/pytorch-retinanet-master/detection_result/composite_gas_1_gmy_500_400'  # xml目录

    a=sorted(glob.glob("%s/*.*" % dir))
    # print(a)
    for i in a:
        f = open(i, "r+")
        f.truncate()
