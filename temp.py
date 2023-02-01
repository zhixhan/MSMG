import seaborn as sns
import numpy
from matplotlib import pyplot as plt
import torch
import pickle
import os
import seaborn
import pandas as pd
from matplotlib.pyplot import MultipleLocator
def rank1_vis():
    import json
    import cv2
    import os
    with open("vis/draw_dad1.json", 'r') as f:
        results = json.load(f)["results"]
    for anno in results:
        correct = 0
        for j in range(10):
            correct += anno['gallery'][j]['correct']
        dir_name = './logs/vis_dad1/'+anno['query_img'].split('.')[0]+'___'+str(correct/10)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for i in range(10):
            imgopen1 = cv2.imread('/data/hanzhixiong/SeqNet/data/PRW/frames/'+anno['query_img'])
            d = anno['query_roi']
            cv2.rectangle(imgopen1, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0,255,0),2)
            
            imgopen2 = cv2.imread('/data/hanzhixiong/SeqNet/data/PRW/frames/'+anno['gallery'][i]['img'])
            d2 = anno['gallery'][i]['roi']
            score = anno['gallery'][i]['score']
            correct = anno['gallery'][i]['correct']
            cv2.rectangle(imgopen2, (int(d2[0]), int(d2[1])), (int(d2[2]), int(d2[3])), (255,0,0),2)
            cv2.putText(imgopen2, str(round(score,2)), (int(d2[0]), int(d2[1])-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
            cv2.putText(imgopen2, str(correct), (int(d2[0]), int(d2[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
            h = max(imgopen1.shape[0],imgopen2.shape[0])
            w = max(imgopen1.shape[1],imgopen2.shape[1])
            imgopen1 = cv2.resize(imgopen1, (w,h))
            imgopen2 = cv2.resize(imgopen2, (w,h))
            imgopen = np.hstack([imgopen1,imgopen2])
            
            cv2.imwrite(dir_name+'/'+str(i)+'___'+anno['gallery'][i]['img'], imgopen)

def promote_rank1():
    base = os.walk('logs/vis_partfeat/base_imgs/')
    for root, dirs, files in os.walk('logs/vis_partfeat/base_imgs/', topdown=False):
        print(root, dirs, files)
        assert False

def list_dad():
    dirs = os.listdir('./logs/vis_dad3/')
    dirs2 = os.listdir('./logs/vis_dad1/')
    d3 = {}
    d1 = {}
    for _dir in dirs:
        _dir = _dir.split('___')
        d3[_dir[0]] = float(_dir[1])
    for _dir in dirs2:
        _dir = _dir.split('___')
        d1[_dir[0]] = float(_dir[1])
    for k in d3:
        if k in d1 and d3[k]>d1[k]:
            print(k, ' ', d3[k], ' ', d1[k])

def kp():
    value = [49.2, 52.8, 54.7, 54.9, 56.2, 56.3, 87.3, 87.8, 88.8, 88.0, 89.1, 88.7]
    k = [1, 2, 4, 5, 7, 14, 1, 2, 4, 5, 7, 14]
    metric = ['mAP']*6 + ['top-1']*6
    d={'performance': value, 'K': k, 'metric':metric}
    df=pd.DataFrame.from_dict(d)
    sns.lineplot(data=df, x="K", y="performance", hue="metric")
    plt.savefig('kp.jpg',dpi=600,bbox_inches='tight')

def draw2():
    fig = plt.figure(figsize=(4.1,6))
    lx = [1, 2, 3, 4, 5, 6]
    dy = {'mAP (%)':[49.2, 52.8, 54.9, 54.9, 56.2, 56.3], 'top-1 (%)':[87.3, 87.8, 88.4, 88.0, 89.1, 88.7]}
    fyt = list(dy.keys())[0]
    syt = list(dy.keys())[1]
    plt.plot(lx, dy.get(fyt), label=fyt, linewidth=3,marker="o",markersize=7, color='#4e92e4')
    plt.grid(linestyle="--", alpha=0.3)
    #plt.title(title, fontsize=12)
    plt.xticks([1,2,3,4,5,6],['1', '2', '4', '5', '7', '14'],fontsize=16)
    plt.ylabel(fyt, fontsize=20)
    plt.yticks(fontsize=16)
    plt.xlabel('K',fontsize=20)
    plt.ylim(49, 57)
    plt.legend(bbox_to_anchor=(0.99,0.215),fontsize=16)
    # 调用twinx后可绘制次坐标轴
    plt.twinx()
    plt.plot(lx, dy.get(syt), label=syt, linewidth=3, marker="o", markersize=7, color='orange')
    plt.ylabel(syt, fontsize=20)
    plt.ylim(87, 91)
    plt.yticks(fontsize=16)
    plt.legend(bbox_to_anchor=(0.99,0.12),fontsize=16)
    # 设置x轴刻度
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(MultipleLocator(2))

    plt.savefig('kp.jpg',dpi=600,bbox_inches='tight')

def draw1():
    fig = plt.figure(figsize=(4.1,6))
    lx = [1, 2, 3, 4, 5]
    dy = {'mAP (%)':[55.8,56.3, 56.2, 54.9, 52.8], 'top-1 (%)':[88.7, 88.8, 89.1, 88.1, 87.8]}
    fyt = list(dy.keys())[0]
    syt = list(dy.keys())[1]
    plt.plot(lx, dy.get(fyt), label=fyt, linewidth=3,marker="o",markersize=7, color='#4e92e4')
    plt.grid(linestyle="--", alpha=0.3)
    #plt.title(title, fontsize=12)
    plt.xticks([1,2,3,4,5],['0', '0.25', '0.5', '0.75', '1.0'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel(fyt, fontsize=20)
    plt.xlabel('P',fontsize=20)
    plt.ylim(52, 57)
    plt.legend(bbox_to_anchor=(0.643,0.215),fontsize=16)
    # 调用twinx后可绘制次坐标轴
    plt.twinx()
    plt.plot(lx, dy.get(syt), label=syt, linewidth=3, marker="o", markersize=7, color='orange')
    plt.ylabel(syt, fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylim(87.5, 90)
    plt.legend(bbox_to_anchor=(0.67,0.12),fontsize=16)
    # 设置x轴刻度
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(MultipleLocator(2))

    plt.savefig('kp2.jpg',dpi=600,bbox_inches='tight')
draw2()
draw1()

