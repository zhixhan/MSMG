import seaborn as sns
import numpy as np
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
    with open("logs/vis/msmg.json", 'r') as f:
        results = json.load(f)["results"]
    for anno in results:
        correct = 0
        for j in range(10):
            correct += anno['gallery'][j]['correct']
        dir_name = './logs/vis/imgs_msmg/'+anno['query_img'].split('.')[0]+'___'+str(correct/10)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        for i in range(10):
            imgopen1 = cv2.imread('/data/hanzhixiong/SeqNet/data/PRW/frames/'+anno['query_img'])
            d = anno['query_roi']
            cv2.rectangle(imgopen1, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0,0,255),2)
            
            imgopen2 = cv2.imread('/data/hanzhixiong/SeqNet/data/PRW/frames/'+anno['gallery'][i]['img'])
            d2 = anno['gallery'][i]['roi']
            score = anno['gallery'][i]['score']
            correct = anno['gallery'][i]['correct']
            cv2.rectangle(imgopen2, (int(d2[0]), int(d2[1])), (int(d2[2]), int(d2[3])), (0,0,255),2)
            #cv2.putText(imgopen2, str(round(score,2)), (int(d2[0]), int(d2[1])-10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
            #cv2.putText(imgopen2, str(correct), (int(d2[0]), int(d2[1])+10),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),2)
            h = max(imgopen1.shape[0],imgopen2.shape[0])
            w = max(imgopen1.shape[1],imgopen2.shape[1])
            imgopen1 = cv2.resize(imgopen1, (w,h))
            imgopen2 = cv2.resize(imgopen2, (w,h))
            imgopen = np.hstack([imgopen1,imgopen2])
            
            cv2.imwrite(dir_name+'/'+str(i)+'_' +str(correct)+ '_'+anno['gallery'][i]['img'], imgopen)

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
    dy['mAP (%)'] = dy['mAP (%)'][::-1]
    dy['top-1 (%)'] = dy['top-1 (%)'][::-1]
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
    plt.legend(bbox_to_anchor=(0.99,0.215),fontsize=16)
    # 调用twinx后可绘制次坐标轴
    plt.twinx()
    plt.plot(lx, dy.get(syt), label=syt, linewidth=3, marker="o", markersize=7, color='orange')
    plt.ylabel(syt, fontsize=20)
    plt.yticks(fontsize=16)
    plt.ylim(87.5, 90)
    plt.legend(bbox_to_anchor=(0.99,0.12),fontsize=16)
    # 设置x轴刻度
    #ax = plt.gca()
    #ax.xaxis.set_major_locator(MultipleLocator(2))

    plt.savefig('kp2.jpg',dpi=600,bbox_inches='tight')

def gs1():
    fig = plt.figure(figsize=(4.5,6))
    lx = [1, 2, 3, 4, 5, 6]
    
    dy_oim = [79, 75.5, 66, 61.3, 57, 51.8]
    dy_rcaa = [83.8, 79.4, 71.2, 64.2, 61, 56.7]
    dy_ctxg = [87.2, 84.2, 78.4, 74.6, 71, 66.5]
    dy_nae = [93, 92, 87.5, 84.8, 82, 78.7]
    dy_aps = [94, 93, 89.4, 88.5, 84.6, 81.5]
    dy_dmr = [94.3, 93, 89.8, 87.7, 86.4, 83.5]
    dy_coat = [95.1, 94.2, 91.3, 89.1, 86.8, 84]
    dy = [94.5, 93.4, 90.1, 87.8, 86.7, 82.8]
    plt.plot(lx, dy_oim, label='OIM', linewidth=2, marker="o",markersize=7, color='fuchsia')
    plt.plot(lx, dy_rcaa, label='RCAA', linewidth=2, marker="<",markersize=7, color='orange')
    plt.plot(lx, dy_ctxg, label='CTXG', linewidth=2, marker="D",markersize=7, color='blue')
    plt.plot(lx, dy_nae, label='NAE+', linewidth=2, marker="P",markersize=7, color='black')
    plt.plot(lx, dy_aps, label='AlignPS', linewidth=2, marker="p",markersize=7, color='green')
    #plt.plot(lx, dy_dmr, label='DMRNet', linewidth=2, marker="o",markersize=7, color='red')
    plt.plot(lx, dy_coat, label='COAT', linewidth=2, marker="*",markersize=7, color='cyan')
    plt.plot(lx, dy, label='MSMG', linewidth=2, marker="s",markersize=7, color='red')
    plt.grid(linestyle="--", alpha=0.3)
    plt.xticks([1,2,3,4,5,6],['50', '100', '500', '1000', '2000', '4000'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Gallery size',fontsize=18)
    plt.ylabel('mAP (%)',fontsize=18)
    plt.legend(fontsize=11.6)
    plt.ylim(50, 97)
    plt.savefig('gs1.jpg',dpi=600,bbox_inches='tight')

def gs2():
    fig = plt.figure(figsize=(4.5,6))
    lx = [1, 2, 3, 4, 5, 6]
    
    dy_tcts = [94.7, 93.9, 90.6, 88.8, 87.2, 84.1]
    dy_rdlr = [94.1, 93, 89.2, 87.3, 85.1, 82.4]
    dy_clsa = [88.3, 87.1, 85.1, 84.5, 82.8, 76.8]
    dy_mgts = [84.9, 83, 77.1, 74, 71.5, 67]
    dy = [94.5, 93.4, 90.1, 87.8, 86.7, 82.8]
    plt.plot(lx, dy_tcts, label='TCTS', linewidth=2, marker="o",markersize=7, color='black')
    plt.plot(lx, dy_rdlr, label='RDLR', linewidth=2, marker="<",markersize=7, color='green')
    plt.plot(lx, dy_clsa, label='CLSA', linewidth=2, marker="D",markersize=7, color='blue')
    plt.plot(lx, dy_mgts, label='MGTS+', linewidth=2, marker="P",markersize=7, color='orange')
    plt.plot(lx, dy, label='MSMG', linewidth=2, marker="s",markersize=7, color='red')
    plt.grid(linestyle="--", alpha=0.3)
    plt.xticks([1,2,3,4,5,6],['50', '100', '500', '1000', '2000', '4000'], fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel('Gallery size',fontsize=18)
    plt.ylabel('mAP (%)',fontsize=18)
    plt.legend(fontsize=11.6)
    plt.savefig('gs2.jpg',dpi=600,bbox_inches='tight')
rank1_vis()