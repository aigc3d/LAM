import glob
import json
import os.path as osp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
import pandas as pd


def L2(p1, p2):
    return np.linalg.norm(p1 - p2)


def NME(landmarks_gt, landmarks_pv):
    pts_num = landmarks_gt.shape[0]
    if pts_num == 29:
        left_index = 16
        right_index = 17
    elif pts_num == 68:
        left_index = 36
        right_index = 45
    elif pts_num == 98:
        left_index = 60
        right_index = 72

    nme = 0
    eye_span = L2(landmarks_gt[left_index], landmarks_gt[right_index])
    nmeList = []
    for i in range(pts_num):
        error = L2(landmarks_pv[i], landmarks_gt[i])
        _nme = error / eye_span
        nmeList.append(_nme)
        nme += _nme
    nme /= pts_num
    return nme, nmeList


def NME_analysis(listA):
    for jsonA in listA:
        pred = np.array(jsonA['pred'])
        gt = np.array(jsonA['gt'])
        nme, nmeList = NME(gt, pred)
        jsonA['nme'] = nme
        jsonA['nmeList'] = nmeList
    return listA


def nme_analysis(listA):
    bdy_nmeList = []
    scene_nmeList = []
    for jsonA in tqdm(listA):
        nme = jsonA['nmeList']
        nme = np.array(nme)
        bdy_nme = np.mean(nme[:33])
        scene_nme = np.mean(nme[33:])
        # scene_nme = np.mean(nme[[33, 35, 40, 38,
        #                          60, 62, 96, 66, 64,
        #                          50, 44, 48, 46,
        #                          68, 70, 97, 74, 72,
        #                          54, 55, 57, 59,
        #                          76, 82, 79, 90, 94, 85, 16]])
        bdy_nmeList.append(bdy_nme)
        scene_nmeList.append(scene_nme)
    print('bdy nme: {:.4f}'.format(np.mean(bdy_nmeList)))
    print('scene_nmeList: {:.4f}'.format(np.mean(scene_nmeList)))


def Energy_analysis(listA, easyThresh=0.02, easyNum=10, hardThresh=0.07, hardNum=10):
    easyDict = {'energy': [], 'nme': []}
    hardDict = {'energy': [], 'nme': []}

    _easyNum, _hardNum = 0, 0

    def cal_energy(evalues):
        evalues = np.array(evalues)
        # _energy = _energy.max(1)
        eccentricity = evalues.max(1) / evalues.min(1)
        # _energy = _energy.sum() / 2
        _energy = np.mean(eccentricity)
        return _energy

    for jsonA in tqdm(listA):
        nme = jsonA['nme']
        evalues = jsonA['evalues']

        if _easyNum == easyNum and _hardNum == hardNum:
            break

        if nme < easyThresh and _easyNum < easyNum:
            energy = cal_energy(evalues)
            easyDict['energy'].append(energy)
            easyDict['nme'].append(nme)
            _easyNum += 1
        elif nme > hardThresh and _hardNum < hardNum:
            energy = cal_energy(evalues)
            hardDict['energy'].append(energy)
            hardDict['nme'].append(nme)
            _hardNum += 1

    print('easyThresh: < {}; hardThresh > {}'.format(easyThresh, hardThresh))
    print('              |nme    |energy |num |')
    print('easy samples: |{:.4f} |{:.4f} |{} |'.format(np.mean(easyDict['nme']),
                                                       np.mean(easyDict['energy']),
                                                       len(easyDict['energy'])))
    print('hard samples: |{:.4f} |{:.4f} |{} |'.format(np.mean(hardDict['nme']),
                                                       np.mean(hardDict['energy']),
                                                       len(hardDict['energy'])))

    return easyDict, hardDict


def Eccentricity_analysis(listA):
    eyecornerList = []
    boundaryList = []
    for jsonA in listA:
        evalues = np.array(jsonA['evalues'])
        eccentricity = evalues.max(1) / evalues.min(1)

        eyecorner = np.mean(eccentricity[[60, 64, 68, 72]])
        boundary = np.mean(eccentricity[0:33])
        eyecornerList.append(eyecorner)
        boundaryList.append(boundary)

    print('eyecorner: {:.4f}'.format(np.mean(eyecornerList)))
    print('boundary:  {:.4f}'.format(np.mean(boundaryList)))
    return eyecornerList, boundaryList


def plot_bar(dataList):
    x = list(range(98))
    assert len(x) == len(dataList)
    _x = 'Landmark Index'
    # _y = 'elliptical eccentricity (λ1/λ2)'
    _y = 'PCA Analyze (λ1/λ2)'
    data = {
        _x: x,
        _y: dataList
    }
    df = DataFrame(data)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=_x, y=_y, data=df)
    plt.show()


def Eccentricity_analysis2(listA, is_vis=False):
    landmarksList = [[] for i in range(98)]
    for jsonA in listA:
        evalues = np.array(jsonA['evalues'])
        eccentricity = evalues.max(1) / evalues.min(1)
        for i, e in enumerate(eccentricity):
            landmarksList[i].append(e)
    print('Mean value: {:.4f}'.format(np.mean(np.array(landmarksList))))
    landmarksList = [np.mean(l) for l in landmarksList]
    if is_vis:
        plot_bar(landmarksList)
    return landmarksList


def std_analysis2():
    save_dir = '/apdcephfs/share_1134483/charlinzhou/experiment/cvpr-23/wflw_results'
    # l2_npy = glob.glob(osp.join(save_dir, '*DSNT*.npy'))
    l2_npy = glob.glob(osp.join(save_dir, '*MHNLoss_v2_l2*.npy'))

    def npy2std(npyList):
        datas = [np.load(npy)[np.newaxis, :] for npy in npyList]
        datas = np.concatenate(datas, axis=0)
        # denormalization
        datas = (datas + 1) * 256 / 2
        mean = datas.mean(axis=0)[np.newaxis, :]
        dist = np.linalg.norm(datas - mean, axis=-1)
        std = np.std(dist, 0)
        print('min: {}, max:{}, mean:{}'.format(std.min(), std.max(), std.mean()))
        return std

    std1 = npy2std(l2_npy)
    std1 = std1.mean(0)
    # plot_bar(std1)
    bdy_std = np.mean(std1[:33])
    cofw_std = np.mean(std1[[33, 35, 40, 38,
                             60, 62, 96, 66, 64,
                             50, 44, 48, 46,
                             68, 70, 97, 74, 72,
                             54, 55, 57, 59,
                             76, 82, 79, 90, 94, 85, 16]])
    print('bdy_std: {:.4f}, cofw_std: {:.4f}'.format(bdy_std, cofw_std))
    print('the ratio of Boundary std and ALL std: {:.4f} / {:.4f}'.format(np.sum(std1[:33]), np.sum(std1)))


if __name__ == '__main__':
    # 4.29模型
    json_path = '/apdcephfs/share_1134483/charlinzhou/ckpts/STAR/WFLW/WFLW_256x256_adam_ep500_lr0.001_bs128_STARLoss_smoothl1_1_b0183746-161a-4b76-9cb9-8a2059090233/results.json'
    # 无初始化
    # json_path = '/apdcephfs/share_1134483/charlinzhou/ckpts/STAR/WFLW/WFLW_256x256_adam_ep500_lr0.001_bs128_STARLoss_smoothl1_1_9cff3656-8ca8-4c3d-a95d-da76f9f76ea5/results.json'
    # 4.02模型
    # json_path = '/apdcephfs/share_1134483/charlinzhou/ckpts/STAR/WFLW/WFLW_256x256_adam_ep500_lr0.001_bs128_STARLoss_smoothl1_1_AAM_2d2bb70e-6fdb-459c-baf7-18c89e7a165f/results.json'
    listA = json.load(open(json_path, 'r'))
    print('Load Done!')
    listA = NME_analysis(listA)
    print('NME analysis Done!')
    # Exp1: 分析简单样本和困难样本的能量差异
    easyDict, hardDict = Energy_analysis(listA, easyNum=2500, hardNum=2500, easyThresh=0.03, hardThresh=0.08)

    # Exp2.1: 分析眼角点和轮廓点的斜率差异
    # eyecornerList, boundaryList = Eccentricity_analysis(listA)

    # Exp2.2: 可视化所有点的斜率分布
    # landmarksList = Eccentricity_analysis2(listA, is_vis=True)

    # Exp2.3: 可视化所有点的方差分布
    # std_analysis2()

    # Exp3: 五官和轮廓NME分析
    # nme_analysis(listA)
    # print(easyDict)
    # print(hardDict)

    # nmeList = [jsonA['nme'] for jsonA in listA]
    # print(len(nmeList))
