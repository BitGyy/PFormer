import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage
import argparse


def read_nii(path):
    itk_img = sitk.ReadImage(path)
    spacing = np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img), spacing


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3

    return rv, myo, lv


'''    
def hd(pred,gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = binary.dc(pred, gt)
        hd95 = binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0
'''


def hd(pred, gt):
    # labelPred=sitk.GetImageFromArray(lP.astype(np.float32), isVector=False)
    # labelTrue=sitk.GetImageFromArray(lT.astype(np.float32), isVector=False)
    # hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    # hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    # return hausdorffcomputer.GetAverageHausdorffDistance()
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        #         print(hd95)
        return hd95
    else:
        return 0


def asd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        ASD = binary.asd(pred, gt)
        #         print(ASD)
        return ASD
    else:
        return 0


def pre(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        pre = binary.precision(pred, gt)
        #         print(ASD)
        return pre
    else:
        return 0


def sen(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        sen = binary.sensitivity(pred, gt)
        #         print(ASD)
        return sen
    else:
        return 0


def test(fold):
    label_path = './labelsTs'
    pred_path = './92.48/'

    label_list = sorted(glob.glob(os.path.join(label_path, '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(pred_path, '*nii.gz')))

    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_rv = []
    Dice_myo = []
    Dice_lv = []

    hd_rv = []
    hd_myo = []
    hd_lv = []

    asd_rv = []
    asd_myo = []
    asd_lv = []

    pre_rv = []
    pre_myo = []
    pre_lv = []

    sen_rv = []
    sen_myo = []
    sen_lv = []

    path = './'
    file = path + 'inferTs/' + fold
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + '/dice_pre.txt', 'w')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        #         label_path = infer_path.replace('./validation_raw','./labelsTs')
        print(label_path)
        print(pred_path)
        label, spacing = read_nii(label_path)
        infer, spacing = read_nii(infer_path)
        label_rv, label_myo, label_lv = process_label(label)
        infer_rv, infer_myo, infer_lv = process_label(infer)

        Dice_rv.append(dice(infer_rv, label_rv))
        Dice_myo.append(dice(infer_myo, label_myo))
        Dice_lv.append(dice(infer_lv, label_lv))

        hd_rv.append(hd(infer_rv, label_rv))
        hd_myo.append(hd(infer_myo, label_myo))
        hd_lv.append(hd(infer_lv, label_lv))

        asd_rv.append(asd(infer_rv, label_rv))
        asd_myo.append(asd(infer_myo, label_myo))
        asd_lv.append(asd(infer_lv, label_lv))

        pre_rv.append(pre(infer_rv, label_rv))
        pre_myo.append(pre(infer_myo, label_myo))
        pre_lv.append(pre(infer_lv, label_lv))

        sen_rv.append(sen(infer_rv, label_rv))
        sen_myo.append(sen(infer_myo, label_myo))
        sen_lv.append(sen(infer_lv, label_lv))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('asd_rv: {:.4f}\n'.format(asd_rv[-1]))
        fw.write('asd_myo: {:.4f}\n'.format(asd_myo[-1]))
        fw.write('asd_lv: {:.4f}\n'.format(asd_lv[-1]))
        # fw.write('*'*20+'\n')
        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_rv: {:.4f}\n'.format(Dice_rv[-1]))
        fw.write('Dice_myo: {:.4f}\n'.format(Dice_myo[-1]))
        fw.write('Dice_lv: {:.4f}\n'.format(Dice_lv[-1]))
        fw.write('hd_rv: {:.4f}\n'.format(hd_rv[-1]))
        fw.write('hd_myo: {:.4f}\n'.format(hd_myo[-1]))
        fw.write('hd_lv: {:.4f}\n'.format(hd_lv[-1]))
        fw.write('*' * 20 + '\n')
        fw.write('pre_rv: {:.4f}\n'.format(pre_rv[-1]))
        fw.write('pre_myo: {:.4f}\n'.format(pre_myo[-1]))
        fw.write('pre_lv: {:.4f}\n'.format(pre_lv[-1]))
        fw.write('sen_rv: {:.4f}\n'.format(sen_rv[-1]))
        fw.write('sen_myo: {:.4f}\n'.format(sen_myo[-1]))
        fw.write('sen_lv: {:.4f}\n'.format(sen_lv[-1]))
        fw.write('*' * 20 + '\n')
    # fw.write('*'*20+'\n')
    # fw.write('Mean_hd\n')
    # fw.write('hd_rv'+str(np.mean(hd_rv))+'\n')
    # fw.write('hd_myo'+str(np.mean(hd_myo))+'\n')
    # fw.write('hd_lv'+str(np.mean(hd_lv))+'\n')
    # fw.write('*'*20+'\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_rv' + str(np.mean(Dice_rv)) + '\n')
    fw.write('Dice_myo' + str(np.mean(Dice_myo)) + '\n')
    fw.write('Dice_lv' + str(np.mean(Dice_lv)) + '\n')
    fw.write('Mean_HD\n')
    fw.write('HD_rv' + str(np.mean(hd_rv)) + '\n')
    fw.write('HD_myo' + str(np.mean(hd_myo)) + '\n')
    fw.write('HD_lv' + str(np.mean(hd_lv)) + '\n')
    fw.write('Mean_ASD\n')
    fw.write('ASD_rv' + str(np.mean(asd_rv)) + '\n')
    fw.write('ASD_myo' + str(np.mean(asd_myo)) + '\n')
    fw.write('ASD_lv' + str(np.mean(asd_lv)) + '\n')
    fw.write('*' * 20 + '\n')
    fw.write('Mean_Pre\n')
    fw.write('Pre_rv' + str(np.mean(pre_rv)) + '\n')
    fw.write('Pre_myo' + str(np.mean(pre_myo)) + '\n')
    fw.write('Pre_lv' + str(np.mean(pre_lv)) + '\n')
    fw.write('*' * 20 + '\n')
    fw.write('Mean_Sen\n')
    fw.write('Sen_rv' + str(np.mean(sen_rv)) + '\n')
    fw.write('Sen_myo' + str(np.mean(sen_myo)) + '\n')
    fw.write('Sen_lv' + str(np.mean(sen_lv)) + '\n')
    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    avg_hd = []
    avg_hd.append(np.mean(hd_rv))
    avg_hd.append(np.mean(hd_myo))
    avg_hd.append(np.mean(hd_lv))
    avg_asd = []
    avg_asd.append(np.mean(asd_rv))
    avg_asd.append(np.mean(asd_myo))
    avg_asd.append(np.mean(asd_lv))
    avg_pre = []
    avg_pre.append(np.mean(pre_rv))
    avg_pre.append(np.mean(pre_myo))
    avg_pre.append(np.mean(pre_lv))
    avg_sen = []
    avg_sen.append(np.mean(sen_rv))
    avg_sen.append(np.mean(sen_myo))
    avg_sen.append(np.mean(sen_lv))
    fw.write('avg_hd:' + str(np.mean(avg_hd)) + '\n')
    fw.write('DSC:' + str(np.mean(dsc)) + '\n')
    fw.write('HD:' + str(np.mean(avg_hd)) + '\n')
    fw.write('ASD:' + str(np.mean(avg_asd)) + '\n')
    fw.write('Pre:' + str(np.mean(avg_pre)) + '\n')
    fw.write('Sen:' + str(np.mean(avg_sen)) + '\n')

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold = args.fold
    test(fold)
