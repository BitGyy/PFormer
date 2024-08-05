import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
from sklearn.neighbors import KDTree
from scipy import ndimage
import argparse


def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


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


def assd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        ASD = binary.asd(pred, gt)
        #         print(ASD)
        return ASD
    else:
        return 0


def pree(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        pre = binary.precision(pred, gt)
        #         print(ASD)
        return pre
    else:
        return 0


def senn(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        sen = binary.sensitivity(pred, gt)
        #         print(ASD)
        return sen
    else:
        return 0


def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    pancreas = label == 11

    return spleen, right_kidney, left_kidney, gallbladder, liver, stomach, aorta, pancreas


def test(fold):
    #     path_img='./'
    #     label_list=sorted(glob.glob(os.path.join(path_img,'labelsTs','*nii.gz')))
    # #     infer_list=sorted(glob.glob(os.path.join(path_img,'validation_raw','*nii.gz')))
    #     infer_list=sorted(glob.glob(os.path.join(path_img,'validation_raw_postprocessed','*nii.gz')))
    #     print("loading success...")
    label_path = './labelsTs'
    pred_path = './validation_raw_postprocessed/'

    label_list = sorted(glob.glob(os.path.join(label_path, '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(pred_path, '*nii.gz')))

    print("loading success...")
    print(label_list)
    print(infer_list)
    Dice_spleen = []
    Dice_right_kidney = []
    Dice_left_kidney = []
    Dice_gallbladder = []
    Dice_liver = []
    Dice_stomach = []
    Dice_aorta = []
    Dice_pancreas = []

    hd_spleen = []
    hd_right_kidney = []
    hd_left_kidney = []
    hd_gallbladder = []
    hd_liver = []
    hd_stomach = []
    hd_aorta = []
    hd_pancreas = []

    asd_spleen = []
    asd_right_kidney = []
    asd_left_kidney = []
    asd_gallbladder = []
    asd_liver = []
    asd_stomach = []
    asd_aorta = []
    asd_pancreas = []

    pre_spleen = []
    pre_right_kidney = []
    pre_left_kidney = []
    pre_gallbladder = []
    pre_liver = []
    pre_stomach = []
    pre_aorta = []
    pre_pancreas = []

    sen_spleen = []
    sen_right_kidney = []
    sen_left_kidney = []
    sen_gallbladder = []
    sen_liver = []
    sen_stomach = []
    sen_aorta = []
    sen_pancreas = []

    file = './inferTs_lsm'
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + '/dice_pre.txt', 'a')

    for label_path, infer_path in zip(label_list, infer_list):
        print(label_path.split('/')[-1])
        print(infer_path.split('/')[-1])
        label, infer = read_nii(label_path), read_nii(infer_path)
        label_spleen, label_right_kidney, label_left_kidney, label_gallbladder, label_liver, label_stomach, label_aorta, label_pancreas = process_label(
            label)
        infer_spleen, infer_right_kidney, infer_left_kidney, infer_gallbladder, infer_liver, infer_stomach, infer_aorta, infer_pancreas = process_label(
            infer)

        Dice_spleen.append(dice(infer_spleen, label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney, label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney, label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder, label_gallbladder))
        Dice_liver.append(dice(infer_liver, label_liver))
        Dice_stomach.append(dice(infer_stomach, label_stomach))
        Dice_aorta.append(dice(infer_aorta, label_aorta))
        Dice_pancreas.append(dice(infer_pancreas, label_pancreas))

        hd_spleen.append(hd(infer_spleen, label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney, label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney, label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder, label_gallbladder))
        hd_liver.append(hd(infer_liver, label_liver))
        hd_stomach.append(hd(infer_stomach, label_stomach))
        hd_aorta.append(hd(infer_aorta, label_aorta))
        hd_pancreas.append(hd(infer_pancreas, label_pancreas))

        asd_spleen.append(assd(infer_spleen, label_spleen))
        asd_right_kidney.append(assd(infer_right_kidney, label_right_kidney))
        asd_left_kidney.append(assd(infer_left_kidney, label_left_kidney))
        asd_gallbladder.append(assd(infer_gallbladder, label_gallbladder))
        asd_liver.append(assd(infer_liver, label_liver))
        asd_stomach.append(assd(infer_stomach, label_stomach))
        asd_aorta.append(assd(infer_aorta, label_aorta))
        asd_pancreas.append(assd(infer_pancreas, label_pancreas))

        pre_spleen.append(pree(infer_spleen, label_spleen))
        pre_right_kidney.append(pree(infer_right_kidney, label_right_kidney))
        pre_left_kidney.append(pree(infer_left_kidney, label_left_kidney))
        pre_gallbladder.append(pree(infer_gallbladder, label_gallbladder))
        pre_liver.append(pree(infer_liver, label_liver))
        pre_stomach.append(pree(infer_stomach, label_stomach))
        pre_aorta.append(pree(infer_aorta, label_aorta))
        pre_pancreas.append(pree(infer_pancreas, label_pancreas))

        sen_spleen.append(senn(infer_spleen, label_spleen))
        sen_right_kidney.append(senn(infer_right_kidney, label_right_kidney))
        sen_left_kidney.append(senn(infer_left_kidney, label_left_kidney))
        sen_gallbladder.append(senn(infer_gallbladder, label_gallbladder))
        sen_liver.append(senn(infer_liver, label_liver))
        sen_stomach.append(senn(infer_stomach, label_stomach))
        sen_aorta.append(senn(infer_aorta, label_aorta))
        sen_pancreas.append(senn(infer_pancreas, label_pancreas))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))

        fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))

        fw.write('asd_spleen: {:.4f}\n'.format(asd_spleen[-1]))
        fw.write('asd_right_kidney: {:.4f}\n'.format(asd_right_kidney[-1]))
        fw.write('asd_left_kidney: {:.4f}\n'.format(asd_left_kidney[-1]))
        fw.write('asd_gallbladder: {:.4f}\n'.format(asd_gallbladder[-1]))
        fw.write('asd_liver: {:.4f}\n'.format(asd_liver[-1]))
        fw.write('asd_stomach: {:.4f}\n'.format(asd_stomach[-1]))
        fw.write('asd_aorta: {:.4f}\n'.format(asd_aorta[-1]))
        fw.write('asd_pancreas: {:.4f}\n'.format(asd_pancreas[-1]))

        fw.write('pre_spleen: {:.4f}\n'.format(pre_spleen[-1]))
        fw.write('pre_right_kidney: {:.4f}\n'.format(pre_right_kidney[-1]))
        fw.write('pre_left_kidney: {:.4f}\n'.format(pre_left_kidney[-1]))
        fw.write('pre_gallbladder: {:.4f}\n'.format(pre_gallbladder[-1]))
        fw.write('pre_liver: {:.4f}\n'.format(pre_liver[-1]))
        fw.write('pre_stomach: {:.4f}\n'.format(pre_stomach[-1]))
        fw.write('pre_aorta: {:.4f}\n'.format(pre_aorta[-1]))
        fw.write('pre_pancreas: {:.4f}\n'.format(pre_pancreas[-1]))

        fw.write('sen_spleen: {:.4f}\n'.format(sen_spleen[-1]))
        fw.write('sen_right_kidney: {:.4f}\n'.format(sen_right_kidney[-1]))
        fw.write('sen_left_kidney: {:.4f}\n'.format(sen_left_kidney[-1]))
        fw.write('sen_gallbladder: {:.4f}\n'.format(sen_gallbladder[-1]))
        fw.write('sen_liver: {:.4f}\n'.format(sen_liver[-1]))
        fw.write('sen_stomach: {:.4f}\n'.format(sen_stomach[-1]))
        fw.write('sen_aorta: {:.4f}\n'.format(sen_aorta[-1]))
        fw.write('sen_pancreas: {:.4f}\n'.format(sen_pancreas[-1]))

    #         dsc.append(Dice_spleen[-1])
    #         dsc.append((Dice_right_kidney[-1]))
    #         dsc.append(Dice_left_kidney[-1])
    #         dsc.append(np.mean(Dice_gallbladder[-1]))
    #         dsc.append(np.mean(Dice_liver[-1]))
    #         dsc.append(np.mean(Dice_stomach[-1]))
    #         dsc.append(np.mean(Dice_aorta[-1]))
    #         dsc.append(np.mean(Dice_pancreas[-1]))
    #         fw.write('DSC:'+str(np.mean(dsc))+'\n')

    #         HD.append(hd_spleen[-1])
    #         HD.append(hd_right_kidney[-1])
    #         HD.append(hd_left_kidney[-1])
    #         HD.append(hd_gallbladder[-1])
    #         HD.append(hd_liver[-1])
    #         HD.append(hd_stomach[-1])
    #         HD.append(hd_aorta[-1])
    #         HD.append(hd_pancreas[-1])
    #         fw.write('hd:'+str(np.mean(HD))+'\n')

    #         asd.append(asd_spleen[-1])
    #         asd.append(asd_right_kidney[-1])
    #         asd.append(asd_left_kidney[-1])
    #         asd.append(asd_gallbladder[-1])
    #         asd.append(asd_liver[-1])
    #         asd.append(asd_stomach[-1])
    #         asd.append(asd_aorta[-1])
    #         asd.append(asd_pancreas[-1])
    #         fw.write('asd:'+str(np.mean(asd))+'\n')

    #         pre.append(pre_spleen[-1])
    #         pre.append(pre_right_kidney[-1])
    #         pre.append(pre_left_kidney[-1])
    #         pre.append(pre_gallbladder[-1])
    #         pre.append(pre_liver[-1])
    #         pre.append(pre_stomach[-1])
    #         pre.append(pre_aorta[-1])
    #         pre.append(pre_pancreas[-1])
    #         fw.write('pre:'+str(np.mean(pre))+'\n')

    #         sen.append(sen_spleen[-1])
    #         sen.append(sen_right_kidney[-1])
    #         sen.append(sen_left_kidney[-1])
    #         sen.append(sen_gallbladder[-1])
    #         sen.append(sen_liver[-1])
    #         sen.append(sen_stomach[-1])
    #         sen.append(sen_aorta[-1])
    #         sen.append(sen_pancreas[-1])
    #         fw.write('sen:'+str(np.mean(sen))+'\n')

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_spleen' + str(np.mean(Dice_spleen)) + '\n')
    fw.write('Dice_right_kidney' + str(np.mean(Dice_right_kidney)) + '\n')
    fw.write('Dice_left_kidney' + str(np.mean(Dice_left_kidney)) + '\n')
    fw.write('Dice_gallbladder' + str(np.mean(Dice_gallbladder)) + '\n')
    fw.write('Dice_liver' + str(np.mean(Dice_liver)) + '\n')
    fw.write('Dice_stomach' + str(np.mean(Dice_stomach)) + '\n')
    fw.write('Dice_aorta' + str(np.mean(Dice_aorta)) + '\n')
    fw.write('Dice_pancreas' + str(np.mean(Dice_pancreas)) + '\n')

    fw.write('Mean_hd\n')
    fw.write('hd_spleen' + str(np.mean(hd_spleen)) + '\n')
    fw.write('hd_right_kidney' + str(np.mean(hd_right_kidney)) + '\n')
    fw.write('hd_left_kidney' + str(np.mean(hd_left_kidney)) + '\n')
    fw.write('hd_gallbladder' + str(np.mean(hd_gallbladder)) + '\n')
    fw.write('hd_liver' + str(np.mean(hd_liver)) + '\n')
    fw.write('hd_stomach' + str(np.mean(hd_stomach)) + '\n')
    fw.write('hd_aorta' + str(np.mean(hd_aorta)) + '\n')
    fw.write('hd_pancreas' + str(np.mean(hd_pancreas)) + '\n')

    fw.write('Mean_asd\n')
    fw.write('asd_spleen' + str(np.mean(asd_spleen)) + '\n')
    fw.write('asd_right_kidney' + str(np.mean(asd_right_kidney)) + '\n')
    fw.write('asd_left_kidney' + str(np.mean(asd_left_kidney)) + '\n')
    fw.write('asd_gallbladder' + str(np.mean(asd_gallbladder)) + '\n')
    fw.write('asd_liver' + str(np.mean(asd_liver)) + '\n')
    fw.write('asd_stomach' + str(np.mean(asd_stomach)) + '\n')
    fw.write('asd_aorta' + str(np.mean(asd_aorta)) + '\n')
    fw.write('asd_pancreas' + str(np.mean(asd_pancreas)) + '\n')

    fw.write('Mean_pre\n')
    fw.write('pre_spleen' + str(np.mean(pre_spleen)) + '\n')
    fw.write('pre_right_kidney' + str(np.mean(pre_right_kidney)) + '\n')
    fw.write('pre_left_kidney' + str(np.mean(pre_left_kidney)) + '\n')
    fw.write('pre_gallbladder' + str(np.mean(pre_gallbladder)) + '\n')
    fw.write('pre_liver' + str(np.mean(pre_liver)) + '\n')
    fw.write('pre_stomach' + str(np.mean(pre_stomach)) + '\n')
    fw.write('pre_aorta' + str(np.mean(pre_aorta)) + '\n')
    fw.write('pre_pancreas' + str(np.mean(pre_pancreas)) + '\n')

    fw.write('Mean_sen\n')
    fw.write('sen_spleen' + str(np.mean(sen_spleen)) + '\n')
    fw.write('sen_right_kidney' + str(np.mean(sen_right_kidney)) + '\n')
    fw.write('sen_left_kidney' + str(np.mean(sen_left_kidney)) + '\n')
    fw.write('sen_gallbladder' + str(np.mean(sen_gallbladder)) + '\n')
    fw.write('sen_liver' + str(np.mean(sen_liver)) + '\n')
    fw.write('sen_stomach' + str(np.mean(sen_stomach)) + '\n')
    fw.write('sen_aorta' + str(np.mean(sen_aorta)) + '\n')
    fw.write('sen_pancreas' + str(np.mean(sen_pancreas)) + '\n')

    fw.write('*' * 20 + '\n')

    dsc = []
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_pancreas))
    fw.write('dsc:' + str(np.mean(dsc)) + '\n')

    HD = []
    HD.append(np.mean(hd_spleen))
    HD.append(np.mean(hd_right_kidney))
    HD.append(np.mean(hd_left_kidney))
    HD.append(np.mean(hd_gallbladder))
    HD.append(np.mean(hd_liver))
    HD.append(np.mean(hd_stomach))
    HD.append(np.mean(hd_aorta))
    HD.append(np.mean(hd_pancreas))
    fw.write('hd:' + str(np.mean(HD)) + '\n')

    asd = []
    asd.append(np.mean(asd_spleen))
    asd.append(np.mean(asd_right_kidney))
    asd.append(np.mean(asd_left_kidney))
    asd.append(np.mean(asd_gallbladder))
    asd.append(np.mean(asd_liver))
    asd.append(np.mean(asd_stomach))
    asd.append(np.mean(asd_aorta))
    asd.append(np.mean(asd_pancreas))
    fw.write('asd:' + str(np.mean(asd)) + '\n')

    pre = []
    pre.append(np.mean(pre_spleen))
    pre.append(np.mean(pre_right_kidney))
    pre.append(np.mean(pre_left_kidney))
    pre.append(np.mean(pre_gallbladder))
    pre.append(np.mean(pre_liver))
    pre.append(np.mean(pre_stomach))
    pre.append(np.mean(pre_aorta))
    pre.append(np.mean(pre_pancreas))
    fw.write('hd:' + str(np.mean(pre)) + '\n')

    sen = []
    sen.append(np.mean(sen_spleen))
    sen.append(np.mean(sen_right_kidney))
    sen.append(np.mean(sen_left_kidney))
    sen.append(np.mean(sen_gallbladder))
    sen.append(np.mean(sen_liver))
    sen.append(np.mean(sen_stomach))
    sen.append(np.mean(sen_aorta))
    sen.append(np.mean(sen_pancreas))
    fw.write('hd:' + str(np.mean(sen)) + '\n')

    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("fold", help="fold name")
    args = parser.parse_args()
    fold = args.fold
    test(fold)
