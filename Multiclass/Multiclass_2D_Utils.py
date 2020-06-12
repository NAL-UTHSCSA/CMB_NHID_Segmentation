
import Multiclass_2D_DataGenerator_V1
import nibabel as nib
import numpy as np
import os
from skimage.exposure import rescale_intensity

def load_pad_save_image(mask, output):
    '''
    Simply loads an Niftii image, applies a padding, and saves to file
    :param mask: Mask image file to pad and save
    :param output: Output path
    :return:
    '''
    mask_niftii = nib.load(mask)
    mask_img = mask_niftii.get_data()

    mask_img = Multiclass_2D_DataGenerator_V1.Multiclass_2D_DataGenerator_V1.pad_volume(mask_img)

    padded_mask_niftii = nib.Nifti1Image(mask_img, mask_niftii.affine, mask_niftii.header)
    # padded_mask_niftii.set_data_dtype(np.int16)
    nib.save(padded_mask_niftii, output)


def compute_iou(label_gt, label_pred):
    label_gt = label_gt.astype(np.float)
    label_pred = label_pred.astype(np.float)

    intersection = np.sum(label_gt * label_pred)
    union = np.sum(label_gt) + np.sum(label_pred) - intersection

    return (intersection / union)



def compute_dice(label_gt, label_pred):
    label_gt = label_gt.astype(np.float)
    label_pred = label_pred.astype(np.float)

    dice = 2.0 * np.sum(label_pred) / (np.sum(label_gt) + np.sum(label_pred))
    return dice



def compute_labelwise_metrics(groundtruth_file, prediction_file, use_metric = "IoU", gt_label_names = None):
    # Make sure groundtruth and prediction files exist
    assert ((os.path.exists(groundtruth_file) == True) and (os.path.exists(prediction_file) == True)), "\n\nERROR\nGroundtruth and/or prediction files cannot be found.\n    Groundtruth file: " + groundtruth_file + "\n    Prediction file: " + prediction_file + "\n\n"

    gt_niftii = nib.load(groundtruth_file)
    gt_img = gt_niftii.get_data()
    gt_img = gt_img + 1 # For simplifying computations, increase all voxels by 1


    pred_niftii = nib.load(prediction_file)
    pred_img = pred_niftii.get_data()
    pred_img = pred_img + 1 # For simplifying computations, increase all voxels by 1

    # Make sure the groundtruth and prediction images have same dimensions
    assert (gt_img.shape[0] == pred_img.shape[0]) and (gt_img.shape[1] == pred_img.shape[1]) and (gt_img.shape[2] == pred_img.shape[2]), "\n\nERROR\nGroundtruth and prediction images have different dimensions.\n    Groundtruth: (" + str(gt_img.shape[0]) + ", " + str(gt_img.shape[1]) + ", " + str(gt_img.shape[2]) + ")\n    Prediction: (" + str(pred_img.shape[0]) + ", " + str(pred_img.shape[1]) + ", " + str(pred_img.shape[2]) + ")\n\n"


    gt_labels = np.unique(gt_img)

    if (gt_label_names is None) :
        gt_label_names = np.copy(gt_labels)

    print(use_metric + " metrics for labels", gt_labels, "are: ")
    for label in gt_labels:

        label_gt = np.copy(gt_img)
        label_gt[gt_img != label] = 0
        label_gt = label_gt / label


        label_pred = np.copy(pred_img)
        label_pred[pred_img != label] = 0
        label_pred = label_pred / label

        score = -999.
        if ("dice" in use_metric) or ("Dice" in use_metric) or ("DICE" in use_metric):
            score = compute_dice(label_gt, label_pred)
        else:
            score = compute_iou(label_gt, label_pred)

        gtnif = nib.Nifti1Image(label_gt, gt_niftii.affine, gt_niftii.header)
        nib.save(gtnif, "/media/rashidt/Data/MESA_CMB/Scripts/Hippocampus_and_Ventricles_Segmentation3/Testing/3014703/Label_" + str(label) + "_GT.nii.gz")

        prednif = nib.Nifti1Image(label_pred, pred_niftii.affine, pred_niftii.header)
        nib.save(prednif, "/media/rashidt/Data/MESA_CMB/Scripts/Hippocampus_and_Ventricles_Segmentation3/Testing/3014703/Label_" + str(label) + "_PRED.nii.gz")


        print("    " + gt_label_names[label - 1] + ": " + str(score))



def skullstrip_and_rescale(brain, mask, outputfile, rescale = True):
    '''
    This will perform a skullstripping and image intensity rescaling.
    :param brain: (File path) Path of the whole brain image
    :param mask: (File path) Path of the binary brain mask
    :param outputfile: (File path) Path of the output file to save to
    :param rescale: (Bool) Whether to rescale intensity or not. If True, then output image will be rescaled to [0, 255]
    :return:
    '''
    brain_niftii = nib.load(brain)
    brain_img = brain_niftii.get_data()

    mask_niftii = nib.load(mask)
    mask_img = mask_niftii.get_data().astype(np.bool)

    skullstripped_img = np.copy(brain_img)

    # Apply skullstripping using the mask
    skullstripped_img[mask_img == False] = 0

    print("Skullstripping", end = "")

    # Rescale
    if (rescale == True):
        skullstripped_img = rescale_intensity(skullstripped_img, out_range=(0, 255))
        print(" and rescaling for: ", end = "\n")
    else:
        print(" for", end = "\n")

    # Save output
    skullstripped_niftii = nib.Nifti1Image(skullstripped_img, brain_niftii.affine, brain_niftii.header)
    nib.save(skullstripped_niftii, outputfile)

    print("  Brain image: " + brain + "\n  Mask: " + mask + "\n  Output: " + outputfile)
    print("==============================\n")



def generate_masks_with_background_and_nonbrain_labels(input_mask_file, binary_wholebrain_mask_file, output_mask_file = None, labels_to_keep = None, new_labels = None):
        '''
        :param input_mask_file: (file path). Input mask file containing the ROI labels
        :param binary_wholebrain_mask_file: (file path). Binary mask of the whole brain
        :param output_mask_file: (file path). Path of the output mask
        :param labels_to_keep: (integer list). List of labels to keep if the input_mask_file contains many labels.
                                The labels specified here will be kept, and the other labels (if any) will be
                                removed from the output mask.
        :param new_labels: If the labels specified in labels_to_keep need to be changed to another value
        :return:

        Generate a mask of specific labels fron an input labelled mask.
        This function will return a mask where the background is set to 1, and the
            0 - Background
            1 - non-ROI
            2 ... N - Labels of interest
        '''


        print("Input mask: " + input_mask_file)
        print("  Setting background = 0 and non-ROI = 1")
        print("  Labels to keep: ", end = "")
        for j in labels_to_keep:
            print(j, end = ", ")

        input_niftii = nib.load(input_mask_file)
        input_img = input_niftii.get_data().astype(np.int)

        binary_whole_brain_mask_niftii = nib.load(binary_wholebrain_mask_file)
        binary_whole_brain_mask = binary_whole_brain_mask_niftii.get_data()

        new_img = np.copy(binary_whole_brain_mask)

        # Set the labels for the ROIs with their new labels
        for i in range(len(labels_to_keep)):
            new_img[input_img == labels_to_keep[i]] = new_labels[i]

        # Generate output filename is none is specified
        if (output_mask_file == None):
            fname = os.path.basename(input_mask_file)
            fname = fname.split("_")[0]

            fname += "_Labels_bkgrnd_nonROI_"

            for j in labels_to_keep:
                fname += "_" + str(j)

            output_mask_file = os.path.join(os.path.dirname(input_mask_file), fname + ".nii.gz")



        print("\n  Output file is: " + output_mask_file)
        new_niftii = nib.Nifti1Image(new_img, input_niftii.affine, input_niftii.header)
        # new_niftii.set_data_dtype(np.int)
        nib.save(new_niftii, output_mask_file)


def change_dtype_to_uint(inputmask, outputmask, newdtype = np.uint8):
    niftii = nib.load(inputmask)

    print("Working on mask file " + inputmask)
    print("  Changing datatype from " + str(niftii.get_data_dtype()) + " to " + str(newdtype))
    img = niftii.get_data().astype(newdtype)

    new_niftii = nib.Nifti1Image(img, niftii.affine, niftii.header)
    new_niftii.set_data_dtype(np.uint8)

    nib.save(new_niftii, outputmask)
    print("  New mask file saved to " + outputmask)
    print("==========================================\n")







root_dir = "/media/rashidt/Data/MESA_CMB/Data/BiasCorrected"
for root, dirs, files in os.walk(root_dir, topdown = True):
    for dirname in dirs:
        # print(os.path.join(root, dirname))
        filelist = os.listdir(os.path.join(root, dirname))

        flair = ""
        t1 = ""
        t2 = ""
        mask = ""

        for f in filelist:
            if (("_FLAIR_N4_to_SWI_" in f) and (".nii.gz" in f)):
                flair = os.path.join(root, dirname, f)
            if (("_T1_N4_to_SWI_" in f) and (".nii.gz" in f)):
                t1 = os.path.join(root, dirname, f)
            if (("_T2_N4_to_SWI_" in f) and (".nii.gz" in f)):
                t2 = os.path.join(root, dirname, f)

        masklist = os.listdir(os.path.join("/media/rashidt/Data/MESA_CMB/Data/Skull-Stripped", dirname))
        for m in masklist:
            if (("_MUSE_BrainMask_to_SWI" in m) and (".nii.gz" in m)):
                mask = os.path.join("/media/rashidt/Data/MESA_CMB/Data/Skull-Stripped", dirname, m)


        fl_id = os.path.basename(flair).split("_")[0]
        t1_id = os.path.basename(t1).split("_")[0]
        t2_id = os.path.basename(t2).split("_")[0]
        m_id = os.path.basename(mask).split("_")[0]

        if ((m_id == fl_id) and (m_id == t1_id) and (m_id == t2_id)):
            print("Found Subject " + fl_id)
            print("  FLAIR: " + flair)
            print("  T1: " + t1)
            print("  T2: " + t2)
            print("  MUSE Brain Mask: " + mask)


        print()

# # datafile = open("/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/MUSE_SkullStripped_SWI_QSM_T2.csv", "r")
# #
# # for line in datafile:
# #     line = line.strip("\n").split(",")
# #
# #     mask = line[0]
# #     swi = line[1]
# #     swi_output = line[2]
# #     t2 = line[3]
# #     t2_output = line[4]
# #
# #     # Sanity check. Make sure all subject IDs are the same
# #     maskid = os.path.basename(mask).split("_")[0]
# #     swiid = os.path.basename(swi).split("_")[0]
# #     t2id = os.path.basename(t2).split("_")[0]
# #
# #     if ((maskid == swiid) and (swiid == t2id)):
# #         print()
# #         skullstrip_and_rescale(swi, mask, swi_output, rescale = True)
# #         skullstrip_and_rescale(t2, mask, t2_output, rescale = True)
# #     else:
# #         print("\n\nERROR\n  ID mismatch for:\n    Mask:" + mask + "\n    SWI: " + swi + "\n    T2: " + t2)
#
#
# #
# roimasks = [
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/3014460_QSM_SWI_20180807093751_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/4019113_QSM_SWI_20181003110200_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/5019575_QSM_SWI_20180523085225_13_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7011121_QSM_SWI_20181205132530_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7011679_QSM_SWI_20180621110849_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7013582_QSM_SWI_20180613102854_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7013736_QSM_SWI_20181005104359_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7014538_QSM_SWI_20181019140358_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7014988_QSM_SWI_20181005124901_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7015020_QSM_SWI_20181016100809_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7016794_QSM_SWI_20181002115454_13_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7018479_QSM_SWI_20180710104952_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7019548_QSM_SWI_20181116120831_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7042604_QSM_SWI_20181101120316_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8010617_QSM_SWI_20190131091804_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013039_QSM_SWI_20181114092510_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013179_QSM_SWI_20190124102458_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013659_QSM_SWI_20190123110727_13_e1_CMB_Segmentation_Updated.nii.gz",
#     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8015325_QSM_SWI_20181031093607_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8016046_QSM_SWI_20181121104537_12_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8017093_QSM_SWI_20181115111045_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8018898_QSM_SWI_20181105095031_11_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8022410_QSM_SWI_20180927100012_18_e1_CMB_Segmentation_Updated.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8024316_QSM_SWI_20181121094704_12_e1_CMB_Segmentation_Updated.nii.gz"
# ]
# #
# whole_brain_masks = [
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/3014460/3014460_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/4019113/4019113_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/5019575/5019575_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7011121/7011121_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7011679/7011679_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7013582/7013582_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7013736/7013736_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7014538/7014538_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7014988/7014988_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7015020/7015020_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7016794/7016794_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7018479/7018479_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7019548/7019548_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/7042604/7042604_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8010617/8010617_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8013039/8013039_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8013179/8013179_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8013659/8013659_QSM_MUSE_Mask.nii.gz",
#     "/media/rashidt/Data/MESA_CMB/Data/Skull-Stripped/8015325_20181031/8015325_MUSE_BrainMask_to_SWI_FLIRT_Corrected.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8016046/8016046_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8017093/8017093_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8018898/8018898_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8022410/8022410_QSM_MUSE_Mask.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/QSM_Processing_wMUSE/8024316/8024316_QSM_MUSE_Mask.nii.gz"
# ]
# #
# #
# outputs = [
# # 	"/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/3014460_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/4019113_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/5019575_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7011121_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7011679_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7013582_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7013736_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7014538_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7014988_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7015020_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7016794_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7018479_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7019548_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/7042604_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8010617_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013039_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013179_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8013659_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
#     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8015325_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8016046_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8017093_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8018898_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8022410_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz",
# #     "/media/rashidt/Data/MESA_CMB/Data/Clean_Rescaled_Outliered_Skullstripped_MUSE/CMB_Seg/8024316_bkgnd_OtherBrain_CMB_IronDeposits.nii.gz"
# ]
# #
# for i in range(len(outputs)):
#     labelskeep = [1, 2]
#     newlabels = [2, 3]
#     generate_masks_with_background_and_nonbrain_labels(roimasks[i], whole_brain_masks[i], outputs[i], labels_to_keep = labelskeep, new_labels = newlabels)
#







