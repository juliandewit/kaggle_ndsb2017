import os
COMPUTER_NAME = os.environ['COMPUTERNAME']
print("Computer: ", COMPUTER_NAME)

TARGET_VOXEL_MM = 1.00
MEAN_PIXEL_VALUE_NODULE = 41
LUNA_SUBSET_START_INDEX = 1
SEGMENTER_IMG_SIZE = 320

BASE_DIR_SSD = "C:/werkdata/kaggle/ndsb3/"
BASE_DIR = "D:/werkdata/kaggle/ndsb3/"
EXTRA_DATA_DIR = "resources/"
NDSB3_RAW_SRC_DIR = BASE_DIR + "ndsb_raw/stage12/"
LUNA16_RAW_SRC_DIR = BASE_DIR + "luna_raw/"

NDSB3_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "ndsb3_extracted_images/"
LUNA16_EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "luna16_extracted_images/"
NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"

# LUNA16_ANNOTATION_DIR = BASE_DIR_SSD + "luna16_extracted_images/"


# if COMPUTER_NAME == "BULDRIUM8":
#     BASE_DIR = "G:/werkdata/kaggle/ndsb3/"
#     BASE_DIR_SSD = "F:/werkdata/kaggle/ndsb3/"
#     EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "extracted_images_ndsb3/"
#     # LUNA_16_TRAIN_DIR2D2 = EXTRACTED_IMAGE_DIR
#     LABELS_CSV_PATH = BASE_DIR + "stage1_labels.csv"
#     SUBMISSION_CSV_PATH = BASE_DIR + "stage1_sample_submission.csv"
#     CANDIDATE_METADATA_PATH = BASE_DIR + "patient_metadata.csv"
#     NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"
#     NDSB3_NODULE_TRAIN_DIR = BASE_DIR_SSD + "ndsb3_train_nodules/"
#     HOLISTIC_PREDICTIONS = BASE_DIR_SSD + "holistic_predictions/"
#     NDSB3_MANUAL_ANNOTATIONS_PRECISION_DIR = "G:/dropbox/Dropbox/werk\kaggle/ndsb3/manual_labels_precision/"
# elif COMPUTER_NAME == "TMFMM3":
#     BASE_DIR = "D:/werkdata/kaggle/ndsb3/"
#     BASE_DIR_SSD = "C:/werkdata/kaggle/ndsb3/"
#     EXTRACTED_IMAGE_DIR = BASE_DIR_SSD + "extracted_images/"
#     LUNA_16_TRAIN_DIR2D2 = BASE_DIR_SSD + "luna16_train2d/"
#     LUNA_NODULE_LABELS_DIR = LUNA_16_TRAIN_DIR2D2 + "metadata/"
#     LUNA_NODULE_DETECTION_DIR = BASE_DIR_SSD + "luna_detected_nodules/"
#     NDSB3_MANUAL_ANNOTATIONS_PRECISION_DIR = "D:/dropbox/Dropbox/werk/kaggle/ndsb3/manual_labels_precision/"
#     MANUAL_MASSES_DIR = "D:/Dropbox/Dropbox/werk/kaggle/ndsb3/masses/"
#     MANUAL_EMPHYSEMA_DIR = "D:/Dropbox/Dropbox/werk/kaggle/ndsb3/emphysema/"
#     SEGMENTER_TRAIN_DIR = BASE_DIR_SSD + "train_segmenter/"
#     NDSB3_NODULE_DETECTION_DIR = BASE_DIR_SSD + "ndsb3_nodule_predictions/"
#     LUNA_MANUAL_ANNOTATIONS_DIR = "D:/Dropbox/Dropbox/werk/kaggle/ndsb3/manual_labels_luna16/"
#     LUNA_MANUAL_ANNOTATIONS_DIR = "D:/dropbox/Dropbox/werk/kaggle/ndsb3/manual_labels_luna16/"
#
# else:
#     BASE_DIR = "D:/werkdata/kaggle/ndsb3/"
#     BASE_DIR_SSD = "C:/werkdata/kaggle/ndsb3/"
#     EXTRACTED_IMAGE_DIR = BASE_DIR + "extracted_images/"
#     LUNA_16_TRAIN_DIR2D2 = "C:/werkdata/kaggle/ndsb3/luna16_train2d/"
#     LUNA_NODULE_LABELS_DIR = LUNA_16_TRAIN_DIR2D2 + "metadata/"
#     LUNA_NODULE_DETECTION_DIR = BASE_DIR_SSD + "luna_detected_nodules/"
#     NDSB3_MANUAL_ANNOTATIONS_DIR = "D:/dropbox/Dropbox/werk/kaggle/ndsb3\manual_labels/"
#     LUNA_MANUAL_ANNOTATIONS_DIR = "D:/dropbox/Dropbox/werk/kaggle/ndsb3/manual_labels_luna16/"
#
# RAW_SRC_DIR = BASE_DIR + "stage1/stage1/"
# LUNA_16_TRAIN_DIR = "C:/werkdata/kaggle/ndsb3/luna16_train/"
# LUNA_16_TRAIN_DIR2D = "C:/werkdata/kaggle/ndsb3/luna16_train3d/"
# LUNA_16_ANNOTATION_DIR = BASE_DIR + "CSVFILES/"
# LUNA_16_ANNOTATION_DIR_LABELER = LUNA_16_ANNOTATION_DIR + "labeler_annotations/"
# LABELS_CSV_PATH = BASE_DIR + "stage1_labels.csv"
# SUBMISSION_CSV_PATH = BASE_DIR + "stage1_sample_submission.csv"
#
# LUNA16_SEG_TRAIN_WIDTH = 128
# LUNA16_SEG_TRAIN_HEIGHT = 128
# LUNA16_SEG_TRAIN_DEPTH = 64
#
# OVERLAY_MULTIPLIER = 2
# TARGET_VOXEL_MM = 1.00
# MEAN_PIXEL_VALUE_NODULE = 41
#
