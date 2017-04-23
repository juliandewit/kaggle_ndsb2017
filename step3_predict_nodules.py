import settings
import helpers
import sys
import os
import glob
import random
import pandas
import ntpath
import cv2
import numpy
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, Convolution3D, MaxPooling3D, UpSampling3D, LeakyReLU, BatchNormalization, Flatten, Dense, Dropout, ZeroPadding3D, AveragePooling3D, Activation
from keras.models import Model, load_model, model_from_json
from keras.metrics import binary_accuracy, binary_crossentropy, mean_squared_error, mean_absolute_error
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback, LearningRateScheduler
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import math


# limit memory usage..
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import step2_train_nodule_detector
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

# zonder aug, 10:1 99 train, 97 test, 0.27 cross entropy, before commit 573
# 3 pools istead of 4 gives (bigger end layer) gives much worse validation accuray + logloss .. strange ?
# 32 x 32 x 32 lijkt het beter te doen dan 48 x 48 x 48..

K.set_image_dim_ordering("tf")
CUBE_SIZE = step2_train_nodule_detector.CUBE_SIZE
MEAN_PIXEL_VALUE = settings.MEAN_PIXEL_VALUE_NODULE
NEGS_PER_POS = 20
P_TH = 0.6

PREDICT_STEP = 12
USE_DROPOUT = False


def prepare_image_for_net3D(img):
    img = img.astype(numpy.float32)
    img -= MEAN_PIXEL_VALUE
    img /= 255.
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2], 1)
    return img


def filter_patient_nodules_predictions(df_nodule_predictions: pandas.DataFrame, patient_id, view_size, luna16=False):
    src_dir = settings.LUNA_16_TRAIN_DIR2D2 if luna16 else settings.NDSB3_EXTRACTED_IMAGE_DIR
    patient_mask = helpers.load_patient_images(patient_id, src_dir, "*_m.png")
    delete_indices = []
    for index, row in df_nodule_predictions.iterrows():
        z_perc = row["coord_z"]
        y_perc = row["coord_y"]
        center_x = int(round(row["coord_x"] * patient_mask.shape[2]))
        center_y = int(round(y_perc * patient_mask.shape[1]))
        center_z = int(round(z_perc * patient_mask.shape[0]))

        mal_score = row["diameter_mm"]
        start_y = center_y - view_size / 2
        start_x = center_x - view_size / 2
        nodule_in_mask = False
        for z_index in [-1, 0, 1]:
            img = patient_mask[z_index + center_z]
            start_x = int(start_x)
            start_y = int(start_y)
            view_size = int(view_size)
            img_roi = img[start_y:start_y+view_size, start_x:start_x + view_size]
            if img_roi.sum() > 255:  # more than 1 pixel of mask.
                nodule_in_mask = True

        if not nodule_in_mask:
            print("Nodule not in mask: ", (center_x, center_y, center_z))
            if mal_score > 0:
                mal_score *= -1
            df_nodule_predictions.loc[index, "diameter_mm"] = mal_score
        else:
            if center_z < 30:
                print("Z < 30: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)
                if mal_score > 0:
                    mal_score *= -1
                df_nodule_predictions.loc[index, "diameter_mm"] = mal_score


            if (z_perc > 0.75 or z_perc < 0.25) and y_perc > 0.85:
                print("SUSPICIOUS FALSEPOSITIVE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

            if center_z < 50 and y_perc < 0.30:
                print("SUSPICIOUS FALSEPOSITIVE OUT OF RANGE: ", patient_id, " center z:", center_z, " y_perc: ",  y_perc)

    df_nodule_predictions.drop(df_nodule_predictions.index[delete_indices], inplace=True)
    return df_nodule_predictions


def filter_nodule_predictions(only_patient_id=None):
    src_dir = settings.NDSB3_NODULE_DETECTION_DIR
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        print(csv_index, ": ", patient_id)
        if only_patient_id is not None and patient_id != only_patient_id:
            continue
        df_nodule_predictions = pandas.read_csv(csv_path)
        filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE)
        df_nodule_predictions.to_csv(csv_path, index=False)


def make_negative_train_data_based_on_predicted_luna_nodules():
    src_dir = settings.LUNA_NODULE_DETECTION_DIR
    pos_labels_dir = settings.LUNA_NODULE_LABELS_DIR
    keep_dist = CUBE_SIZE + CUBE_SIZE / 2
    total_false_pos = 0
    for csv_index, csv_path in enumerate(glob.glob(src_dir + "*.csv")):
        file_name = ntpath.basename(csv_path)
        patient_id = file_name.replace(".csv", "")
        # if not "273525289046256012743471155680" in patient_id:
        #     continue
        df_nodule_predictions = pandas.read_csv(csv_path)
        pos_annos_manual = None
        manual_path = settings.MANUAL_ANNOTATIONS_LABELS_DIR + patient_id + ".csv"
        if os.path.exists(manual_path):
            pos_annos_manual = pandas.read_csv(manual_path)

        filter_patient_nodules_predictions(df_nodule_predictions, patient_id, CUBE_SIZE, luna16=True)
        pos_labels = pandas.read_csv(pos_labels_dir + patient_id + "_annos_pos_lidc.csv")
        print(csv_index, ": ", patient_id, ", pos", len(pos_labels))
        patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA_16_TRAIN_DIR2D2, "*_m.png")
        for nod_pred_index, nod_pred_row in df_nodule_predictions.iterrows():
            if nod_pred_row["diameter_mm"] < 0:
                continue
            nx, ny, nz = helpers.percentage_to_pixels(nod_pred_row["coord_x"], nod_pred_row["coord_y"], nod_pred_row["coord_z"], patient_imgs)
            diam_mm = nod_pred_row["diameter_mm"]
            for label_index, label_row in pos_labels.iterrows():
                px, py, pz = helpers.percentage_to_pixels(label_row["coord_x"], label_row["coord_y"], label_row["coord_z"], patient_imgs)
                dist = math.sqrt(math.pow(nx - px, 2) + math.pow(ny - py, 2) + math.pow(nz- pz, 2))
                if dist < keep_dist:
                    if diam_mm >= 0:
                        diam_mm *= -1
                    df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                    break

            if pos_annos_manual is not None:
                for index, label_row in pos_annos_manual.iterrows():
                    px, py, pz = helpers.percentage_to_pixels(label_row["x"], label_row["y"], label_row["z"], patient_imgs)
                    diameter = label_row["d"] * patient_imgs[0].shape[1]
                    # print((pos_coord_x, pos_coord_y, pos_coord_z))
                    # print(center_float_rescaled)
                    dist = math.sqrt(math.pow(px - nx, 2) + math.pow(py - ny, 2) + math.pow(pz - nz, 2))
                    if dist < (diameter + 72):  #  make sure we have a big margin
                        if diam_mm >= 0:
                            diam_mm *= -1
                        df_nodule_predictions.loc[nod_pred_index, "diameter_mm"] = diam_mm
                        print("#Too close",  (nx, ny, nz))
                        break

        df_nodule_predictions.to_csv(csv_path, index=False)
        df_nodule_predictions = df_nodule_predictions[df_nodule_predictions["diameter_mm"] >= 0]
        df_nodule_predictions.to_csv(pos_labels_dir + patient_id + "_candidates_falsepos.csv", index=False)
        total_false_pos += len(df_nodule_predictions)
    print("Total false pos:", total_false_pos)


def predict_cubes(model_path, continue_job, only_patient_id=None, luna16=False, magnification=1, flip=False, train_data=True, holdout_no=-1, ext_name="", fold_count=2):
    if luna16:
        dst_dir = settings.LUNA_NODULE_DETECTION_DIR
    else:
        dst_dir = settings.NDSB3_NODULE_DETECTION_DIR
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    holdout_ext = ""
    # if holdout_no is not None:
    #     holdout_ext = "_h" + str(holdout_no) if holdout_no >= 0 else ""
    flip_ext = ""
    if flip:
        flip_ext = "_flip"

    dst_dir += "predictions" + str(int(magnification * 10)) + holdout_ext + flip_ext + "_" + ext_name + "/"
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    sw = helpers.Stopwatch.start_new()
    model = step2_train_nodule_detector.get_net(input_shape=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE, 1), load_weight_path=model_path)
    if not luna16:
        if train_data:
            labels_df = pandas.read_csv("resources/stage1_labels.csv")
            labels_df.set_index(["id"], inplace=True)
        else:
            labels_df = pandas.read_csv("resources/stage2_sample_submission.csv")
            labels_df.set_index(["id"], inplace=True)

    patient_ids = []
    for file_name in os.listdir(settings.NDSB3_EXTRACTED_IMAGE_DIR):
        if not os.path.isdir(settings.NDSB3_EXTRACTED_IMAGE_DIR + file_name):
            continue
        patient_ids.append(file_name)

    all_predictions_csv = []
    for patient_index, patient_id in enumerate(reversed(patient_ids)):
        if not luna16:
            if patient_id not in labels_df.index:
                continue
        if "metadata" in patient_id:
            continue
        if only_patient_id is not None and only_patient_id != patient_id:
            continue

        if holdout_no is not None and train_data:
            patient_fold = helpers.get_patient_fold(patient_id)
            patient_fold %= fold_count
            if patient_fold != holdout_no:
                continue

        print(patient_index, ": ", patient_id)
        csv_target_path = dst_dir + patient_id + ".csv"
        if continue_job and only_patient_id is None:
            if os.path.exists(csv_target_path):
                continue

        patient_img = helpers.load_patient_images(patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_i.png", [])
        if magnification != 1:
            patient_img = helpers.rescale_patient_images(patient_img, (1, 1, 1), magnification)

        patient_mask = helpers.load_patient_images(patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*_m.png", [])
        if magnification != 1:
            patient_mask = helpers.rescale_patient_images(patient_mask, (1, 1, 1), magnification, is_mask_image=True)

            # patient_img = patient_img[:, ::-1, :]
            # patient_mask = patient_mask[:, ::-1, :]

        step = PREDICT_STEP
        CROP_SIZE = CUBE_SIZE
        # CROP_SIZE = 48

        predict_volume_shape_list = [0, 0, 0]
        for dim in range(3):
            dim_indent = 0
            while dim_indent + CROP_SIZE < patient_img.shape[dim]:
                predict_volume_shape_list[dim] += 1
                dim_indent += step

        predict_volume_shape = (predict_volume_shape_list[0], predict_volume_shape_list[1], predict_volume_shape_list[2])
        predict_volume = numpy.zeros(shape=predict_volume_shape, dtype=float)
        print("Predict volume shape: ", predict_volume.shape)
        done_count = 0
        skipped_count = 0
        batch_size = 128
        batch_list = []
        batch_list_coords = []
        patient_predictions_csv = []
        cube_img = None
        annotation_index = 0

        for z in range(0, predict_volume_shape[0]):
            for y in range(0, predict_volume_shape[1]):
                for x in range(0, predict_volume_shape[2]):
                    #if cube_img is None:
                    cube_img = patient_img[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]
                    cube_mask = patient_mask[z * step:z * step+CROP_SIZE, y * step:y * step + CROP_SIZE, x * step:x * step+CROP_SIZE]

                    if cube_mask.sum() < 2000:
                        skipped_count += 1
                    else:
                        if flip:
                            cube_img = cube_img[:, :, ::-1]

                        if CROP_SIZE != CUBE_SIZE:
                            cube_img = helpers.rescale_patient_images2(cube_img, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))
                            # helpers.save_cube_img("c:/tmp/cube.png", cube_img, 8, 4)
                            # cube_mask = helpers.rescale_patient_images2(cube_mask, (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE))

                        img_prep = prepare_image_for_net3D(cube_img)
                        batch_list.append(img_prep)
                        batch_list_coords.append((z, y, x))
                        if len(batch_list) % batch_size == 0:
                            batch_data = numpy.vstack(batch_list)
                            p = model.predict(batch_data, batch_size=batch_size)
                            for i in range(len(p[0])):
                                p_z = batch_list_coords[i][0]
                                p_y = batch_list_coords[i][1]
                                p_x = batch_list_coords[i][2]
                                nodule_chance = p[0][i][0]
                                predict_volume[p_z, p_y, p_x] = nodule_chance
                                if nodule_chance > P_TH:
                                    p_z = p_z * step + CROP_SIZE / 2
                                    p_y = p_y * step + CROP_SIZE / 2
                                    p_x = p_x * step + CROP_SIZE / 2

                                    p_z_perc = round(p_z / patient_img.shape[0], 4)
                                    p_y_perc = round(p_y / patient_img.shape[1], 4)
                                    p_x_perc = round(p_x / patient_img.shape[2], 4)
                                    diameter_mm = round(p[1][i][0], 4)
                                    # diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(2 * step / patient_img.shape[2], 4)
                                    diameter_perc = round(diameter_mm / patient_img.shape[2], 4)
                                    nodule_chance = round(nodule_chance, 4)
                                    patient_predictions_csv_line = [annotation_index, p_x_perc, p_y_perc, p_z_perc, diameter_perc, nodule_chance, diameter_mm]
                                    patient_predictions_csv.append(patient_predictions_csv_line)
                                    all_predictions_csv.append([patient_id] + patient_predictions_csv_line)
                                    annotation_index += 1

                            batch_list = []
                            batch_list_coords = []
                    done_count += 1
                    if done_count % 10000 == 0:
                        print("Done: ", done_count, " skipped:", skipped_count)

        df = pandas.DataFrame(patient_predictions_csv, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
        filter_patient_nodules_predictions(df, patient_id, CROP_SIZE * magnification)
        df.to_csv(csv_target_path, index=False)

        # cols = ["anno_index", "nodule_chance", "diamete_mm"] + ["f" + str(i) for i in range(64)]
        # df_features = pandas.DataFrame(patient_features_csv, columns=cols)
        # for index, row in df.iterrows():
        #     if row["diameter_mm"] < 0:
        #         print("Dropping")
        #         anno_index = row["anno_index"]
        #         df_features.drop(df_features[df_features["anno_index"] == anno_index].index, inplace=True)
        #
        # df_features.to_csv(csv_target_path_features, index=False)

        # df = pandas.DataFrame(all_predictions_csv, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "nodule_chance", "diameter_mm"])
        # df.to_csv("c:/tmp/tmp2.csv", index=False)

        print(predict_volume.mean())
        print("Done in : ", sw.get_elapsed_seconds(), " seconds")


if __name__ == "__main__":

    CONTINUE_JOB = True
    only_patient_id = None  # "ebd601d40a18634b100c92e7db39f585"

    if not CONTINUE_JOB or only_patient_id is not None:
        for file_path in glob.glob("c:/tmp/*.*"):
            if not os.path.isdir(file_path):
                remove_file = True
                if only_patient_id is not None:
                    if only_patient_id not in file_path:
                        remove_file = False
                        remove_file = False

                if remove_file:
                    os.remove(file_path)

    if True:
        for magnification in [1, 1.5, 2]:  #
            predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=True, holdout_no=None, ext_name="luna16_fs")
            predict_cubes("models/model_luna16_full__fs_best.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=False, holdout_no=None, ext_name="luna16_fs")

    if True:
        for version in [2, 1]:
            for holdout in [0, 1]:
                for magnification in [1, 1.5, 2]:  #
                    predict_cubes("models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=True, holdout_no=holdout, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)
                    if holdout == 0:
                        predict_cubes("models/model_luna_posnegndsb_v" + str(version) + "__fs_h" + str(holdout) + "_end.hd5", CONTINUE_JOB, only_patient_id=only_patient_id, magnification=magnification, flip=False, train_data=False, holdout_no=holdout, ext_name="luna_posnegndsb_v" + str(version), fold_count=2)

