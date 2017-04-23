import settings
import helpers
import glob
import pandas
import ntpath
import numpy
import cv2
import os

CUBE_IMGTYPE_SRC = "_i"


def save_cube_img(target_path, cube_img, rows, cols):
    assert rows * cols == cube_img.shape[0]
    img_height = cube_img.shape[1]
    img_width = cube_img.shape[1]
    res_img = numpy.zeros((rows * img_height, cols * img_width), dtype=numpy.uint8)

    for row in range(rows):
        for col in range(cols):
            target_y = row * img_height
            target_x = col * img_width
            res_img[target_y:target_y + img_height, target_x:target_x + img_width] = cube_img[row * cols + col]

    cv2.imwrite(target_path, res_img)


def get_cube_from_img(img3d, center_x, center_y, center_z, block_size):
    start_x = max(center_x - block_size / 2, 0)
    if start_x + block_size > img3d.shape[2]:
        start_x = img3d.shape[2] - block_size

    start_y = max(center_y - block_size / 2, 0)
    start_z = max(center_z - block_size / 2, 0)
    if start_z + block_size > img3d.shape[0]:
        start_z = img3d.shape[0] - block_size
    start_z = int(start_z)
    start_y = int(start_y)
    start_x = int(start_x)
    res = img3d[start_z:start_z + block_size, start_y:start_y + block_size, start_x:start_x + block_size]
    return res


def make_pos_annotation_images():
    src_dir = settings.LUNA_16_TRAIN_DIR2D2 + "metadata/"
    dst_dir = settings.BASE_DIR_SSD + "luna16_train_cubes_pos/"
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annos_pos.csv", "")
        # print(patient_id)
        # if not "148229375703208214308676934766" in patient_id:
        #     continue
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = helpers.load_patient_images(patient_id, settings.LUNA_16_TRAIN_DIR2D2, "*" + CUBE_IMGTYPE_SRC + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            diam_mm = int(row["diameter"] * images.shape[2])
            anno_index = int(row["anno_index"])
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(diam_mm) + "_1_" + "pos.png", cube_img, 8, 8)
        helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_annotation_images_lidc():
    src_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"

    dst_dir = settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_lidc/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*_annos_pos_lidc.csv")):
        patient_id = ntpath.basename(csv_file).replace("_annos_pos_lidc.csv", "")
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["coord_x"] * images.shape[2])
            coord_y = int(row["coord_y"] * images.shape[1])
            coord_z = int(row["coord_z"] * images.shape[0])
            malscore = int(row["malscore"])
            anno_index = row["anno_index"]
            anno_index = str(anno_index).replace(" ", "xspacex").replace(".", "xpointx").replace("_", "xunderscorex")
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore * malscore) + "_1_pos.png", cube_img, 8, 8)
        helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_pos_annotation_images_manual():
    src_dir = "resources/luna16_manual_labels/"

    dst_dir = settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_manual/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for file_path in glob.glob(dst_dir + "*_manual.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        if "1.3.6.1.4" not in patient_id:
            continue

        print(patient_id)
        # if not "172845185165807139298420209778" in patient_id:
        #     continue
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")

        for index, row in df_annos.iterrows():
            coord_x = int(row["x"] * images.shape[2])
            coord_y = int(row["y"] * images.shape[1])
            coord_z = int(row["z"] * images.shape[0])
            diameter = int(row["d"] * images.shape[2])
            node_type = int(row["id"])
            malscore = int(diameter)
            malscore = min(25, malscore)
            malscore = max(16, malscore)
            anno_index = index
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue

            save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_" + str(malscore) + "_1_" + ("pos" if node_type == 0 else "neg") + ".png", cube_img, 8, 8)
        helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


def make_candidate_auto_images(candidate_types=[]):
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata/luna16_train_cubes_auto/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    for candidate_type in candidate_types:
        for file_path in glob.glob(dst_dir + "*_" + candidate_type + ".png"):
            os.remove(file_path)

    for candidate_type in candidate_types:
        if candidate_type == "falsepos":
            src_dir = "resources/luna16_falsepos_labels/"
        else:
            src_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"

        for index, csv_file in enumerate(glob.glob(src_dir + "*_candidates_" + candidate_type + ".csv")):
            patient_id = ntpath.basename(csv_file).replace("_candidates_" + candidate_type + ".csv", "")
            print(index, ",patient: ", patient_id, " type:", candidate_type)
            # if not "148229375703208214308676934766" in patient_id:
            #     continue
            df_annos = pandas.read_csv(csv_file)
            if len(df_annos) == 0:
                continue
            images = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png", exclude_wildcards=[])

            row_no = 0
            for index, row in df_annos.iterrows():
                coord_x = int(row["coord_x"] * images.shape[2])
                coord_y = int(row["coord_y"] * images.shape[1])
                coord_z = int(row["coord_z"] * images.shape[0])
                anno_index = int(row["anno_index"])
                cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 48)
                if cube_img.sum() < 10:
                    print("Skipping ", coord_x, coord_y, coord_z)
                    continue
                # print(cube_img.sum())
                try:
                    save_cube_img(dst_dir + patient_id + "_" + str(anno_index) + "_0_" + candidate_type + ".png", cube_img, 6, 8)
                except Exception as ex:
                    print(ex)

                row_no += 1
                max_item = 240 if candidate_type == "white" else 200
                if candidate_type == "luna":
                    max_item = 500
                if row_no > max_item:
                    break


def make_pos_annotation_images_manual_ndsb3():
    src_dir = "resources/ndsb3_manual_labels/"
    dst_dir = settings.BASE_DIR_SSD + "generated_traindata/ndsb3_train_cubes_manual/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)


    train_label_df = pandas.read_csv("resources/stage1_labels.csv")
    train_label_df.set_index(["id"], inplace=True)
    for file_path in glob.glob(dst_dir + "*.*"):
        os.remove(file_path)

    for patient_index, csv_file in enumerate(glob.glob(src_dir + "*.csv")):
        patient_id = ntpath.basename(csv_file).replace(".csv", "")
        if "1.3.6.1.4.1" in patient_id:
            continue

        cancer_label = train_label_df.loc[patient_id]["cancer"]
        df_annos = pandas.read_csv(csv_file)
        if len(df_annos) == 0:
            continue
        images = helpers.load_patient_images(patient_id, settings.NDSB3_EXTRACTED_IMAGE_DIR, "*" + CUBE_IMGTYPE_SRC + ".png")

        anno_index = 0
        for index, row in df_annos.iterrows():
            pos_neg = "pos" if row["id"] == 0 else "neg"
            coord_x = int(row["x"] * images.shape[2])
            coord_y = int(row["y"] * images.shape[1])
            coord_z = int(row["z"] * images.shape[0])
            malscore = int(round(row["dmm"]))
            anno_index += 1
            cube_img = get_cube_from_img(images, coord_x, coord_y, coord_z, 64)
            if cube_img.sum() < 5:
                print(" ***** Skipping ", coord_x, coord_y, coord_z)
                continue

            if cube_img.mean() < 10:
                print(" ***** Suspicious ", coord_x, coord_y, coord_z)

            if cube_img.shape != (64, 64, 64):
                print(" ***** incorrect shape !!! ", str(anno_index), " - ",(coord_x, coord_y, coord_z))
                continue
            print(patient_id)
            assert malscore > 0 or pos_neg == "neg"
            save_cube_img(dst_dir + "ndsb3manual_" + patient_id + "_" + str(anno_index) + "_" + pos_neg + "_" + str(cancer_label) + "_" + str(malscore) + "_1_pn.png", cube_img, 8, 8)
        helpers.print_tabbed([patient_index, patient_id, len(df_annos)], [5, 64, 8])


if __name__ == "__main__":
    if not os.path.exists(settings.BASE_DIR_SSD + "generated_traindata/"):
        os.mkdir(settings.BASE_DIR_SSD + "generated_traindata/")

    if True:
        make_annotation_images_lidc()
    if True:
        make_pos_annotation_images_manual()
    # if False:
    #     make_pos_annotation_images()  # not used anymore
    if True:
        make_candidate_auto_images(["falsepos", "luna", "edge"])
    if True:
        make_pos_annotation_images_manual_ndsb3()  # for second model




