import settings
import helpers
import SimpleITK  # conda install -c https://conda.anaconda.org/simpleitk SimpleITK
import numpy
import pandas
import ntpath
import cv2  # conda install -c https://conda.anaconda.org/menpo opencv3
import shutil
import random
import math
import multiprocessing
from bs4 import BeautifulSoup #  conda install beautifulsoup4, coda install lxml
import os
import glob

random.seed(1321)
numpy.random.seed(1321)


def find_mhd_file(patient_id):
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if patient_id in src_path:
                return src_path
    return None


def load_lidc_xml(xml_path, agreement_threshold=0, only_patient=None, save_nodules=False):
    pos_lines = []
    neg_lines = []
    extended_lines = []
    with open(xml_path, 'r') as xml_file:
        markup = xml_file.read()
    xml = BeautifulSoup(markup, features="xml")
    if xml.LidcReadMessage is None:
        return None, None, None
    patient_id = xml.LidcReadMessage.ResponseHeader.SeriesInstanceUid.text

    if only_patient is not None:
        if only_patient != patient_id:
            return None, None, None

    src_path = find_mhd_file(patient_id)
    if src_path is None:
        return None, None, None

    print(patient_id)
    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    rescale = spacing / settings.TARGET_VOXEL_MM

    reading_sessions = xml.LidcReadMessage.find_all("readingSession")
    for reading_session in reading_sessions:
        # print("Sesion")
        nodules = reading_session.find_all("unblindedReadNodule")
        for nodule in nodules:
            nodule_id = nodule.noduleID.text
            # print("  ", nodule.noduleID)
            rois = nodule.find_all("roi")
            x_min = y_min = z_min = 999999
            x_max = y_max = z_max = -999999
            if len(rois) < 2:
                continue

            for roi in rois:
                z_pos = float(roi.imageZposition.text)
                z_min = min(z_min, z_pos)
                z_max = max(z_max, z_pos)
                edge_maps = roi.find_all("edgeMap")
                for edge_map in edge_maps:
                    x = int(edge_map.xCoord.text)
                    y = int(edge_map.yCoord.text)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                if x_max == x_min:
                    continue
                if y_max == y_min:
                    continue

            x_diameter = x_max - x_min
            x_center = x_min + x_diameter / 2
            y_diameter = y_max - y_min
            y_center = y_min + y_diameter / 2
            z_diameter = z_max - z_min
            z_center = z_min + z_diameter / 2
            z_center -= origin[2]
            z_center /= spacing[2]

            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter = max(x_diameter , y_diameter)
            diameter_perc = round(max(x_diameter / img_array.shape[2], y_diameter / img_array.shape[1]), 4)

            if nodule.characteristics is None:
                print("!!!!Nodule:", nodule_id, " has no charecteristics")
                continue
            if nodule.characteristics.malignancy is None:
                print("!!!!Nodule:", nodule_id, " has no malignacy")
                continue

            malignacy = nodule.characteristics.malignancy.text
            sphericiy = nodule.characteristics.sphericity.text
            margin = nodule.characteristics.margin.text
            spiculation = nodule.characteristics.spiculation.text
            texture = nodule.characteristics.texture.text
            calcification = nodule.characteristics.calcification.text
            internal_structure = nodule.characteristics.internalStructure.text
            lobulation = nodule.characteristics.lobulation.text
            subtlety = nodule.characteristics.subtlety.text

            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy]
            extended_line = [patient_id, nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
            pos_lines.append(line)
            extended_lines.append(extended_line)

        nonNodules = reading_session.find_all("nonNodule")
        for nonNodule in nonNodules:
            z_center = float(nonNodule.imageZposition.text)
            z_center -= origin[2]
            z_center /= spacing[2]
            x_center = int(nonNodule.locus.xCoord.text)
            y_center = int(nonNodule.locus.yCoord.text)
            nodule_id = nonNodule.nonNoduleID.text
            x_center_perc = round(x_center / img_array.shape[2], 4)
            y_center_perc = round(y_center / img_array.shape[1], 4)
            z_center_perc = round(z_center / img_array.shape[0], 4)
            diameter_perc = round(max(6 / img_array.shape[2], 6 / img_array.shape[1]), 4)
            # print("Non nodule!", z_center)
            line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, 0]
            neg_lines.append(line)

    if agreement_threshold > 1:
        filtered_lines = []
        for pos_line1 in pos_lines:
            id1 = pos_line1[0]
            x1 = pos_line1[1]
            y1 = pos_line1[2]
            z1 = pos_line1[3]
            d1 = pos_line1[4]
            overlaps = 0
            for pos_line2 in pos_lines:
                id2 = pos_line2[0]
                if id1 == id2:
                    continue
                x2 = pos_line2[1]
                y2 = pos_line2[2]
                z2 = pos_line2[3]
                d2 = pos_line1[4]
                dist = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2))
                if dist < d1 or dist < d2:
                    overlaps += 1
            if overlaps >= agreement_threshold:
                filtered_lines.append(pos_line1)
            # else:
            #     print("Too few overlaps")
        pos_lines = filtered_lines

    df_annos = pandas.DataFrame(pos_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos_lidc.csv", index=False)
    df_neg_annos = pandas.DataFrame(neg_lines, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_neg_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_neg_lidc.csv", index=False)

    # return [patient_id, spacing[0], spacing[1], spacing[2]]
    return pos_lines, neg_lines, extended_lines


def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image


def process_image(src_path):
    patient_id = ntpath.basename(src_path).replace(".mhd", "")
    print("Patient: ", patient_id)

    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)

    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)


    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    img_array = helpers.rescale_patient_images(img_array, spacing, settings.TARGET_VOXEL_MM)

    img_list = []
    for i in range(img_array.shape[0]):
        img = img_array[i]
        seg_img, mask = helpers.get_segmented_lungs(img.copy())
        img_list.append(seg_img)
        img = normalize(img)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_i.png", img * 255)
        cv2.imwrite(dst_dir + "img_" + str(i).rjust(4, '0') + "_m.png", mask * 255)


def process_pos_annotations_patient(src_path, patient_id):
    df_node = pandas.read_csv("resources/luna16_annotations/annotations.csv")
    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_dir = dst_dir + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing /settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

    pos_annos = []
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        diam_mm = annotation["diameter_mm"]
        print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(diam_mm, 2)))
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        # center_int = numpy.rint((center_float - origin) )
        print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        diameter_pixels = diam_mm / settings.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        pos_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(pos_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv", index=False)
    return [patient_id, spacing[0], spacing[1], spacing[2]]


def process_excluded_annotations_patient(src_path, patient_id):
    df_node = pandas.read_csv("resources/luna16_annotations/annotations_excluded.csv")
    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
    dst_dir = dst_dir + patient_id + "/"
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    # pos_annos_df = pandas.read_csv(TRAIN_DIR + "metadata/" + patient_id + "_annos_pos_lidc.csv")
    pos_annos_df = pandas.read_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_pos.csv")
    pos_annos_manual = None
    manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)
        dmm = pos_annos_manual["dmm"]  # check

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    print("Annos: ", len(df_patient))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")

    neg_annos = []
    df_patient = df_node[df_node["seriesuid"] == patient_id]
    anno_index = 0
    for index, annotation in df_patient.iterrows():
        node_x = annotation["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = annotation["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = annotation["coordZ"]
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        diameter_pixels = 6 / settings.TARGET_VOXEL_MM
        diameter_percent = diameter_pixels / float(patient_imgs.shape[1])

        ok = True

        for index, row in pos_annos_df.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            print((pos_coord_x, pos_coord_y, pos_coord_z))
            print(center_float_rescaled)
            dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
            if dist < (diameter + 64):  #  make sure we have a big margin
                ok = False
                print("################### Too close", center_float_rescaled)
                break

        if pos_annos_manual is not None and ok:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * patient_imgs.shape[2]
                pos_coord_y = row["y"] * patient_imgs.shape[1]
                pos_coord_z = row["z"] * patient_imgs.shape[0]
                diameter = row["d"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 72):  #  make sure we have a big margin
                    ok = False
                    print("################### Too close", center_float_rescaled)
                    break

        if not ok:
            continue

        neg_annos.append([anno_index, round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(diameter_percent, 4), 1])
        anno_index += 1

    df_annos = pandas.DataFrame(neg_annos, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_annos.to_csv(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/" + patient_id + "_annos_excluded.csv", index=False)
    return [patient_id, spacing[0], spacing[1], spacing[2]]


def process_luna_candidates_patient(src_path, patient_id):
    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
    img_dir = dst_dir + patient_id + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    pos_annos_manual = None
    manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    direction = numpy.array(itk_img.GetDirection())      # x,y,z  Origin in world coordinates (mm)
    print("Direction: ", direction)
    flip_direction_x = False
    flip_direction_y = False
    if round(direction[0]) == -1:
        origin[0] *= -1
        direction[0] = 1
        flip_direction_x = True
        print("Swappint x origin")
    if round(direction[4]) == -1:
        origin[1] *= -1
        direction[4] = 1
        flip_direction_y = True
        print("Swappint y origin")
    print("Direction: ", direction)
    assert abs(sum(direction) - 3) < 0.01

    src_df = pandas.read_csv("resources/luna16_annotations/" + "candidates_V2.csv")
    src_df = src_df[src_df["seriesuid"] == patient_id]
    src_df = src_df[src_df["class"] == 0]
    patient_imgs = helpers.load_patient_images(patient_id, settings.LUNA16_EXTRACTED_IMAGE_DIR, "*_i.png")
    candidate_list = []

    for df_index, candiate_row in src_df.iterrows():
        node_x = candiate_row["coordX"]
        if flip_direction_x:
            node_x *= -1
        node_y = candiate_row["coordY"]
        if flip_direction_y:
            node_y *= -1
        node_z = candiate_row["coordZ"]
        candidate_diameter = 6
        # print("Node org (x,y,z,diam): ", (round(node_x, 2), round(node_y, 2), round(node_z, 2), round(candidate_diameter, 2)))
        center_float = numpy.array([node_x, node_y, node_z])
        center_int = numpy.rint((center_float-origin) / spacing)
        # center_int = numpy.rint((center_float - origin) )
        # print("Node tra (x,y,z,diam): ", (center_int[0], center_int[1], center_int[2]))
        # center_int_rescaled = numpy.rint(((center_float-origin) / spacing) * rescale)
        center_float_rescaled = (center_float - origin) / settings.TARGET_VOXEL_MM
        center_float_percent = center_float_rescaled / patient_imgs.swapaxes(0, 2).shape
        # center_int = numpy.rint((center_float - origin) )
        # print("Node sca (x,y,z,diam): ", (center_float_rescaled[0], center_float_rescaled[1], center_float_rescaled[2]))
        coord_x = center_float_rescaled[0]
        coord_y = center_float_rescaled[1]
        coord_z = center_float_rescaled[2]

        ok = True

        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * patient_imgs.shape[2]
            pos_coord_y = row["coord_y"] * patient_imgs.shape[1]
            pos_coord_z = row["coord_z"] * patient_imgs.shape[0]
            diameter = row["diameter"] * patient_imgs.shape[2]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 64):  #  make sure we have a big margin
                ok = False
                print("################### Too close", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None and ok:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * patient_imgs.shape[2]
                pos_coord_y = row["y"] * patient_imgs.shape[1]
                pos_coord_z = row["z"] * patient_imgs.shape[0]
                diameter = row["d"] * patient_imgs.shape[2]
                print((pos_coord_x, pos_coord_y, pos_coord_z))
                print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - center_float_rescaled[0], 2) + math.pow(pos_coord_y - center_float_rescaled[1], 2) + math.pow(pos_coord_z - center_float_rescaled[2], 2))
                if dist < (diameter + 72):  #  make sure we have a big margin
                    ok = False
                    print("################### Too close", center_float_rescaled)
                    break

        if not ok:
            continue

        candidate_list.append([len(candidate_list), round(center_float_percent[0], 4), round(center_float_percent[1], 4), round(center_float_percent[2], 4), round(candidate_diameter / patient_imgs.shape[0], 4), 0])

    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_luna.csv", index=False)


def process_auto_candidates_patient(src_path, patient_id, sample_count=1000, candidate_type="white"):
    dst_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + "/_labels/"
    img_dir = settings.LUNA16_EXTRACTED_IMAGE_DIR + patient_id + "/"
    df_pos_annos = pandas.read_csv(dst_dir + patient_id + "_annos_pos_lidc.csv")

    pos_annos_manual = None
    manual_path = settings.EXTRA_DATA_DIR + "luna16_manual_labels/" + patient_id + ".csv"
    if os.path.exists(manual_path):
        pos_annos_manual = pandas.read_csv(manual_path)

    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)

    itk_img = SimpleITK.ReadImage(src_path)
    img_array = SimpleITK.GetArrayFromImage(itk_img)
    print("Img array: ", img_array.shape)
    print("Pos annos: ", len(df_pos_annos))

    num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
    origin = numpy.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
    print("Origin (x,y,z): ", origin)
    spacing = numpy.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
    print("Spacing (x,y,z): ", spacing)
    rescale = spacing / settings.TARGET_VOXEL_MM
    print("Rescale: ", rescale)

    if candidate_type == "white":
        wildcard = "*_c.png"
    else:
        wildcard = "*_m.png"

    src_files = glob.glob(img_dir + wildcard)
    src_files.sort()
    src_candidate_maps = [cv2.imread(src_file, cv2.IMREAD_GRAYSCALE) for src_file in src_files]

    candidate_list = []
    tries = 0
    while len(candidate_list) < sample_count and tries < 10000:
        tries += 1
        coord_z = int(numpy.random.normal(len(src_files) / 2, len(src_files) / 6))
        coord_z = max(coord_z, 0)
        coord_z = min(coord_z, len(src_files) - 1)
        candidate_map = src_candidate_maps[coord_z]
        if candidate_type == "edge":
            candidate_map = cv2.Canny(candidate_map.copy(), 100, 200)

        non_zero_indices = numpy.nonzero(candidate_map)
        if len(non_zero_indices[0]) == 0:
            continue
        nonzero_index = random.randint(0, len(non_zero_indices[0]) - 1)
        coord_y = non_zero_indices[0][nonzero_index]
        coord_x = non_zero_indices[1][nonzero_index]
        ok = True
        candidate_diameter = 6
        for index, row in df_pos_annos.iterrows():
            pos_coord_x = row["coord_x"] * src_candidate_maps[0].shape[1]
            pos_coord_y = row["coord_y"] * src_candidate_maps[0].shape[0]
            pos_coord_z = row["coord_z"] * len(src_files)
            diameter = row["diameter"] * src_candidate_maps[0].shape[1]
            dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
            if dist < (diameter + 48): #  make sure we have a big margin
                ok = False
                print("# Too close", (coord_x, coord_y, coord_z))
                break

        if pos_annos_manual is not None:
            for index, row in pos_annos_manual.iterrows():
                pos_coord_x = row["x"] * src_candidate_maps[0].shape[1]
                pos_coord_y = row["y"] * src_candidate_maps[0].shape[0]
                pos_coord_z = row["z"] * len(src_files)
                diameter = row["d"] * src_candidate_maps[0].shape[1]
                # print((pos_coord_x, pos_coord_y, pos_coord_z))
                # print(center_float_rescaled)
                dist = math.sqrt(math.pow(pos_coord_x - coord_x, 2) + math.pow(pos_coord_y - coord_y, 2) + math.pow(pos_coord_z - coord_z, 2))
                if dist < (diameter + 72):  #  make sure we have a big margin
                    ok = False
                    print("#Too close",  (coord_x, coord_y, coord_z))
                    break

        if not ok:
            continue


        perc_x = round(coord_x / src_candidate_maps[coord_z].shape[1], 4)
        perc_y = round(coord_y / src_candidate_maps[coord_z].shape[0], 4)
        perc_z = round(coord_z / len(src_files), 4)
        candidate_list.append([len(candidate_list), perc_x, perc_y, perc_z, round(candidate_diameter / src_candidate_maps[coord_z].shape[1], 4), 0])

    if tries > 9999:
        print("****** WARING!! TOO MANY TRIES ************************************")
    df_candidates = pandas.DataFrame(candidate_list, columns=["anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore"])
    df_candidates.to_csv(dst_dir + patient_id + "_candidates_" + candidate_type + ".csv", index=False)


def process_images(delete_existing=False, only_process_patient=None):
    if delete_existing and os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        print("Removing old stuff..")
        if os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
            shutil.rmtree(settings.LUNA16_EXTRACTED_IMAGE_DIR)

    if not os.path.exists(settings.LUNA16_EXTRACTED_IMAGE_DIR):
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR)
        os.mkdir(settings.LUNA16_EXTRACTED_IMAGE_DIR + "_labels/")

    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        src_paths = glob.glob(src_dir + "*.mhd")

        if only_process_patient is None and True:
            pool = multiprocessing.Pool(6)
            pool.map(process_image, src_paths)
        else:
            for src_path in src_paths:
                print(src_path)
                if only_process_patient is not None:
                    if only_process_patient not in src_path:
                        continue
                process_image(src_path)


def process_pos_annotations_patient2():
    candidate_index = 0
    only_patient = "197063290812663596858124411210"
    only_patient = None
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print(candidate_index, " patient: ", patient_id)
            process_pos_annotations_patient(src_path, patient_id)
            candidate_index += 1


def process_excluded_annotations_patients(only_patient=None):
    candidate_index = 0
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for src_path in glob.glob(src_dir + "*.mhd"):
            if only_patient is not None and only_patient not in src_path:
                continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print(candidate_index, " patient: ", patient_id)
            process_excluded_annotations_patient(src_path, patient_id)
            candidate_index += 1


def process_auto_candidates_patients():
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
            # if not "100621383016233746780170740405" in src_path:
            #     continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            print("Patient: ", patient_index, " ", patient_id)
            # process_auto_candidates_patient(src_path, patient_id, sample_count=500, candidate_type="white")
            process_auto_candidates_patient(src_path, patient_id, sample_count=200, candidate_type="edge")


def process_luna_candidates_patients(only_patient_id=None):
    for subject_no in range(settings.LUNA_SUBSET_START_INDEX, 10):
        src_dir = settings.LUNA16_RAW_SRC_DIR + "subset" + str(subject_no) + "/"
        for patient_index, src_path in enumerate(glob.glob(src_dir + "*.mhd")):
            # if not "100621383016233746780170740405" in src_path:
            #     continue
            patient_id = ntpath.basename(src_path).replace(".mhd", "")
            if only_patient_id is not None and patient_id != only_patient_id:
                continue
            print("Patient: ", patient_index, " ", patient_id)
            process_luna_candidates_patient(src_path, patient_id)


def process_lidc_annotations(only_patient=None, agreement_threshold=0):
    # lines.append(",".join())
    file_no = 0
    pos_count = 0
    neg_count = 0
    all_lines = []
    for anno_dir in [d for d in glob.glob("resources/luna16_annotations/*") if os.path.isdir(d)]:
        xml_paths = glob.glob(anno_dir + "/*.xml")
        for xml_path in xml_paths:
            print(file_no, ": ",  xml_path)
            pos, neg, extended = load_lidc_xml(xml_path=xml_path, only_patient=only_patient, agreement_threshold=agreement_threshold)
            if pos is not None:
                pos_count += len(pos)
                neg_count += len(neg)
                print("Pos: ", pos_count, " Neg: ", neg_count)
                file_no += 1
                all_lines += extended
            # if file_no > 10:
            #     break

            # extended_line = [nodule_id, x_center_perc, y_center_perc, z_center_perc, diameter_perc, malignacy, sphericiy, margin, spiculation, texture, calcification, internal_structure, lobulation, subtlety ]
    df_annos = pandas.DataFrame(all_lines, columns=["patient_id", "anno_index", "coord_x", "coord_y", "coord_z", "diameter", "malscore", "sphericiy", "margin", "spiculation", "texture", "calcification", "internal_structure", "lobulation", "subtlety"])
    df_annos.to_csv(settings.BASE_DIR + "lidc_annotations.csv", index=False)


if __name__ == "__main__":
    if True:
        only_process_patient = None
        process_images(delete_existing=False, only_process_patient=only_process_patient)

    if True:
        process_lidc_annotations(only_patient=None, agreement_threshold=0)

    if True:
        process_pos_annotations_patient2()
        process_excluded_annotations_patients(only_patient=None)

    if True:
        process_luna_candidates_patients(only_patient_id=None)
    if True:
        process_auto_candidates_patients()
