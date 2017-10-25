import settings
import helpers

import os
import glob
import random
import ntpath
import cv2
import numpy
from typing import List, Tuple
from keras.optimizers import Adam, SGD
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge, BatchNormalization, SpatialDropout2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import pandas
import shutil

MEAN_FRAME_COUNT = 1
CHANNEL_COUNT = 1


def random_scale_img(img, xy_range, lock_xy=False):
    if random.random() > xy_range.chance:
        return img

    if not isinstance(img, list):
        img = [img]

    import cv2
    scale_x = random.uniform(xy_range.x_min, xy_range.x_max)
    scale_y = random.uniform(xy_range.y_min, xy_range.y_max)
    if lock_xy:
        scale_y = scale_x

    org_height, org_width = img[0].shape[:2]
    xy_range.last_x = scale_x
    xy_range.last_y = scale_y

    res = []
    for img_inst in img:
        scaled_width = int(org_width * scale_x)
        scaled_height = int(org_height * scale_y)
        scaled_img = cv2.resize(img_inst, (scaled_width, scaled_height), interpolation=cv2.INTER_CUBIC)
        if scaled_width < org_width:
            extend_left = (org_width - scaled_width) / 2
            extend_right = org_width - extend_left - scaled_width
            scaled_img = cv2.copyMakeBorder(scaled_img, 0, 0, extend_left, extend_right, borderType=cv2.BORDER_CONSTANT)
            scaled_width = org_width

        if scaled_height < org_height:
            extend_top = (org_height - scaled_height) / 2
            extend_bottom = org_height - extend_top - scaled_height
            scaled_img = cv2.copyMakeBorder(scaled_img, extend_top, extend_bottom, 0, 0,  borderType=cv2.BORDER_CONSTANT)
            scaled_height = org_height

        start_x = (scaled_width - org_width) / 2
        start_y = (scaled_height - org_height) / 2
        tmp = scaled_img[start_y: start_y + org_height, start_x: start_x + org_width]
        res.append(tmp)

    return res


class XYRange:
    def __init__(self, x_min, x_max, y_min, y_max, chance=1.0):
        self.chance = chance
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.last_x = 0
        self.last_y = 0

    def get_last_xy_txt(self):
        res = "x_" + str(int(self.last_x * 100)).replace("-", "m") + "-" + "y_" + str(int(self.last_y * 100)).replace("-", "m")
        return res


def random_translate_img(img, xy_range, border_mode="constant"):
    if random.random() > xy_range.chance:
        return img
    import cv2
    if not isinstance(img, list):
        img = [img]

    org_height, org_width = img[0].shape[:2]
    translate_x = random.randint(xy_range.x_min, xy_range.x_max)
    translate_y = random.randint(xy_range.y_min, xy_range.y_max)
    trans_matrix = numpy.float32([[1, 0, translate_x], [0, 1, translate_y]])

    border_const = cv2.BORDER_CONSTANT
    if border_mode == "reflect":
        border_const = cv2.BORDER_REFLECT

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, trans_matrix, (org_width, org_height), borderMode=border_const)
        res.append(img_inst)
    if len(res) == 1:
        res = res[0]
    xy_range.last_x = translate_x
    xy_range.last_y = translate_y
    return res


def random_rotate_img(img, chance, min_angle, max_angle):
    import cv2
    if random.random() > chance:
        return img
    if not isinstance(img, list):
        img = [img]

    angle = random.randint(min_angle, max_angle)
    center = (img[0].shape[0] / 2, img[0].shape[1] / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

    res = []
    for img_inst in img:
        img_inst = cv2.warpAffine(img_inst, rot_matrix, dsize=img_inst.shape[:2], borderMode=cv2.BORDER_CONSTANT)
        res.append(img_inst)
    if len(res) == 0:
        res = res[0]
    return res


def random_flip_img(img, horizontal_chance=0, vertical_chance=0):
    import cv2
    flip_horizontal = False
    if random.random() < horizontal_chance:
        flip_horizontal = True

    flip_vertical = False
    if random.random() < vertical_chance:
        flip_vertical = True

    if not flip_horizontal and not flip_vertical:
        return img

    flip_val = 1
    if flip_vertical:
        flip_val = -1 if flip_horizontal else 0

    if not isinstance(img, list):
        res = cv2.flip(img, flip_val) # 0 = X axis, 1 = Y axis,  -1 = both
    else:
        res = []
        for img_item in img:
            img_flip = cv2.flip(img_item, flip_val)
            res.append(img_flip)
    return res


ELASTIC_INDICES = None  # needed to make it faster to fix elastic deformation per epoch.
def elastic_transform(image, alpha, sigma, random_state=None):
    global ELASTIC_INDICES
    shape = image.shape

    if ELASTIC_INDICES == None:
        if random_state is None:
            random_state = numpy.random.RandomState(1301)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
        ELASTIC_INDICES = numpy.reshape(y + dy, (-1, 1)), numpy.reshape(x + dx, (-1, 1))
    return map_coordinates(image, ELASTIC_INDICES, order=1).reshape(shape)


def prepare_image_for_net(img):
    img = img.astype(numpy.float)
    img /= 255.
    if len(img.shape) == 3:
        img = img.reshape(img.shape[-3], img.shape[-2], img.shape[-1])
    else:
        img = img.reshape(1, img.shape[-2], img.shape[-1], 1)
    return img


def get_train_holdout_files(model_type, holdout, train_percentage=80, frame_count=8):
    print("Get train/holdout files.")
    file_paths = glob.glob("resources/segmenter_traindata/" + "*_1.png")
    file_paths.sort()
    train_res = []
    holdout_res = []
    for index, file_path in enumerate(file_paths):
        file_name = ntpath.basename(file_path)
        overlay_path = file_path.replace("_1.png", "_o.png")
        train_set = False
        if "1.3.6.1.4" in file_name or "spie" in file_name or "TIME" in file_name:
            train_set = True
        else:
            patient_id = file_name.split("_")[0]
            if helpers.get_patient_fold(patient_id) % 3 != holdout:
                train_set = True

        if train_set:
            train_res.append((file_path, overlay_path))
        else:
            holdout_res.append((file_path, overlay_path))
    print("Train count: ", len(train_res), ", holdout count: ", len(holdout_res))
    return train_res, holdout_res


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)


def dice_coef_np(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = numpy.sum(y_true_f * y_pred_f)
    return (2. * intersection + 100) / (numpy.sum(y_true_f) + numpy.sum(y_pred_f) + 100)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


class DumpPredictions(Callback):

    def __init__(self, dump_filelist : List[Tuple[str, str]], model_type):
        super(DumpPredictions, self).__init__()
        self.dump_filelist = dump_filelist
        self.batch_count = 0
        if not os.path.exists("workdir/segmenter/"):
            os.mkdir("workdir/segmenter/")
        for file_path in glob.glob("workdir/segmenter/*.*"):
            os.remove(file_path)
        self.model_type = model_type

    def on_epoch_end(self, epoch, logs=None):
        model = self.model  # type: Model
        generator = image_generator(self.dump_filelist, 1, train_set=False, model_type=self.model_type)
        for i in range(0, 10):
            x, y = next(generator)
            y_pred = model.predict(x, batch_size=1)

            x = x.swapaxes(0, 3)
            x = x[0]
            # print(x.shape, y.shape, y_pred.shape)
            x *= 255.
            x = x.reshape((x.shape[0], x.shape[0])).astype(numpy.uint8)
            y *= 255.
            y = y.reshape((y.shape[1], y.shape[2])).astype(numpy.uint8)
            y_pred *= 255.
            y_pred = y_pred.reshape((y_pred.shape[1], y_pred.shape[2])).astype(numpy.uint8)
            # cv2.imwrite("workdir/segmenter/img_{0:03d}_{1:02d}_i.png".format(epoch, i), x)
            # cv2.imwrite("workdit/segmenter/img_{0:03d}_{1:02d}_o.png".format(epoch, i), y)
            # cv2.imwrite("workdit/segmenter/img_{0:03d}_{1:02d}_p.png".format(epoch, i), y_pred)


def image_generator(batch_files, batch_size, train_set, model_type):
    global ELASTIC_INDICES
    while True:
        if train_set:
            random.shuffle(batch_files)

        img_list = []
        overlay_list = []
        ELASTIC_INDICES = None
        for batch_file_idx, batch_file in enumerate(batch_files):
            images = []
            img = cv2.imread(batch_file[0], cv2.IMREAD_GRAYSCALE)
            images.append(img)
            overlay = cv2.imread(batch_file[1], cv2.IMREAD_GRAYSCALE)

            if train_set:
                if random.randint(0, 100) > 50:
                    for img_index, img in enumerate(images):
                        images[img_index] = elastic_transform(img, 128, 15)
                    overlay = elastic_transform(overlay, 128, 15)

                if True:
                    augmented = images + [overlay]
                    augmented = random_rotate_img(augmented, 0.8, -20, 20)
                    augmented = random_flip_img(augmented, 0.5, 0.5)

                    # processed = helpers_augmentation.random_flip_img(processed, horizontal_chance=0.5, vertical_chance=0)
                    # processed = helpers_augmentation.random_scale_img(processed, xy_range=helpers_augmentation.XYRange(x_min=0.8, x_max=1.2, y_min=0.8, y_max=1.2, chance=1.0))
                    augmented = random_translate_img(augmented, XYRange(-30, 30, -30, 30, 0.8))
                    images = augmented[:-1]
                    overlay = augmented[-1]

            for index, img in enumerate(images):
                # img = img[crop_y: crop_y + settings.TRAIN_IMG_HEIGHT3D, crop_x: crop_x + settings.TRAIN_IMG_WIDTH3D]
                img = prepare_image_for_net(img)
                images[index] = img

            # helpers_augmentation.dump_augmented_image(img, mean_img=None, target_path="c:\\tmp\\" + batch_file[0])
            # overlay = overlay[crop_y: crop_y + settings.TRAIN_IMG_HEIGHT3D, crop_x: crop_x + settings.TRAIN_IMG_WIDTH3D]
            overlay = prepare_image_for_net(overlay)
            # overlay = overlay.reshape(1, overlay.shape[-3] * overlay.shape[-2])
            # overlay *= settings.OVERLAY_MULTIPLIER
            images3d = numpy.vstack(images)
            images3d = images3d.swapaxes(0, 3)

            img_list.append(images3d)
            overlay_list.append(overlay)
            if len(img_list) >= batch_size:
                x = numpy.vstack(img_list)
                y = numpy.vstack(overlay_list)
                # if len(img_list) >= batch_size:
                yield x, y
                img_list = []
                overlay_list = []


def get_unet(learn_rate, load_weights_path=None) -> Model:
    inputs = Input((settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE, CHANNEL_COUNT))
    filter_size = 32
    growth_step = 32
    x = BatchNormalization()(inputs)
    conv1 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(x)
    conv1 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    pool1 = BatchNormalization()(pool1)
    filter_size += growth_step
    conv2 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = BatchNormalization()(pool2)

    filter_size += growth_step
    conv3 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = BatchNormalization()(pool3)

    filter_size += growth_step
    conv4 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = BatchNormalization()(pool4)

    conv5 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same', name="conv5b")(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name="pool5")(conv5)
    pool5 = BatchNormalization()(pool5)

    conv6 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same', name="conv6b")(conv6)

    up6 = UpSampling2D(size=(2, 2), name="up6")(conv6)
    up6 = merge([up6, conv5], mode='concat', concat_axis=3)
    up6 = BatchNormalization()(up6)

    # up6 = SpatialDropout2D(0.1)(up6)
    filter_size -= growth_step
    conv66 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up6)
    conv66 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv66)

    up7 = merge([UpSampling2D(size=(2, 2))(conv66), conv4], mode='concat', concat_axis=3)
    up7 = BatchNormalization()(up7)
    # up7 = SpatialDropout2D(0.1)(up7)

    filter_size -= growth_step
    conv7 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv3], mode='concat', concat_axis=3)
    up8 = BatchNormalization()(up8)
    filter_size -= growth_step
    conv8 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv8)


    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv2], mode='concat', concat_axis=3)
    up9 = BatchNormalization()(up9)
    conv9 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(filter_size, 3, 3, activation='relu', border_mode='same')(conv9)
    # conv9 = BatchNormalization()(conv9)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(up10)

    model = Model(input=inputs, output=conv10)
    # model.load_weights(load_weights_path)
    # model.compile(optimizer=Adam(lr=1.0e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.compile(optimizer=SGD(lr=learn_rate, momentum=0.9, nesterov=True), loss=dice_coef_loss, metrics=[dice_coef])

    model.summary()
    return model


def train_model(holdout, model_type, continue_from=None):
    batch_size = 4
    train_percentage = 80 if model_type == "masses" else 90
    train_files, holdout_files = get_train_holdout_files( model_type, holdout, train_percentage, frame_count=CHANNEL_COUNT)
    # train_files = train_files[:100]
    # holdout_files = train_files[:10]

    tmp_gen = image_generator(train_files[:2], 2, True, model_type)
    for i in range(10):
        x = next(tmp_gen)
        img = x[0][0].reshape((settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE))
        img *= 255
        # cv2.imwrite("c:/tmp/img_" + str(i).rjust(3, '0') + "i.png", img)
        img = x[1][0].reshape((settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE))
        img *= 255
        # cv2.imwrite("c:/tmp/img_" + str(i).rjust(3, '0') + "o.png", img)
        # print(x.shape)

    train_gen = image_generator(train_files, batch_size, True, model_type)
    holdout_gen = image_generator(holdout_files, batch_size, False, model_type)

    if continue_from is None:
        model = get_unet(0.001)
    else:
        model = get_unet(0.0001)
        model.load_weights(continue_from)

    checkpoint1 = ModelCheckpoint("workdir/" + model_type +"_model_h" + str(holdout) + "_{epoch:02d}-{val_loss:.2f}.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    checkpoint2 = ModelCheckpoint("workdir/" + model_type +"_model_h" + str(holdout) + "_best.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    files = []
    idx = 0
    while (idx < (len(holdout_files))):
        files.append(holdout_files[idx])
        idx += 5
    dumper = DumpPredictions(holdout_files[::10], model_type)
    epoch_div = 1
    epoch_count = 200 if model_type == "masses" else 50
    model.fit_generator(train_gen, len(train_files) / epoch_div, epoch_count, validation_data=holdout_gen, nb_val_samples=len(holdout_files) / epoch_div, callbacks=[checkpoint1, checkpoint2, dumper])
    if not os.path.exists("models"):
        os.mkdir("models")
    shutil.copy("workdir/" + model_type +"_model_h" + str(holdout) + "_best.hd5", "models/" + model_type +"_model_h" + str(holdout) + "_best.hd5")

def predict_patients(patients_dir, model_path, holdout, patient_predictions, model_type):
    model = get_unet(0.001)
    model.load_weights(model_path)
    for item_name in os.listdir(patients_dir):
        if not os.path.isdir(patients_dir + item_name):
            continue
        patient_id = item_name

        if holdout >= 0:
            patient_fold = helpers.get_patient_fold(patient_id, submission_set_neg=True)
            if patient_fold < 0:
                if holdout != 0:
                    continue
            else:
                patient_fold %= 3
                if patient_fold != holdout:
                    continue

        # if "100953483028192176989979435275" not in patient_id:
        #     continue
        print(patient_id)
        patient_dir = patients_dir + patient_id + "/"
        mass = 0
        img_type = "_i" if model_type == "masses" else "_c"
        slices = glob.glob(patient_dir + "*" + img_type + ".png")
        if model_type == "emphysema":
            slices = slices[int(len(slices) / 2):]
        for img_path in slices:
            src_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            src_img = cv2.resize(src_img, dsize=(settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE))
            src_img = prepare_image_for_net(src_img)
            p = model.predict(src_img, batch_size=1)
            p[p < 0.5] = 0
            mass += p.sum()
            p = p[0, :, :, 0] * 255
            # cv2.imwrite(img_path.replace("_i.png", "_mass.png"), p)
            src_img = src_img.reshape((settings.SEGMENTER_IMG_SIZE, settings.SEGMENTER_IMG_SIZE))
            src_img *= 255
            # src_img = cv2.cvtColor(src_img.astype(numpy.uint8), cv2.COLOR_GRAY2BGR)
            # p = cv2.cvtColor(p.astype(numpy.uint8), cv2.COLOR_GRAY2BGRA)
            src_img = cv2.addWeighted(p.astype(numpy.uint8), 0.2, src_img.astype(numpy.uint8), 1 - 0.2, 0)
            cv2.imwrite(img_path.replace(img_type + ".png", "_" + model_type + "o.png"), src_img)

        if mass > 1:
            print(model_type + ": ", mass)
        patient_predictions.append((patient_id, mass))
        df = pandas.DataFrame(patient_predictions, columns=["patient_id", "prediction"])
        df.to_csv(settings.BASE_DIR + model_type + "_predictions.csv", index=False)


if __name__ == "__main__":
    continue_from = None
    if True:
        for model_type_name in ["masses"]:
            train_model(holdout=0, model_type=model_type_name, continue_from=continue_from)
            train_model(holdout=1, model_type=model_type_name, continue_from=continue_from)
            train_model(holdout=2, model_type=model_type_name, continue_from=continue_from)

    if True:
        for model_type_name in ["masses"]:
            patient_predictions_global = []
            for holdout_no in [0, 1, 2]:
                patient_base_dir = settings.NDSB3_EXTRACTED_IMAGE_DIR
                predict_patients(patients_dir=patient_base_dir, model_path="models/" + model_type_name + "_model_h" + str(holdout_no) + "_best.hd5", holdout=holdout_no, patient_predictions=patient_predictions_global, model_type=model_type_name)


