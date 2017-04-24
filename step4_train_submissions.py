import settings
import helpers
import sys
import os
from collections import defaultdict
import glob
import random
import pandas
import ntpath
import numpy
from sklearn import cross_validation
import xgboost
from sklearn.metrics import log_loss


def combine_nodule_predictions(dirs, train_set=True, nodule_th=0.5, extensions=[""]):
    print("Combining nodule predictions: ", "Train" if train_set else "Submission")
    if train_set:
        labels_df = pandas.read_csv("resources/stage1_labels.csv")
    else:
        labels_df = pandas.read_csv("resources/stage2_sample_submission.csv")

    mass_df = pandas.read_csv(settings.BASE_DIR + "masses_predictions.csv")
    mass_df.set_index(["patient_id"], inplace=True)

    # meta_df = pandas.read_csv(settings.BASE_DIR + "patient_metadata.csv")
    # meta_df.set_index(["patient_id"], inplace=True)

    data_rows = []
    for index, row in labels_df.iterrows():
        patient_id = row["id"]
        # mask = helpers.load_patient_images(patient_id, settings.EXTRACTED_IMAGE_DIR, "*_m.png")
        print(len(data_rows), " : ", patient_id)
        # if len(data_rows) > 19:
        #     break
        cancer_label = row["cancer"]
        mass_pred = int(mass_df.loc[patient_id]["prediction"])
        # meta_row = meta_df.loc[patient_id]
        # z_scale = meta_row["slice_thickness"]
        # x_scale = meta_row["spacingx"]
        # vendor_low = 1 if "1.2.276.0.28.3.145667764438817.42.13928" in meta_row["instance_id"] else 0
        # vendor_high = 1 if "1.3.6.1.4.1.14519.5.2.1.3983.1600" in meta_row["instance_id"] else 0
        #         row_items = [cancer_label, 0, mass_pred, x_scale, z_scale, vendor_low, vendor_high] # mask.sum()

        row_items = [cancer_label, 0, mass_pred] # mask.sum()

        for magnification in [1, 1.5, 2]:
            pred_df_list = []
            for extension in extensions:
                src_dir = settings.NDSB3_NODULE_DETECTION_DIR + "predictions" + str(int(magnification * 10)) + extension + "/"
                pred_nodules_df = pandas.read_csv(src_dir + patient_id + ".csv")
                pred_nodules_df = pred_nodules_df[pred_nodules_df["diameter_mm"] > 0]
                pred_nodules_df = pred_nodules_df[pred_nodules_df["nodule_chance"] > nodule_th]
                pred_df_list.append(pred_nodules_df)

            pred_nodules_df = pandas.concat(pred_df_list, ignore_index=True)

            nodule_count = len(pred_nodules_df)
            nodule_max = 0
            nodule_median = 0
            nodule_chance = 0
            nodule_sum = 0
            coord_z = 0
            second_largest = 0
            nodule_wmax = 0

            count_rows = []
            coord_y = 0
            coord_x = 0

            if len(pred_nodules_df) > 0:
                max_index = pred_nodules_df["diameter_mm"].argmax
                max_row = pred_nodules_df.loc[max_index]
                nodule_max = round(max_row["diameter_mm"], 2)
                nodule_chance = round(max_row["nodule_chance"], 2)
                nodule_median = round(pred_nodules_df["diameter_mm"].median(), 2)
                nodule_wmax = round(nodule_max * nodule_chance, 2)
                coord_z = max_row["coord_z"]
                coord_y = max_row["coord_y"]
                coord_x = max_row["coord_x"]


                rows = []
                for row_index, row in pred_nodules_df.iterrows():
                    dist = helpers.get_distance(max_row, row)
                    if dist > 0.2:
                        nodule_mal = row["diameter_mm"]
                        if nodule_mal > second_largest:
                            second_largest = nodule_mal
                    rows.append(row)

                count_rows = []
                for row in rows:
                    ok = True
                    for count_row in count_rows:
                        dist = helpers.get_distance(count_row, row)
                        if dist < 0.2:
                            ok = False
                    if ok:
                        count_rows.append(row)
            nodule_count = len(count_rows)
            row_items += [nodule_max, nodule_chance, nodule_count, nodule_median, nodule_wmax, coord_z, second_largest, coord_y, coord_x]

        row_items.append(patient_id)
        data_rows.append(row_items)

    # , "x_scale", "z_scale", "vendor_low", "vendor_high"
    columns = ["cancer_label", "mask_size", "mass"]
    for magnification in [1, 1.5, 2]:
        str_mag = str(int(magnification * 10))
        columns.append("mx_" + str_mag)
        columns.append("ch_" + str_mag)
        columns.append("cnt_" + str_mag)
        columns.append("med_" + str_mag)
        columns.append("wmx_" + str_mag)
        columns.append("crdz_" + str_mag)
        columns.append("mx2_" + str_mag)
        columns.append("crdy_" + str_mag)
        columns.append("crdx_" + str_mag)

    columns.append("patient_id")
    res_df = pandas.DataFrame(data_rows, columns=columns)

    if not os.path.exists(settings.BASE_DIR + "xgboost_trainsets/"):
        os.mkdir(settings.BASE_DIR + "xgboost_trainsets/")
    target_path = settings.BASE_DIR + "xgboost_trainsets/" "train" + extension + ".csv" if train_set else settings.BASE_DIR + "xgboost_trainsets/" + "submission" + extension + ".csv"
    res_df.to_csv(target_path, index=False)



def train_xgboost_on_combined_nodules_ensembletest(fixed_holdout=False, submission_is_fixed_holdout=False, ensemble_lists=[]):
    train_cols = ["mass", "mx_10", "mx_20", "mx_15", "crdz_10", "crdz_15", "crdz_20"]
    runs = 5 if fixed_holdout else 1000
    test_size = 0.1
    record_count = 0
    seed = random.randint(0, 500) if fixed_holdout else 4242

    variants = []
    x_variants = dict()
    y_variants = dict()
    for ensemble in ensemble_lists:
        for variant in ensemble:
            variants.append(variant)
            df_train = pandas.read_csv(settings.BASE_DIR + "xgboost_trainsets/" + "train" + variant + ".csv")

            y = df_train["cancer_label"].as_matrix()
            y = y.reshape(y.shape[0], 1)

            cols = df_train.columns.values.tolist()
            cols.remove("cancer_label")
            cols.remove("patient_id")
            x = df_train[train_cols].as_matrix()

            x_variants[variant] = x
            record_count = len(x)
            y_variants[variant] = y

    scores = defaultdict(lambda: [])
    ensemble_scores = []
    for i in range(runs):
        submission_preds_list = defaultdict(lambda: [])
        train_preds_list = defaultdict(lambda: [])
        holdout_preds_list = defaultdict(lambda: [])

        train_test_mask = numpy.random.choice([True, False], record_count, p=[0.8, 0.2])
        for variant in variants:
            x = x_variants[variant]
            y = y_variants[variant]
            x_train = x[train_test_mask]
            y_train = y[train_test_mask]
            x_holdout = x[~train_test_mask]
            y_holdout = y[~train_test_mask]
            if fixed_holdout:
                x_train = x[300:]
                y_train = y[300:]
                x_holdout = x[:300]
                y_holdout = y[:300]

            if True:
                clf = xgboost.XGBRegressor(max_depth=4,
                                           n_estimators=80, #50
                                           learning_rate=0.05,
                                           min_child_weight=60,
                                           nthread=8,
                                           subsample=0.95, #95
                                           colsample_bytree=0.95, # 95
                                           # subsample=1.00,
                                           # colsample_bytree=1.00,
                                           seed=seed)
                #
                clf.fit(x_train, y_train, verbose=fixed_holdout and False, eval_set=[(x_train, y_train), (x_holdout, y_holdout)], eval_metric="logloss", early_stopping_rounds=5, )
                holdout_preds = clf.predict(x_holdout)

            holdout_preds = numpy.clip(holdout_preds, 0.001, 0.999)
            # holdout_preds *= 0.93
            holdout_preds_list[variant].append(holdout_preds)
            train_preds_list[variant].append(holdout_preds.mean())
            score = log_loss(y_holdout, holdout_preds, normalize=True)
            print(score, "\tbest:\t", clf.best_score, "\titer\t", clf.best_iteration, "\tmean:\t", train_preds_list[-1], "\thomean:\t", y_holdout.mean(), " variant:", variant)
            scores[variant].append(score)

        total_predictions = []
        for ensemble in ensemble_lists:
            ensemble_predictions = []
            for variant in ensemble:
                variant_predictions = numpy.array(holdout_preds_list[variant], dtype=numpy.float)
                ensemble_predictions.append(variant_predictions.swapaxes(0, 1))
            ensemble_predictions_np = numpy.hstack(ensemble_predictions)
            ensemble_predictions_np = ensemble_predictions_np.mean(axis=1)
            score = log_loss(y_holdout, ensemble_predictions_np, normalize=True)
            print(score)
            total_predictions.append(ensemble_predictions_np.reshape(ensemble_predictions_np.shape[0], 1))
        total_predictions_np = numpy.hstack(total_predictions)
        total_predictions_np = total_predictions_np.mean(axis=1)
        score = log_loss(y_holdout, total_predictions_np, normalize=True)
        print("Total: ", score)
        ensemble_scores.append(score)

    print("Average score: ", sum(ensemble_scores) / len(ensemble_scores))


def train_xgboost_on_combined_nodules(extension, fixed_holdout=False, submission=False, submission_is_fixed_holdout=False):
    df_train = pandas.read_csv(settings.BASE_DIR + "xgboost_trainsets/" + "train" + extension + ".csv")
    if submission:
        df_submission = pandas.read_csv(settings.BASE_DIR + "xgboost_trainsets/" + "submission" + extension + ".csv")
        submission_y = numpy.zeros((len(df_submission), 1))

    if submission_is_fixed_holdout:
        df_submission = df_train[:300]
        df_train = df_train[300:]
        submission_y = df_submission["cancer_label"].as_matrix()
        submission_y = submission_y.reshape(submission_y.shape[0], 1)

    y = df_train["cancer_label"].as_matrix()
    y = y.reshape(y.shape[0], 1)
    # print("Mean y: ", y.mean())

    cols = df_train.columns.values.tolist()
    cols.remove("cancer_label")
    cols.remove("patient_id")

    train_cols = ["mass", "mx_10", "mx_20", "mx_15", "crdz_10", "crdz_15", "crdz_20"]
    x = df_train[train_cols].as_matrix()
    if submission:
        x_submission = df_submission[train_cols].as_matrix()

    if submission_is_fixed_holdout:
        x_submission = df_submission[train_cols].as_matrix()

    runs = 20 if fixed_holdout else 1000
    scores = []
    submission_preds_list = []
    train_preds_list = []
    holdout_preds_list = []
    for i in range(runs):
        test_size = 0.1 if submission else 0.1
        # stratify=y,
        x_train, x_holdout, y_train, y_holdout = cross_validation.train_test_split(x, y,  test_size=test_size)
        # print(y_holdout.mean())
        if fixed_holdout:
            x_train = x[300:]
            y_train = y[300:]
            x_holdout = x[:300]
            y_holdout = y[:300]

        seed = random.randint(0, 500) if fixed_holdout else 4242
        if True:
            clf = xgboost.XGBRegressor(max_depth=4,
                                       n_estimators=80, #55
                                       learning_rate=0.05,
                                       min_child_weight=60,
                                       nthread=8,
                                       subsample=0.95, #95
                                       colsample_bytree=0.95, # 95
                                       # subsample=1.00,
                                       # colsample_bytree=1.00,
                                       seed=seed)
            #
            clf.fit(x_train, y_train, verbose=fixed_holdout and False, eval_set=[(x_train, y_train), (x_holdout, y_holdout)], eval_metric="logloss", early_stopping_rounds=5, )
            holdout_preds = clf.predict(x_holdout)

        holdout_preds = numpy.clip(holdout_preds, 0.001, 0.999)
        # holdout_preds *= 0.93
        holdout_preds_list.append(holdout_preds)
        train_preds_list.append(holdout_preds.mean())
        score = log_loss(y_holdout, holdout_preds, normalize=True)

        print(score, "\tbest:\t", clf.best_score, "\titer\t", clf.best_iteration, "\tmean:\t", train_preds_list[-1], "\thomean:\t", y_holdout.mean())
        scores.append(score)

        if submission_is_fixed_holdout:
            submission_preds = clf.predict(x_submission)
            submission_preds_list.append(submission_preds)

        if submission:
            submission_preds = clf.predict(x_submission)
            submission_preds_list.append(submission_preds)

    if fixed_holdout:
        all_preds = numpy.vstack(holdout_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.001] = 0.001
        avg_preds[avg_preds > 0.999] = 0.999
        deltas = numpy.abs(avg_preds.reshape(300) - y_holdout.reshape(300))
        df_train = df_train[:300]
        df_train["deltas"] = deltas
        # df_train.to_csv("c:/tmp/deltas.csv")
        loss = log_loss(y_holdout, avg_preds)
        print("Fixed holout avg score: ", loss)
        # print("Fixed holout mean: ", y_holdout.mean())

    if submission:
        all_preds = numpy.vstack(submission_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.01] = 0.01
        avg_preds[avg_preds > 0.99] = 0.99
        submission_preds_list = avg_preds.tolist()
        df_submission["id"] = df_submission["patient_id"]
        df_submission["cancer"] = submission_preds_list
        df_submission = df_submission[["id", "cancer"]]
        if not os.path.exists("submission/"):
            os.mkdir("submission/")
        if not os.path.exists("submission/level1/"):
            os.mkdir("submission/level1/")

        df_submission.to_csv("submission/level1/s" + extension + ".csv", index=False)
        # print("Submission mean chance: ", avg_preds.mean())

    if submission_is_fixed_holdout:
        all_preds = numpy.vstack(submission_preds_list)
        avg_preds = numpy.average(all_preds, axis=0)
        avg_preds[avg_preds < 0.01] = 0.01
        avg_preds[avg_preds > 0.99] = 0.99
        submission_preds_list = avg_preds.tolist()
        loss = log_loss(submission_y, submission_preds_list)
        # print("First 300 patients : ", loss)
    if submission_is_fixed_holdout:
        print("First 300 patients score: ", sum(scores) / len(scores), " mean chance: ", sum(train_preds_list) / len(train_preds_list))
    else:
        print("Average score: ", sum(scores) / len(scores), " mean chance: ", sum(train_preds_list) / len(train_preds_list))


def combine_submissions(level, model_type=None):
    print("Combine submissions.. level: ", level, " model_type: ", model_type)
    src_dir = "submission/level" + str(level) + "/"

    dst_dir = "submission/"
    if level == 1:
        dst_dir += "level2/"
    if not os.path.exists("submission/level2/"):
        os.mkdir("submission/level2/")

    submission_df = pandas.read_csv("resources/stage2_sample_submission.csv")
    submission_df["id2"] = submission_df["id"]
    submission_df.set_index(["id2"], inplace=True)
    search_expr = "*.csv" if model_type is None else "*" + model_type + "*.csv"
    csvs = glob.glob(src_dir + search_expr)
    print(len(csvs), " found..")
    for submission_idx, submission_path in enumerate(csvs):
        print(ntpath.basename(submission_path))
        column_name = "s" + str(submission_idx)
        submission_df[column_name] = 0
        sub_df = pandas.read_csv(submission_path)
        for index, row in sub_df.iterrows():
            patient_id = row["id"]
            cancer = row["cancer"]
            submission_df.loc[patient_id, column_name] = cancer

    submission_df["cancer"] = 0
    for i in range(len(csvs)):
        submission_df["cancer"] += submission_df["s" + str(i)]
    submission_df["cancer"] /= len(csvs)

    if not os.path.exists(dst_dir + "debug/"):
        os.mkdir(dst_dir + "debug/")
    if level == 2:
        target_path = dst_dir + "final_submission.csv"
        target_path_allcols = dst_dir + "debug/final_submission.csv"
    else:
        target_path_allcols = dst_dir + "debug/" + "combined_submission_" + model_type + ".csv"
        target_path = dst_dir + "combined_submission_" + model_type + ".csv"

    submission_df.to_csv(target_path_allcols, index=False)
    submission_df[["id", "cancer"]].to_csv(target_path, index=False)


if __name__ == "__main__":
    if True:
        for model_variant in ["_luna16_fs", "_luna_posnegndsb_v1", "_luna_posnegndsb_v2"]:
            print("Variant: ", model_variant)
            if True:
                combine_nodule_predictions(None, train_set=True, nodule_th=0.7, extensions=[model_variant])
                combine_nodule_predictions(None, train_set=False, nodule_th=0.7, extensions=[model_variant])
            if True:
                train_xgboost_on_combined_nodules(fixed_holdout=False, submission=True, submission_is_fixed_holdout=False, extension=model_variant)
                train_xgboost_on_combined_nodules(fixed_holdout=True, extension=model_variant)

    combine_submissions(level=1, model_type="luna_posnegndsb")
    combine_submissions(level=1, model_type="luna16_fs")
    combine_submissions(level=1, model_type="daniel")
    combine_submissions(level=2)
