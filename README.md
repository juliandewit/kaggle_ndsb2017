# Kaggle national datascience bowl 2017 2nd place code
This is the source code for my part of the 2nd place solution to the [National Data Science Bowl 2017](https://www.kaggle.com/c/data-science-bowl-2017/) hosted by Kaggle.com. For documenation about the approach go to: http://juliandewit.github.io/kaggle-ndsb2017/
<br>
Note that this is my part of the code.<br> The work of my teammate Daniel Hammack can be found here: https://github.com/dhammack/DSB2017

#### Dependencies & data
The solution is built using Keras with a tensorflow backend on windows 64bit.
Next to this I used scikit-learn, pydicom, simpleitk, beatifulsoup, opencv and XgBoost.
All in all it was quite an engineering effort.

#### General
The source is cleaned up as much as possible. However I was afraid that results would not be 100% reproducible if I changed too much. Therefore some pieces could be a bit cleaner. Also I left in some bugs that I found while cleaning up. (See end of this document),

The solution relies on manual labels, generated labels and 2 resulting submissions from team member Daniel Hammack. These files are all in the "resources" map. All other file location can be configured in the settings.py. The raw patient data must be downloaded from the Kaggle website and the LUNA16 website. 

Trained models as provided to Kaggle after phase 1 are also provided through the following download: https://retinopaty.blob.core.windows.net/ndsb3/trained_models.rar

The solution is a combination of nodule detectors/malignancy regressors. My two parts are trained with LUNA16 data with a mix of positive and negative labels + malignancy info from the LIDC dataset. My second part also uses some manual annotations made on the NDSB3 trainset. Predictions are generated from the raw nodule/malignancy predictions combined with the location information and general “mass” information. Masses are no nodules but big suspicious tissues present in the CT-images. De masses are detected with a U-net trained with manual labels.

The 3rd and 4th part of te solution come from Daniel Hammack. 
The final solution is a blend of the 4 different part. Blending is done by taking a simple average.

#### Preprocessing
First run *step1_preprocess_ndsb.py*. This will extract all the ndsb dicom files , scale to 1x1x1 mm, and make a directory containing .png slice images. Lung segmentation mask images are also generated. They will be used later in the process for faster predicting.
Then run *step1_preprocess_luna16.py*. This will extract all the LUNA source files , scale to 1x1x1 mm, and make a directory containing .png slice images. Lung segmentation mask images are also generated. This step also generates various CSV files for positive and negative examples.

The nodule detectors are trained on positive and negative 3d cubes which must be generated from the LUNA16 and NDSB datasets. *step1b_preprocess_make_train_cubes.py* takes the different csv files and cuts out 3d cubes from the patient slices. The cubes are saved in different directories. *resources/step1_preprocess_mass_segmenter.py* is to generate the mass u-net trainset. It can be run but the generated resized images + labels is provided in this archive so this step does not need to be run. However, this file can be used to regenerate the traindata.

#### Training neural nets
First train the 3D convnets that detect nodules and predict malignancy. This can be done by running 
the *step2_train_nodule_detector.py* file. This will train various combinations of positive and negative labels. The resulting models (NAMES) are stored in the ./workdir directory and the final results are copied to the models folder.
The mass detector can be trained using *step2_train_mass_segmenter.py*. It trains 3 folds and final models are stored in the models (names) folder. Training the 3D convnets will be around 10 hours per piece. The 3 mass detector folds will take around 8 hours in total

#### Predicting neural nets
Once trained or downloaded through the url (https://retinopaty.blob.core.windows.net/ndsb3/trained_models.rar) the models are placed in the ./models/ directory.
From there the nodule detector *step3_predict_nodules.py*  can be run to detect nodules in a 3d grid per patient. The detected nodules and predicted malignancy are stored per patient in a separate directory. 
The masses detector is already run through the *step2_train_mass_segmenter.py* and will stored a csv with estimated masses per patient.

#### Training of submissions, combining submissions for final  submission.
Based on the per-patient csv’s the masses.csv and other metadata we will train an xgboost model to generate submissions (*step4_train_submissions.py*). There are 3 levels of submissions. First the per-model submissions. (level1). Different models are combined in level2, and Daniel’s submissions are added. These level 2 submissions will be combined (averaged) into one final submission.
Below are the different models that will be generated/combined.

- Level 1:<br>
Luna16_fs (trained on full luna16 set)<br>
Luna16_ndsbposneg v1 (trained on luna16 + manual pos/neg labels in ndsb)<br>
Luna16_ndsbposneg v2 (trained on luna16 + manual pos/neg labels in ndsb)<br>
Daniel model 1<br>
Daniel model 2<br>
posneg, daniel will be averaged into one level 2 model<br>

- Level 2.<br>
Luna16_fs<br>
Luna16_ndsbposneg<br>
Daniel<br><br>

These 3 models will be averaged into 1 *final_submission.csv*

#### Bugs and suggestions.
First of all. Duringing cleanup I noticed that I missed 10% of the LUNA16 patients because I overlooked subset0. That might be a 100.000 dollar mistake. For reprodicibility reasons I kept the bug in. In settings.py you can adjust the code to also take this subset into account.

Suggestions for improvement would be:
- Take the 10% extra LUNA16 condidates.
- Use different blends of the positive and negative labels
- Other neural network architectures.
- Etc..











