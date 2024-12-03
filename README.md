# Participant ID Decoding Code

## Dataset
Firstly, the dataset used and the original files are introduced. The dataset is located in the '../eremus_dataset' directory, and can be downloaded directly from the following link: https://zenodo.org/records/14028845?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4MzMxNTI3LTljMjMtNDQxNi1iMTc4LWU4ODAxZTEyMTZiZSIsImRhdGEiOnt9LCJyYW5kb20iOiI4OTExYWM4ZTdhNjUyNjBlNTRjNjliMTgxZmIxZmYwZCJ9.oiKqLBl7GPxNDXautcVhcR7VbuTSJqaS5rnubPrZX8KV6n9_Vs-BIi8x70erGQAlaxCTBvTNtgrx_4qHBolXEQ

The dataset includes two files: raw and puned. The file that provides the labels is splits_subject_identification.json. In the current version of the code, we use the puned data.

## Code
### fif2npy.py
This script converts the original fif files to npy files for easier reading, and the converted dataset is saved in the current directory.

### save_list.py
We do not preprocess the data, but preprocess the splits file to facilitate direct reading in the dataset. This processing generates two files: all_train.npy and all_label.npy.

### main.py
This is the main program, which includes training, validation, testing (train_and_test), and inference (inference_result).
#### Function: split_train_valid
This function splits two trials of each participant as a validation set. To ensure fairness, we use five-fold cross-validation, where one fold is used for validation and one fold is used for testing. The remaining data is used for training.
#### Function: train_and_test
This function calls train_valid_model and test_model to perform training, validation, and testing. Training and validation are performed on 1s data, while testing is performed on the complete data.
#### Function: inference_result
This function calls inference to perform the final inference.

### dataset.py
This script includes AADataset and AADataset_test.
#### AADataset
AADataset splits the long EEG data into 1s segments (128 sampling points), and normalizes the EEG data along the channel dimension in init.
#### AADataset_test
AADataset_test does not perform segmentation and reads the entire data. However, the long data is segmented in test_model and inference for voting purposes. Similarly, the EEG data is normalized along the channel dimension in init.

### end_to_end.py
This script includes train_valid_model and test_model.
#### Function: train_valid_model
This is a classic neural network building process, where each epoch prints the training and validation results.
#### Function: test_model
This is a classic testing process. One point to note is the segmentation of the EEG data. The long data is segmented into 128 sampling point segments, and voting is performed by taking the mode.

### inference.py
There is a concern that the order of the data may be disrupted in this script. The EEG data is normalized, and the same content as in test_model is performed.

### DenseNet.py
The DenseNet architecture is used, as described in https://ieeexplore.ieee.org/document/10448013.