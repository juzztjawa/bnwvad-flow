This work is inspired from batch-norm wealy supervised anomaly detection paper but instead of using only RGB features, Flow features are also used.

XD-violence dataset achieved a value of 84.92%, very similar to using only RGB features, but by incorporating flow features, the training process is more stable.

Download the XD-violence I3D features from [this link](https://roc-ng.github.io/XD-Violence/)
Download the UCF-Crime I3D features from [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc )

Note that UCF-crime don't have flow features. You may download a I3D flow model or use the model in feature_extract/ to extract the flow features.

Note that the feature extraction code inside Feature_extract/ is only for extracting flow features.In order to extract the features, keep the videos in a folder and execute:

python video2flow2i3d.py --src_dir ../../UCF_actual/Anomaly-Videos/Burglary/ --output_dir ./your_directory/Burglary/

In order to train the model and test the model,execute

python main.py --root_dir ./XD_violence/i3d-features/

