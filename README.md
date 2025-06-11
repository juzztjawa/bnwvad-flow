# Weakly Supervised Anomaly Detection with RGB and Flow Features

This work is inspired by the paper *"Batch Normalized Weakly Supervised Anomaly Detection"*. However, unlike the original, which uses only RGB features, this implementation incorporates both **RGB and Optical Flow** features to improve training stability.

## Dataset and Features

- **XD-Violence**  
  We achieved an **84.92%** score on the XD-Violence dataset, which is comparable to using RGB-only features. However, the inclusion of Flow features leads to a more **stable training process**.

  - 📥 Download the XD-Violence I3D features from [this link](https://roc-ng.github.io/XD-Violence/)

- **UCF-Crime**  
  Note: UCF-Crime does **not** provide Flow features by default.

  - 📥 Download the UCF-Crime I3D RGB features from [this link](https://stuxidianeducn-my.sharepoint.com/:f:/g/personal/pengwu_stu_xidian_edu_cn/EvYcZ5rQZClGs_no2g-B0jcB4ynsonVQIreHIojNnUmPyA?e=xNrGxc)

  To extract Flow features for UCF-Crime:
  - Either download a pretrained I3D Flow model
  - **Or** use the provided script in `feature_extract/` to extract Flow features from your video files.

## Feature Extraction

The `feature_extract/` directory contains a script to extract **Optical Flow** features from videos.

Make sure to organize your input videos in a folder, then run:

```bash
python video2flow2i3d.py --src_dir ../../UCF_actual/Anomaly-Videos/Burglary/ --output_dir ./your_directory/Burglary/
```

This script will compute the optical flow and extract I3D features for each video.


## Training and Testing
To train and test the model using the extracted features, run:

```bash
python main.py --root_dir ./XD_violence/i3d-features/
```