# Fusion of Vision and Thermal Imaging for Object Tracking - BSc Thesis

This repository contains the programs used in my Bachelor of Science thesis:  
**"Object Tracking Utilizing Fusion of Vision and Thermal Imaging Implemented in an SoC FPGA Using the Vitis AI Tool."**

## Abstract
Vision systems are an integral part of our world, and we use them almost every day in various situations. This paper presents a vision system combining images from a visual and a thermal camera. The calibration process of the two cameras is carried out using the OpenCV library. The system, implemented in Python, performs object detection using the YOLOv3 detector and tracks the objects with the SORT tracking algorithm. The YOLO convolutional neural network model was trained and evaluated on the PST900 image dataset. The Vitis AI 2.0 tool was utilized to implement the detector on the SoC FPGA platform. The vision system was tested on an eGPU (Nvidia Jetson Xavier NX) and SoC FPGA (Xilinx Kria KV260).

## Repository Structure
```plaintext
.
├── Calibration
│   ├── C++ app on Jetson
│   └── ImageRegistration
├── Detection
├── Tracking
└── VAI Implementation
```
Each folder contains files used during a different stage of the project:<br>
In folder *Calibration* there are files used for calibration of vision and thermal cameras and GUI RT app source files designed for Jetson.<br>
In folder *Detection* there are files containing model of YOLOv3 detector, its training program and quantization process.<br>
In folder *Tracking* there is an implementation of SORT tracking algortihm.<br>
In the last folder *VAI Implementation* there are jupyter notebooks for Vitis AI and PYNQ supporting boards.<br>

## Sources
For development of these programs provided below repositories have been copied and modified.
 - https://github.com/ermuur/PyTorch-YOLOv3 used in *Detection*
 - https://github.com/RunqiuBao/fov_alignment/blob/main/fov_align.ipynb used in *Calibration/ImageRegistration*
 - https://github.com/realizator/stereopi-fisheye-robot used in *Calibration/ImageRegistration*
 - https://github.com/groupgets/LeptonModule/tree/master used in *Calibration/C++ app on Jetson*
 - https://github.com/abewley/sort/tree/master used in *Tracking*

 ## Note
For reconstructing programs downloading the PST900 dataset is required. Replace empty folders *data\PST900_RGBT_Dataset* with downloaded images. <br> 
 - https://drive.google.com/file/d/1hZeM-MvdUC_Btyok7mdF00RV-InbAadm/view
 - https://github.com/ShreyasSkandanS/pst900_thermal_rgb?tab=readme-ov-file
