OPM-MVS

Octagram Propagation Matching for Multi-Scale View Stereopsis

This repository contains the official implementation of:

Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)

The paper was published in IEEE Access, Volume 13, 2025.

News

The source code for OPM-MVS has been released.

About

OPM-MVS is a PatchMatch-based multi-view stereo method that uses octagram propagation matching, multi-scale processing, structured region information, and confidence-guided geometric consistency to improve dense 3D reconstruction, particularly in textureless and geometrically challenging regions.

When using this implementation in your research, please cite the following paper:

@ARTICLE{11003953,
  author={Khuboni, Ray L. and Xu, Hongjun},
  journal={IEEE Access},
  title={Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)},
  year={2025},
  volume={13},
  number={},
  pages={86203-86217},
  keywords={Accuracy;Three-dimensional displays;Image reconstruction;
            Depth measurement;Costs;Cameras;Surface texture;Reliability;
            Surface reconstruction;Pipelines;Checkerboard propagation;
            Multi-scale PatchMatch;Multi-view stereo;
            Structured region information;
            Confidence-guided geometric consistency;
            Textureless regions},
  doi={10.1109/ACCESS.2025.3569913}
}

Paper:

Octagram Propagation Matching for Multi-Scale View Stereopsis

DOI:

10.1109/ACCESS.2025.3569913

System Requirements

The implementation has been tested using:

Windows 11

Microsoft Visual Studio 2017

NVIDIA GeForce RTX 2060

CUDA 11.x

OpenCV 3.3 or later

CMake 3.18 or later

Dependencies

CUDA Toolkit version 11.0 or later

OpenCV version 3.3 or later

CMake version 3.18 or later

Microsoft Visual Studio 2017 with C++ development tools

For Visual Studio 2017, CUDA 11.8 is recommended.

Project Structure

The main source files are:

OPM-MVS/
├── CMakeLists.txt
├── main.cpp
├── main.h
├── OCMM.cpp
├── OCMM.cu
├── OCMM.h
└── README.md

Compilation

1. Clone the repository

git clone https://github.com/RayKhuboni/OPM-MVS.git
cd OPM-MVS

2. Create a build directory

Using Command Prompt:

mkdir build

3. Configure the project

Run the following command from the project root directory:

cmake -S . -B build ^
  -G "Visual Studio 15 2017" ^
  -A x64 ^
  -DOpenCV_DIR="C:/opencv/build/x64/vc15/lib"

Replace the value of OpenCV_DIR with the directory containing:

OpenCVConfig.cmake

The exact path depends on the OpenCV installation.

4. Build the project

Compile the Release configuration:

cmake --build build --config Release

Alternatively, open the generated Visual Studio solution:

build/OPM_MVS.sln

Select:

Release | x64

Then build the solution in Visual Studio.

The executable should be generated in:

build/Release/OPM-MVS.exe

Input Preparation

OPM-MVS expects an ACMM/MVSNet-style input structure.

Use the colmap2mvsnet_acm.py script to convert a COLMAP sparse reconstruction into the required input format.

A typical dataset directory should contain:

data_folder/
├── images/
│   ├── 00000000.jpg
│   ├── 00000001.jpg
│   └── ...
├── cams/
│   ├── 00000000_cam.txt
│   ├── 00000001_cam.txt
│   └── ...
└── pair.txt

The input components are:

images/: Input images

cams/: Camera intrinsic parameters, extrinsic parameters, and depth ranges

pair.txt: Reference and source-view relationships

Usage

Run OPM-MVS from Command Prompt as follows:

build\Release\OPM-MVS.exe path\to\data_folder

Example:

build\Release\OPM-MVS.exe D:\datasets\ETH3D\courtyard

The program performs the following stages:

Initial photometric PatchMatch estimation

Geometric consistency refinement

Multi-scale joint bilateral upsampling

Hierarchical PatchMatch refinement

Multi-view depth-map fusion

Output

Reconstruction results are written to:

data_folder/OPM-MVS/

Per-view results are stored in directories such as:

OPM-MVS/
├── 2333_00000000/
│   ├── depths.dmb
│   ├── depths.png
│   ├── depths_geom.dmb
│   ├── depths_geom.png
│   ├── normals.dmb
│   ├── normals.png
│   ├── costs.dmb
│   └── costs.png
└── OPM-MVS_model.ply

The final fused point cloud is saved as:

OPM-MVS/OPM-MVS_model.ply

The point cloud can be viewed using software such as:

CloudCompare

MeshLab

Open3D

COLMAP

ETH3D Results

OPM-MVS results on the high-resolution ETH3D training dataset using the 2 cm evaluation threshold are available on the ETH3D benchmark:

ETH3D OPM-MVS Results

Mean

Courtyard

Delivery Area

Electro

Facade

Kicker

Meadow

Office

Pipes

Playground

Relief

Relief 2

Terrace

Terrains

79.68

87.99

86.77

88.52

72.72

64.55

67.65

67.87

74.33

74.31

86.37

85.25

89.73

89.81

Tanks and Temples Results

OPM-MVS was also evaluated on the Tanks and Temples dataset.

Tanks and Temples Benchmark

Troubleshooting

CUDA compiler not detected

Confirm that CUDA is installed and that nvcc is available:

nvcc --version

Run CMake from a Visual Studio 2017 x64 Native Tools Command Prompt.

OpenCV cannot be found

Specify the directory containing OpenCVConfig.cmake:

-DOpenCV_DIR="C:/opencv/build/x64/vc15/lib"

CUDA architecture errors

The NVIDIA GeForce RTX 2060 uses compute capability 7.5. The CMake configuration should therefore contain:

set(CMAKE_CUDA_ARCHITECTURES 75)

Missing OpenCV DLL files

Add the OpenCV binary directory to the Windows PATH, or copy the required OpenCV DLL files into the same directory as OPM-MVS.exe.

For example:

C:\opencv\build\x64\vc15\bin

CUDA kernel timeout

Long-running CUDA kernels may trigger the Windows Timeout Detection and Recovery mechanism when the RTX 2060 is also being used as the display GPU.

Reducing the input image size or using the GPU as a dedicated compute device may help during testing.

Acknowledgements

This implementation benefited substantially from the following open-source projects:

ACMM, developed by Qingshan Xu and collaborators

Gipuma

COLMAP

We thank the authors of these projects for making their source code publicly available.

Citation

Please cite OPM-MVS when using this repository, its implementation, or its results in academic work:

@ARTICLE{11003953,
  author={Khuboni, Ray L. and Xu, Hongjun},
  journal={IEEE Access},
  title={Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)},
  year={2025},
  volume={13},
  pages={86203-86217},
  doi={10.1109/ACCESS.2025.3569913}
}# Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)
The official implementation of 'Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)' is released here. This paper has been accepted for publication in IEEE Access 2025.

# OPM-MVS
[News] The code for [OPM-MVS](https://github.com/RayKhuboni/OPM-MVS) is released!!!  

## About
[OPM-MVS](https://ieeexplore.ieee.org/abstract/document/11003953) is an Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS). If you find this project useful for your research, please cite:  
```
@ARTICLE{11003953,
  author={Khuboni, Ray L. and Xu, Hongjun},
  journal={IEEE Access}, 
  title={Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)}, 
  year={2025},
  volume={13},
  number={},
  pages={86203-86217},
  keywords={Accuracy;Three-dimensional displays;Image reconstruction;Depth measurement;Costs;Cameras;Surface texture;Reliability;Surface reconstruction;Pipelines;Checkerboard propagation;multi-scale patchmatch;multi-view stereo;structured region information;confidence-guided geometric consistency;texture-less regions},
  doi={10.1109/ACCESS.2025.3569913}}
```
## Dependencies
The code has been tested on Windows 11 on Visual Studio 2017 with RTX2060.  
* [Cuda](https://developer.nvidia.com/zh-cn/cuda-downloads) >= 11.0
* [OpenCV](https://opencv.org/) >= 3.3
* [cmake](https://cmake.org/)
## Usage
* Compile OPM-MVS
```  
cmake.  
make
```
* Test 
``` 
Use script colmap2mvsnet_acm.py to convert COLMAP SfM result to ACMM input   
Run ./OPM-MVS $data_folder to get reconstruction results
```
## Results on high-res ETH3D training dataset [2cm]
Our OPM-MVS results are published on [ETH3D dataset] (https://www.eth3d.net/result_details?id=1131)
| Mean   | courtyard | delivery_area | electro | facade | kicker | meadow | office | pipes  | playgroud | relief | relief_2 | terrace | terrains |
|--------|-----------|---------------|---------|--------|--------|--------|--------|--------|-----------|--------|----------|---------|----------|
| 79.68  | 87.99     | 86.77         | 88.52   | 72.72  | 64.55  | 67.65  | 67.87  | 74.33  | 74.31     | 86.37  | 85.25    | 89.73   | 89.81    |
## Results on Tanks and Temples Dataset
Our OPM-MVS results are published on [Tanks and Temples dataset](https://www.tanksandtemples.org/)
## Acknowledgements
This code largely benefits from the following repositories: Special thanks to Qingshan Xu [ACMM](https://github.com/GhiXu/ACMM), [Gipuma](https://github.com/kysucix/gipuma), and [COLMAP](https://colmap.github.io/). Thanks to their authors for making the source code of their excellent works available.
