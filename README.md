# Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS) 
The official implementation of 'Octagram Propagation Matching for Multi-Scale View Stereopsis (OPM-MVS)' is released here. The paper was published in IEEE Access, Volume 13, 2025.

# OPM-MVS
[News] The code for [OPM-MVS](https://github.com/RayKhuboni/OPM-MVS) is released!!!  

## About
OPM-MVS is a PatchMatch-based multi-view stereo method that uses octagram propagation matching, multi-scale processing, structured region information, and confidence-guided geometric consistency to improve dense 3D reconstruction, particularly in textureless and geometrically challenging regions. 

When using this implementation in your research, please cite the following paper:
[OPM-MVS](https://ieeexplore.ieee.org/abstract/document/11003953)  
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
cmake  
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
This code largely benefits from the following repositories: Special thanks to Qingshan Xu [ACMM](https://github.com/GhiXu/ACMM), [Gipuma](https://github.com/kysucix/gipuma) and [COLMAP](https://colmap.github.io/). Thanks to their authors for opening source of their excellent works.
