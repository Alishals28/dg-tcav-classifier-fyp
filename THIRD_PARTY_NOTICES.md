# MedicalNet attribution

`src/model.py` adapts the ResNet-18 backbone architecture from
https://github.com/Tencent/MedicalNet/blob/20f76aaab5cac8056eaf50b79ed97c09dbfbd3bd/models/resnet.py.
It replaces the segmentation decoder with pooling/classification and fixes type-A shortcut autograd detachment.
Checkpoint conversion is not supplied; the user obtains trusted weights from the upstream project.

Citation: Sihong Chen, Kai Ma, Yefeng Zheng. *Med3D: Transfer Learning for 3D Medical Image Analysis* (2019), arXiv:1904.00625.

Upstream license notice follows:

Tencent is pleased to support the open source community by making MedicalNet available.  

Copyright (C) 2019 THL A29 Limited, a Tencent company.  All rights reserved.

MedicalNet is licensed under the MIT License, including the third-party component listed below. 

A copy of the MIT License is included in this file.

Other dependency and license:


Open Source Software Licensed Under the MIT License:
--------------------------------------------------------------------
1. 3D-ResNets-PyTorch 3.0
Copyright (c) 2017 Kensho Hara


Terms of the MIT License:
---------------------------------------------------
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
