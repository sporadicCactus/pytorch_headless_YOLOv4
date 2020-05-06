# pytorch_headless_YOLOv4
PyTorch re-implementation of YOLOv4 architecture (CSPDarknet53+SPP+PANet) without prediction heads.

Usage:
```python
from pytorch_headless_YOLOv4 import _load
model = _load('$path to state dict file$')
```

Pretrained weights provided by [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), converted to .pth format: https://drive.google.com/open?id=1je_EaNMgkLK-h1P_8XspwDjwGOYdaomT

CUDA-optimized implementation of Mish (provided by [thomasbrandon/mish-cuda](https://github.com/thomasbrandon/mish-cuda)) is used if installed.


Source code: https://github.com/AlexeyAB/darknet

References:
* https://github.com/Tianxiaomo/pytorch-YOLOv4
* https://github.com/romulus0914/YOLOv4-PyTorch
* https://github.com/thomasbrandon/mish-cuda

```
@article{yolov4,
  title={YOLOv4: YOLOv4: Optimal Speed and Accuracy of Object Detection},
  author={Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao},
  journal = {arXiv},
  year={2020}
}
```
