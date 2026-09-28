# Dependencies and prior work

The MIT license covers this repository's project code. Datasets, pretrained
weights, and dependencies retain their own licenses and terms.

- PyTorch / torchvision: https://github.com/pytorch/pytorch and https://github.com/pytorch/vision
- timm (backbones and pretrained weights): https://github.com/huggingface/pytorch-image-models
- Albumentations: https://github.com/albumentations-team/albumentations
- OpenCV: https://github.com/opencv/opencv

The comparison modules are the research project's implementations/adaptations,
not claimed to be the original authors' official repositories. In particular,
the U-Net module uses a pretrained backbone, the CrackFormer module uses a Swin
backbone with the local decoder, and the SegNeXt module uses the local MSCAN
implementation. Consult the model docstrings/configurations before comparing
against results from other implementations.

Related methods include U-Net, ResNet/FPN, DeepLabV3+, CrackFormer, Swin-UNet,
SegFormer, SegNeXt, ConvNeXt, and UPerNet. OGDNet uses torchvision's deformable
convolution operator (Dai et al., ICCV 2017).

No dataset images, third-party paper PDFs, or pretrained model weights are
redistributed in this code repository.
