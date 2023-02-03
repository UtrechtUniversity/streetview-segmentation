# Model configurations

This folder contains the configuration for four models from Facebook’s [Mask2Former Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md):

- `ade20k.config.json`: ADE20K Semantic Segmentation, model id: 48004474_0 (weights file: model_final_6b4a3a.pkl)
- `cityscapes.config.json`: Cityscapes Panoptic Segmentation, model id: 48318254_2 (weights file: model_final_064788.pkl)
- `coco_panoptic.config.json`: COCO Panoptic Segmentation, model id: 47429163_0 (weights file: model_final_f07440.pkl)
- `mapillary.config.json`: Mapillary Vistas Panoptic Segmentation, model id: 48267065_4 (weights file: model_final_132c71.pkl)

Each configuration file contains two paths, one to the YAML-file that contains the model configuration; and one to the file with the model weights. Note that both these paths should be as 'seen' from within the Docker container. 

The YAML-files are part of the Mask2Former repository, which is included in the Docker container in its entirety. The paths in the config files point to the fixed path of the files within the container, and the value of `path_model_cfg` should not be changed.

The weights files are not part of the container, and should be downloaded from the Model Zoo, and be made available to the container by mapping a volume from the host computer to a path in the container, and configuring the value of `path_model_weights` to match their location.