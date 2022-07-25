# Streetview segmentation
Semantic image segmentation with Facebook's Mask2Former models.

## Description
This script performs semantic segmentation on images, using models from Facebook's collection of Mask2Former 
models. Output consists of a CSV-file detailing, for each input image, the number of pixels in each semantic
class in that image. Optionally, it can save a copy of each input image overlaid with the semantic segmentation.
If the input images are 360° photo's, the script provides the possibility of tranforming them by projecting them
onto a cube, resulting in six images per input image.

This script is specifically designed to run on a computer without a GPU. Some of the underlying libraries
require the presence of CUDA-drivers to run, even if the actual device is absent. As it can be problematic to
install such drivers on a computer without an actual GPU, the program is packaged as Docker-container, based on
an official NVIDIA-image, which comes with pre-installed drivers. Building and running the container requires
the presence of [Docker enginge](https://docs.docker.com/engine/install/).

## Building the docker container
Check out this repository, and to build, run:
```bash
docker build -t my_tag:latest -f Dockerfile .
```

## Getting model files
Models have two parts: configuration files, and a file with weights. The configuration files are located in
the Mask2Former repository, which is included in the container at build time. Weights files are not included
in the container; they must be downloaded and made available to the script in the container.

After selecting the appropriate model configuration files and downloading the corresponding file with model weights,
the location of both needs to registered in a configuration file. For example:

```json
{
    "path_model_cfg" : "/configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml",
    "path_model_weights" : "/data/model/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_f07440.pkl"
}
```
See below for more detail on the paths. Save this to a configuration file, which will be passed to the script using
the `--config` parameter.

Please note that the software has been tested with a limited number of models, specifically one from the Mapillary Vistas
Panoptic Segmentation collection (model_id 49189528_0), one from the COCO Panoptic Segmentation collection (47429163_0),
and one from the Cityscapes set (48318254_2). Generally, the Swin-L based models seem to work properly.

### Model configuration
Go the [Mask2Model Model Zoo](https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md) and pick
a model to use. Find the path of the model's configuration file by clicking the 'Mask2Former' link for the appropriate
model in the Model Zoo-table. This leads to the corresponding configuration file in the Mask2Former repository.
Take the path of that file relative to the repository's root, as it would be when the repository were checked out.

Example: the first model from the COCO Model Zoo, Panoptic Segmentation-table (model_id 47430278_4) links to [its configuration
in the Mask2Former repo](https://github.com/facebookresearch/Mask2Former/blob/main/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml),
the relative path of which, when checked out, would be:

`Mask2Former/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml`

This would be entered into the config file as:
```json
"path_model_cfg" : "/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
```

#### External configuration
If the configuration of a specific model is not present in the container (for example because it was added to the Model Zoo
after the instance of the container was built), it can be made available to the script from a location outside of the container,
much in the same way as a weights file (see next paragraph). Download the configuration file or files, make them available to
the container via a volume mapping, and update the value for `path_model_cfg` accordingly. The script will first look for the
file in its Mask2Former repository, and, if it isn't found there, subsequently in the absolute location specified in `path_model_cfg`.
For example, if `path_model_cfg` is set to

`/data/model/config/my_maskformer2.yaml`

the script will first look for

`../Mask2Former/data/model/config/my_maskformer2.yaml`

and failing that, look for

`/data/model/config/my_maskformer2.yaml`

which is a location that can be mapped to a folder on the host-machine. When using a external configuration, make sure to include *all*
yaml-files required; model configurations are usually made up of several files, chained by \_BASE\_ statements.


### Weights-file
A file with model weights needs to be downloaded and made available in the container. Click the appropriate 'model'-link in the
Model Zoo-table to download the pickle-file containing the model. In the example above, this is `model_final_94dc52.pkl`.

Place the file in a folder that will be mapped into the container, and edit the config file accordingly:

```json
"path_model_weights" : "/data/model/model_final_94dc52.pkl",
```
Note that this should be the path as it is 'seen' from inside the container.


## Running a job
Command to run a job:
```bash
docker run -v /local/path/to/data:/data --rm -it my_tag:latest \
	--config "/data/model/config.json" \
	--non-recursive \
	--input "/data/images" \
	--transform360 \
	--transform360exclude "5" \
	--save-segmentation-images \
	--suppress-warnings
```
The parameter `-v` maps a host directory to one inside the container, allowing the container access to files on the host computer. Mappings have the form of `<path on host>:<path in container>`; it's advised to always use '/data' for the second part.

Parameters are as follows:

+ **config**: path to a JSON-file containing the model paths (mandatory; [example file](code/config.json.example))
+ **input**: folder with images (mandatory). By default, the folder is read recursively.
+ **non-recursive**: switch to force input folder to be read non-recursively (default: False).
+ **transform360**: order the program to tranform the input photo's from 360° to six cube projections (default: False). If images should be processed _as is_, skip this flag. Transformed images are written to a subdirectory for each 360° image.
+ **transform360exclude**: comma-separated list of sides to exclude from the transformation. sides: 0 = left most, 1 = middle left, 2 = middle right, 3 = right most, 4 = top, 5 = bottom. For instance, `--transform360exclude "4,5"` outputs four projected images, omitting the cube's top and bottom.
+ **save-segmentation-images**: whether the program has to save a copy of each image with the segmentation as overlay (default: False). These images are written to a subfolder called 'segmentations' amd can be useful for checking results at a glance. Numerical output is always written to a CSV-file in the input image folder.
+ **suppress-warnings**: suppresses some of the warnings generated by the Mask2Former software (default: False).

## Used libraries 
+ [Mask2Former](https://github.com/facebookresearch/Mask2Former)
+ [detectron2](https://github.com/facebookresearch/detectron2)
+ [three60cube](https://pypi.org/project/three60cube/)

## A note on package versions
Programs that work with CUDA can be sensitive to changes in versions of used packages. Even a change in the minor version of a package can sometimes cause serious problems. To avoid such problems, versions have been explicitly pinned to versions that have proved to work well together. However, after cloning the Mak2Former repository, [its requirements are installed](/UtrechtUniversity/streetview-segmentation/blob/main/Dockerfile#L39), none of which are pinned ([see requirements.txt](https://github.com/facebookresearch/Mask2Former/blob/main/requirements.txt)). If this causes problems in the future, try uninstalling the packages and reinstalling them to the pinned versions listed below.
+ cython==0.29.30
+ scipy==1.8.1
+ shapely==1.8.2
+ timm==0.6.5
+ h5py==3.7.0
+ submitit==1.4.4
+ scikit-image==0.19.3
