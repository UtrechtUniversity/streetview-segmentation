"""
Program performs semantic segmentation on images, using a deep learning model
from Facebook's detectron2/Mask2Former model zoo. It is designed to run on a
computer without a GPU.

July 2022, Maarten Schermer / Utrecht University

https://github.com/facebookresearch/detectron2/
https://detectron2.readthedocs.io/en/latest/tutorials/install.html
https://github.com/facebookresearch/Mask2Former/blob/main/MODEL_ZOO.md
https://github.com/UtrechtUniversity/three60cube

"""
import warnings
import sys
import os
import csv
import re
import argparse
import datetime
import json
import glob
import cv2
from three60cube import Three60Cube
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config
import utils

LOCAL_MODEL_BASE_PATH = "/model"
LOCAL_CODE_BASE_PATH = "/code"
MASK2FORMER_REPO = "Mask2Former"

sys.path.append(LOCAL_CODE_BASE_PATH + '/' + MASK2FORMER_REPO)
from mask2former import add_maskformer2_config


class ImageSegmentation:
    """
    Semantic image segmentation using Facebook's Mask2Former models.
    """
    model_config_file = "config.json"
    valid_extensions = [".jpg",".png"]

    def __init__(self):
        self.args = None
        self.image_dir = None
        self.input_image = None
        self.output_folder = None
        self.recursive = True
        self.do_transform360 = None
        self.transform360exclude = []
        self.save_segmentation_images = None
        self.model_base_path = None
        self.code_base_path = None
        self.mask2former_repo = None
        self.transformer = None
        self.predictor = None
        self.model_metadata = None
        self.path_model_cfg = None
        self.path_model_weights = None
        self.model_metadata_catalog = None
        self.csv_filename = None
        self.images = []
        self.logger = utils.get_logger('image segmentation')
        self.now = datetime.datetime.now()

    def main(self):
        """
        main program.
        """
        self.set_base_paths()
        self.parse_arguments()
        self.read_configuration()
        self.initialize()
        self.acquire_images()
        self.transform360()
        self.load_model()
        self.run_predictions()
        self.finalize()

    def set_base_paths(self):
        """
        set_base_paths() sets base paths from constants
        LOCAL_MODEL_BASE_PATH & LOCAL_CODE_BASE_PATH are currently fixed
        but could at some point be made to be set by the user.
        """
        self.code_base_path = "/" + LOCAL_CODE_BASE_PATH.lstrip("/")
        self.model_base_path = "/" + LOCAL_MODEL_BASE_PATH.lstrip("/")
        self.mask2former_repo = MASK2FORMER_REPO

    def comma_list(self,string):
        """
        turns a string with comma's as separator into a list.
        """
        return list(map(lambda a: a.strip(),string.split(",")))

    def parse_arguments(self):
        """
        parse_arguments() parses command line arguments, assigning them to self.vars.
        """
        parser = argparse.ArgumentParser(description="Semantic segmentation of images")
        parser.add_argument("--input", type=str, required=True,
                            help="path to input folder, e.g. './input/', " +
                            "or single image, e.g. './input/photo.jpg'")
        parser.add_argument("--output", type=str, required=True,
                            help="path to output folder")
        parser.add_argument("--non-recursive", action='store_true',
                            help="read the input folder non-recursively")
        parser.add_argument("--config", type=str, required=True,
                            help="path to config file containing model paths")
        parser.add_argument("--transform360", action='store_true',
                            help="input is 360° photo, transform to cubic projections")
        parser.add_argument("--transform360exclude", type=self.comma_list,
                            help="comma separated list of cubic projections to exclude (0-5). " +
                            "0=left most image ... 3=right most, 4=top, 5=bottom")
        parser.add_argument("--save-segmentation-images", action='store_true',
                            help="save image with segmantation overlay")
        parser.add_argument("--suppress-warnings", action='store_true',
                            help="suppress warnings (some)")

        self.args = vars(parser.parse_args())

    def read_configuration(self):
        """
        read_configuration() reads configuration from file: paths to the model config
        and to the model weights file.
        """
        config_file = self.args["config"]

        if os.path.exists(config_file):

            with open(config_file, 'r', encoding='utf-8') as file:
                config = json.load(file)

            if not "path_model_cfg" in config:
                self.logger.error("'path_model_cfg' missing from config file '%s'; exiting",
                                  config_file)
                sys.exit()

            if not "path_model_weights" in config:
                self.logger.error("'path_model_weights' missing from config file '%s'; exiting'",
                                  config_file)
                sys.exit()

            config = {k: v.strip() for k,v in config.items()}

            # loading config
            external_cfg = config["path_model_cfg"]
            internal_cfg = os.path.join(self.code_base_path, self.mask2former_repo,
                                        config["path_model_cfg"].lstrip("/"))

            if os.path.exists(external_cfg):
                self.path_model_cfg = external_cfg
            elif os.path.exists(internal_cfg):
                self.path_model_cfg = internal_cfg
            else:
                self.logger.error("found no valid model config matching '%s'; exiting",
                                  config["path_model_cfg"])
                sys.exit()

            self.logger.info("using model config '%s'", self.path_model_cfg)

            # loading weights
            if os.path.exists(config["path_model_weights"]):
                self.path_model_weights = config["path_model_weights"]
            else:
                self.logger.error("found no valid model at '%s'; exiting",
                                  config["path_model_weights"])
                sys.exit()

            self.logger.info("using model weights '%s'", self.path_model_weights)

        else:
            self.logger.error("config file '%s' not found; exiting", config_file)
            sys.exit()

    def initialize(self):
        """
        initialize() performs some preliminary checks and actions.
        """
        if not os.path.exists(self.path_model_weights):
            self.logger.error("model '%s' doesn't exist: exiting", self.path_model_weights)
            sys.exit()

        if not os.path.exists(self.path_model_cfg):
            self.logger.error("model config '%s' doesn't exist: exiting", self.path_model_cfg)
            sys.exit()

        if not os.path.exists(self.args["input"]):
            self.logger.error("input path '%s' does not exist (is the container's volume mapping"
                              + " correct?); exiting", self.args['input'])
            sys.exit()

        self.recursive = not self.args["non_recursive"]
        self.save_segmentation_images = self.args["save_segmentation_images"]
        self.do_transform360 = self.args["transform360"]
        self.csv_filename = f"results-{self.now.strftime('%Y-%m-%dT%H%M')}" \
                            + "-({model_metadata_catalog}).csv"

        if os.path.isfile(self.args["input"]):
            self.input_image = self.args["input"]
            self.logger.info("input image: %s",self.input_image)
        else:
            self.image_dir = self.args["input"].rstrip("/")
            self.logger.info("input directory: %s", self.image_dir)

        if not self.input_image and not self.image_dir:
            self.logger.error("no image path specified; exiting")
            sys.exit()

        if not self.args["output"] is None:
            self.output_folder = self.args["output"]
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                utils.chmod_recursively(self.output_folder)
        else:
            self.output_folder = os.path.dirname(self.input_image) if self.input_image \
                                 else self.image_dir

        self.logger.info("output directory: %s", self.output_folder)

        self.csv_filename = os.path.join(self.output_folder,self.csv_filename)

        if not self.transform360 and len(self.transform360exclude)>0:
            self.logger.warning("ignoring --transform360exclude (--transform360 is absent)")

        if self.do_transform360:
            try:
                self.transform360exclude = list(map(lambda a: int(a.strip()),
                                                self.args["transform360exclude"]))
            except Exception:
                self.transform360exclude = []
                self.logger.warning("transform360exclude reset (list items should be integers)")

            self.transformer = Three60Cube()

        if self.args["suppress_warnings"]:
            warnings.filterwarnings("ignore")

        self.logger.info("valid extensions: %s", "; ".join(self.valid_extensions))

    def acquire_images(self):
        """
        acquire_images() reads images from the input folder.
        """
        self.images = []

        if not self.input_image is None \
        and os.path.splitext(self.input_image)[1].lower() in self.valid_extensions:
            self.images.append(self.input_image)
            self.image_dir = os.path.dirname(self.input_image)
        else:
            for filename in glob.iglob(self.image_dir + '**/**', recursive=self.recursive):
                if os.path.isfile(filename) \
                and os.path.splitext(filename)[1].lower() in self.valid_extensions:
                    self.images.append(filename)

        if len(self.images)==0:
            self.logger.info("found no images; exiting")
            sys.exit()
        else:
            self.logger.info("found %s image(s)", len(self.images))
            self.images.sort()

    def transform360(self):
        """
        transform360() makes a subfolder for each photo to be transformed from a 360°
        photos to cube projections and calls the actual transformer
        """
        if self.do_transform360:
            self.logger.info("transforming 360° photos to cube projections")

            if len(self.transform360exclude)!=0:
                self.logger.info("skipping cube projection(s): " +
                                 ", ".join([str(x) for x in self.transform360exclude]))

            new_images = []
            for image_url in self.images:
                output_folder = os.path.join(
                    os.path.dirname(image_url),
                    (re.sub(r'[^a-zA-Z\d\s:]','_',re.sub(r'(^[^a-zA-Z\d\s:]*|\.(.*)$)','',
                            os.path.basename(image_url)))) + "/"
                )

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                new_paths = self._transform360(image_url,output_folder)
                new_images += new_paths
                utils.chmod_recursively(output_folder)
                self.logger.info("%s --> %s (%s)", image_url, output_folder, len(new_paths))

            self.images = new_images
            self.images.sort()
            self.logger.info("created %s images", len(self.images))
        else:
            self.logger.info("skipping transformation to cube projection")

    def _transform360(self,input_file,output_folder):
        """
        _transform360() does the actual transformation of a 360° photos to 6 cubic panes.
        """
        self.transformer.open_image(input_file)
        self.transformer.get_pane(pane=1, dim=512)
        self.transformer.save_cache("templ.pickle")
        self.transformer = Three60Cube("templ.pickle")
        self.transformer.open_image(input_file)

        new_paths = []

        for i in range(6):
            if i not in self.transform360exclude:
                name_bits = os.path.splitext(os.path.basename(input_file))
                new_path = os.path.join(output_folder,f'{name_bits[0]}_{i}{name_bits[1]}')
                if not os.path.exists(new_path):
                    self.transformer.save_pane(new_path, pane=i, dim=512)
                else:
                    self.logger.info("skipping pane %s ('%s' already exists).", i, new_path) 
                new_paths.append(new_path)

        return new_paths

    def load_model(self):
        """
        load_model() creates a configuration, and load the model. explicitly forces CPU as device.
        """
        cfg = get_cfg()
        cfg.MODEL.DEVICE = "cpu"
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)

        # load model
        cfg.merge_from_file(self.path_model_cfg)
        cfg.MODEL.WEIGHTS = self.path_model_weights
        cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
        # not doing recognition of instance or panoptic
        # cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
        # cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

        self.predictor = DefaultPredictor(cfg)

        # find the metadata catalog used, based on the test-dataset
        if len(cfg.get("DATASETS").get("TEST"))==0:
            self.logger.error("found no datasets in model config; exiting")
            sys.exit()
        elif len(cfg.get("DATASETS").get("TEST")) > 1:
            self.logger.info('found %s" datasets in model config; defaulting to the first',
                             len(cfg.get("DATASETS").get("TEST")))

        # model_metadata contains class labels etc.
        self.model_metadata_catalog = cfg.get("DATASETS").get("TEST")[0]
        self.model_metadata = MetadataCatalog.get(self.model_metadata_catalog)
        self.logger.info("using metadata catalog '%s'", self.model_metadata_catalog)

    def run_predictions(self):
        """
        run_predictions() loops through all images, and has the model perform predictions,
        outputting the results as CSV and optionally as images with overlaid semantic
        segmentations.
        """
        self.logger.info("running predictions on %s image(s)", len(self.images))

        calc = utils.SegmentationAreasCalculator()
        calc.set_model_metadata(self.model_metadata)

        self.csv_filename = self.csv_filename.format(model_metadata_catalog=
                                                     self.model_metadata_catalog)

        with open(self.csv_filename, 'w', encoding='utf-8') as file:

            csv_header = ["image","total pixels"]
            for idx, item in enumerate(calc.get_model_classes()):
                csv_header.append(f"{idx} ({item})")

            csv_writer = csv.writer(file)
            csv_writer.writerow(csv_header)

            for idx, image_url in enumerate(self.images):
                if not os.path.exists(image_url):
                    self.logger.error("image '%s' doesn't exist!? (skipping)", image_url)
                    continue

                self.logger.info("processing '%s' (%s/%s)", image_url[len(self.image_dir)+1:],
                                 (idx+1), len(self.images))

                # run predictions
                image = cv2.imread(image_url)
                semantic_result, outputs = self._run_prediction(image)

                calc.set_outputs(outputs)
                calc.set_areas()

                if self.do_transform360:
                    calc.set_areas_corrected()
                    classes = calc.get_classes_corrected()
                else:
                    classes = calc.get_classes()

                # write to CSV
                row = [image_url[len(self.image_dir)+1:], calc.get_total_area()]

                for jdx, item in enumerate(calc.get_model_classes()):
                    t_cls = [x for x in classes if x["key"]==jdx]
                    if len(t_cls)>0:
                        # row.append(t_cls[0][2])
                        row.append(t_cls[0]["pixels"])
                    else:
                        row.append(0)

                csv_writer.writerow(row)
                file.flush()

                # talk to user
                buff = []
                for key, t_cls in enumerate(classes):
                    buff.append(f'{t_cls["class"]}: {t_cls["percentage"]}%')
                    if key >= 2:
                        break

                self.logger.info("top results: %s; ...", "; ".join(buff))

                # save copy of image with segmentation overlay
                if self.save_segmentation_images:
                    relative_subfolder = os.path.dirname(image_url) \
                                         .replace(self.image_dir,'').strip('/')
                    seg_folder = os.path.join(self.output_folder,relative_subfolder,
                                              self.model_metadata_catalog)

                    if not os.path.exists(seg_folder):
                        os.makedirs(seg_folder)

                    name_bits = os.path.splitext(os.path.basename(image_url))
                    seg_file = os.path.join(seg_folder,
                                            f'{name_bits[0]}-segmentation{name_bits[1]}')

                    cv2.imwrite(seg_file, semantic_result)
                    utils.chmod_recursively(seg_file, 0o777)
                    self.logger.info("saved '%s'", seg_file[len(self.image_dir)+2:])

        # utils.chmod_recursively(self.csv_filename, 0o777)
        utils.chmod_recursively(self.output_folder, 0o777)

    def _run_prediction(self,image):
        """
        _run_prediction() runs the actual predictions.
        """
        outputs = self.predictor(image)
        viz = Visualizer(image[:, :, ::-1], self.model_metadata, scale=1,
                         instance_mode=ColorMode.IMAGE_BW)
        semantic_result = viz.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        return semantic_result, outputs

    def finalize(self):
        """
        some final messages to the user
        """
        self.logger.info("wrote results to '%s'", self.csv_filename)
        self.logger.info("done")


if __name__ == '__main__':
    seg = ImageSegmentation()
    seg.main()
