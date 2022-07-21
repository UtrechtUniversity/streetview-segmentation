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
import detectron2
import cv2
import torch
import sys, os, csv, re, argparse, datetime, json
import utils

from three60cube import Three60Cube

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.projects.deeplab import add_deeplab_config

LOCAL_MODEL_BASE_PATH = "/model"
LOCAL_CODE_BASE_PATH = "/code"
MASK2FORMER_REPO = "Mask2Former"

sys.path.append(LOCAL_CODE_BASE_PATH + '/' + MASK2FORMER_REPO)
from mask2former import add_maskformer2_config

class ImageSegmentation:

    logger = None
    now = None
    args = None

    image_dir = None
    tranform360 = None
    save_segmentation_images = None
    model_base_path = None
    code_base_path = None
    mask2former_repo = None

    transformer = None
    predictor = None
    model_metadata = None

    model_config_file = "config.json"
    path_model_cfg = None
    path_model_weights = None
    model_metadata_catalog = None

    valid_extensions = [".jpg",".png"]

    def __init__(self):
        self.logger = utils.get_logger('image segmentation')
        self.now = datetime.datetime.now()

    def main(self):
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
        set_base_paths() sets base paths from constants; LOCAL_MODEL_BASE_PATH & LOCAL_CODE_BASE_PATH
        are currently fixed but could at some point be made to be set by the user.
        """         
        self.code_base_path = "/" + LOCAL_CODE_BASE_PATH.lstrip("/")
        self.model_base_path = "/" + LOCAL_MODEL_BASE_PATH.lstrip("/")
        self.mask2former_repo = MASK2FORMER_REPO

    def parse_arguments(self):
        """
        parse_arguments() parses command line arguments, assigning them to self.vars.
        """         
        parser = argparse.ArgumentParser(description="Semantic segmentation of images")
        parser.add_argument("--input",type=str,required=True,help="path to input e.g. './input/'")
        parser.add_argument("--config",type=str,required=True,help="path to config file")
        parser.add_argument("--transform360", action='store_true')
        parser.add_argument("--save-segmentation-images", action='store_true')
        parser.add_argument("--suppress-warnings", action='store_true',help="suppress warnings (some)")
        
        self.args = vars(parser.parse_args())

    def read_configuration(self):
        """
        read_configuration() reads configuration from file: paths to the model config and to the model weights file.
        """
        config_file = self.args["config"]

        if os.path.exists(config_file):

            with open(config_file, 'r') as f:
                config = json.load(f)

            if not "path_model_cfg" in config:
                self.logger.error(f"'path_model_cfg' missing from config file '{config_file}'; exiting")
                exit()

            if not "path_model_weights" in config:
                self.logger.error(f"'path_model_weights' missing from config file '{config_file}'; exiting'")
                exit()

            # loading config
            external_cfg = config["path_model_cfg"]
            internal_cfg = os.path.join(self.code_base_path,self.mask2former_repo,config["path_model_cfg"].lstrip("/"))

            if os.path.exists(external_cfg):
                self.path_model_cfg = external_cfg
            elif os.path.exists(internal_cfg):
                self.path_model_cfg = internal_cfg
            else:
                self.logger.error(f'found no valid model config matching \'{config["path_model_cfg"]}\'; exiting')
                exit()

            self.logger.info(f"using model config '{self.path_model_cfg}'")

            # loading weights
            if os.path.exists(config["path_model_weights"]):
                self.path_model_weights = config["path_model_weights"]
            else:
                self.logger.error(f'found no valid model at \'{config["path_model_weights"]}\'; exiting')
                exit()

            self.logger.info(f"using model weights '{self.path_model_weights}'")

        else:
            self.logger.error(f"config file '{config_file}' not found; exiting")
            exit()

    def initialize(self):
        """
        initialize() performs some necessary checks.
        """         
        self.image_dir = self.args["input"]
        self.tranform360 = self.args["transform360"]
        self.save_segmentation_images = self.args["save_segmentation_images"]

        if not os.path.exists(self.path_model_weights):
            self.logger.error(f"model '{self.path_model_weights}' doesn't exist: exiting")
            exit()

        if not os.path.exists(self.path_model_cfg):
            self.logger.error(f"model config '{self.path_model_cfg}' doesn't exist: exiting")
            exit()

        if len(self.image_dir)==0:
            self.logger.error(f"no image path specified; exiting")
            exit()

        if not os.path.exists(self.image_dir):
            self.logger.error(f"image path '{self.image_dir}' does not exist (is the container's volume mapping correct?); exiting")
            exit()

        if self.args["suppress_warnings"]:
            warnings.filterwarnings("ignore")

        if self.save_segmentation_images:
            self.transformer = Three60Cube()

    def acquire_images(self):
        """
        acquire_images() reads images from the input folder.
        """ 
        self.images = []
        for filename in os.listdir(self.image_dir):
            if os.path.isfile(os.path.join(self.image_dir,filename)) and os.path.splitext(filename)[1].lower() in self.valid_extensions:
                self.images.append(os.path.join(self.image_dir,filename))

        if (len(self.images)==0):
            self.logger.info(f"found no images in '{self.image_dir}'; exiting")
            exit()
        else:
            self.logger.info(f"found {len(self.images)} image(s) in '{self.image_dir}'")
            self.images.sort()

    def transform360(self):
        """
        transform360() makes a subfolder for each photo to be transformed from a 360° photos to cube projections
        and calls the actula transformer
        """ 
        if self.tranform360:
            self.logger.info(f"transforming 360° photos to cube projections")
            new_images = []
            for image_url in self.images:
                output_folder = os.path.join(self.image_dir,(re.sub(r'[^a-zA-Z\d\s:]','_',re.sub(r'(^[^a-zA-Z\d\s:]*|\.(.*)$)','',os.path.basename(image_url)))) + "/")

                if not os.path.exists(output_folder):
                    # 0o1411 = 777
                    os.mkdir(output_folder,mode=0o777)

                new_paths = self.do_transform360(image_url,output_folder)
                new_images += new_paths
                self.logger.info(f"{image_url} --> {output_folder}")

            self.images = new_images
            self.images.sort()
            self.logger.info(f"created {len(self.images)} images")
        else:
            self.logger.info(f"skipping transformation to cube projection")

    def do_transform360(self,input_file,output_folder):
        """
        do_transform360() does the actual transformation of a 360° photos to 6 cubic panes.
        """ 
        self.transformer.open_image(input_file)
        self.transformer.get_pane(pane=1, dim=512)
        self.transformer.save_cache("templ.pickle")
        self.transformer = Three60Cube("templ.pickle")
        self.transformer.open_image(input_file)

        new_paths = []

        for i in range(6):
            name_bits = os.path.splitext(os.path.basename(input_file))
            new_path = os.path.join(output_folder,f'{name_bits[0]}_{i}{name_bits[1]}')
            self.transformer.save_pane(new_path, pane=i, dim=512)
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
            self.logger.error(f"found no datasets in model config; exiting")
            exit()
        elif len(cfg.get("DATASETS").get("TEST")) > 1:
            self.logger.info(f'found {len(cfg.get("DATASETS").get("TEST"))} datasets in model config; defaulting to the first')

        # model_metadata contains class labels etc.
        self.model_metadata_catalog = cfg.get("DATASETS").get("TEST")[0]
        self.model_metadata = MetadataCatalog.get(self.model_metadata_catalog)
        self.logger.info(f"using metadata catalog '{self.model_metadata_catalog}'")

    def run_predictions(self):
        """
        run_predictions() loops through all images, and has the model perform predictions, outputting the
        results as CSV and optionally as images with overlaid semantic segmentations.
        """ 
        self.logger.info(f"running predictions on {len(self.images)} image(s)")

        calc = utils.SegmentationAreasCalculator()
        calc.set_model_metadata(self.model_metadata)
        
        self.csv_filename = f"segmentation_results-{self.now.strftime('%Y-%m-%dT%H%M')}.csv"

        with open(os.path.join(self.image_dir,self.csv_filename), 'w') as f:

            csv_header = ["image","total pixels"]
            for idx, item in enumerate(self.model_metadata.as_dict()["stuff_classes"]):
                csv_header.append(f"{idx} ({item})")

            csv_writer = csv.writer(f)
            csv_writer.writerow(csv_header)

            for idx, image_url in enumerate(self.images):
                if not os.path.exists(image_url):
                    logger.error(f"image '{image_url}' doesn't exist!? (skipping)")
                    continue

                self.logger.info(f"processing '{image_url}' ({idx+1}/{len(self.images)})")

                # run predictions
                im = cv2.imread(image_url)
                semantic_result, outputs = self.do_run_prediction(im)

                calc.set_outputs(outputs)
                calc.calculate_areas()
                calc.calculate_areas_corrected()

                # utils.print_areas(classes=calc.get_classes_sorted(),total_area=calc.get_total_area(),model_metadata=self.model_metadata,logger=self.logger)
                # utils.print_areas_corrected(classes=calc.get_classes_corrected(),total_area=calc.get_total_area(),logger=self.logger)

                # write to CSV
                row = [image_url[len(self.image_dir):].rstrip("/"),calc.get_total_area()]

                for jdx, item in enumerate(self.model_metadata.as_dict()["stuff_classes"]):
                    t = [x for x in calc.get_classes_corrected() if x[0]==jdx]
                    if len(t)>0:
                        row.append(t[0][2])
                    else:
                        row.append(0)
                
                csv_writer.writerow(row)

                # save copy of image with segmentation overlay
                if self.save_segmentation_images:
                    seg_folder = os.path.join(os.path.dirname(image_url),"segmentations")
                    if not os.path.exists(seg_folder):
                        os.mkdir(seg_folder,mode=0o777)

                    name_bits = os.path.splitext(os.path.basename(image_url))
                    seg_file = os.path.join(seg_folder,f'{name_bits[0]}-segmentation{name_bits[1]}')

                    cv2.imwrite(seg_file, semantic_result)
                    os.chmod(seg_file,mode=0o777)
                    self.logger.info(f"saved '{seg_file}'")

                # talk to user
                t = []
                for i, c in enumerate(calc.get_classes_corrected()):
                    t.append(f'{c[1]}: {c[3]}%')
                    if i  >= 2:
                        break

                self.logger.info(f"top results: {'; '.join(t)}; ...")

    def do_run_prediction(self,im):
        """
        do_run_prediction() runs the actual predictions.
        """ 
        outputs = self.predictor(im)
        v = Visualizer(im[:, :, ::-1], self.model_metadata, scale=1, instance_mode=ColorMode.IMAGE_BW)
        semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
        return semantic_result, outputs

    def finalize(self):
        self.logger.info(f"wrote results to '{os.path.join(self.image_dir,self.csv_filename)}'")
        self.logger.info(f"done")


if __name__ == '__main__':
    i = ImageSegmentation()
    i.main()
