import logging, os, stat
import numpy as np

def get_logger(name):
    # adding logger (screen only)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def chmod_recursively(file_or_folder,mask=0o777):
    os.chmod(file_or_folder,mask)
    if os.path.isdir(file_or_folder):
        for path,subdir,files in os.walk(file_or_folder):
           for name in subdir:
               os.chmod(os.path.join(path,name),mask)
           for name in files:
               os.chmod(os.path.join(path,name),mask)

class SegmentationAreasCalculator:

    semantic_outputs = None
    model_metadata = None
    stuff_classes = None
    classes = None
    classes_corrected = None
    total_area = None
    total_area_weighted = None

    def set_outputs(self,outputs):
        self.semantic_outputs = outputs["sem_seg"].argmax(0).cpu().detach().numpy()

    def set_model_metadata(self,model_metadata):
        self.model_metadata = model_metadata
        self.stuff_classes = self.model_metadata.as_dict()["stuff_classes"]

    def set_total_area(self):
        self.total_area = (sum(self.classes[c] for c in self.classes))

    def get_model_classes(self):
        """
        get_model_classes() returns all available "stuff" classes in the model.
        """
        return self.stuff_classes

    def get_classes(self):
        """
        get_classes() returns a list of semantic classes found in the image and the amount of pixels
        they take up. per class, a dict is returned containing "key", "class", "pixels" and "percentage".
        note, this is not a complete list of all classes present in the model, only those that are actually
        present in the image. a complete list is available by calling get_model_classes()
        """
        return self.classes

    def get_classes_corrected(self):
        """
        get_classes_corrected() as get_classes(), but corrected for the deviations that occur due to the 360°
        image being projected onto a cube.
        """
        return self.classes_corrected

    def get_total_area(self):
        return self.total_area

    def set_areas(self):
        class_ids, counts = np.unique(self.semantic_outputs, return_counts=True)
        self.classes = dict(zip(class_ids, counts))
        self.set_total_area()

        classes = []
        for key in self.classes:
            pct = round((int(self.classes[key]) / self.total_area) * 100)
            classes.append({"key":key,"class":self.stuff_classes[key],"pixels":self.classes[key],"percentage":pct})

        self.classes = sorted(classes, key=lambda item: item["pixels"], reverse=True)

    def set_areas_corrected(self):
        """
        calculate_areas_corrected() corrected for the deviations that occur due to the 360° image being projected
        onto a cube.
        """
        # correct cubic weights
        fac = self.do_cubic_weighted_weights(np.shape(self.semantic_outputs))
        fac = fac.reshape(np.shape(self.semantic_outputs))

        classes = {}

        for iy, ix in np.ndindex(self.semantic_outputs.shape):
            c = self.semantic_outputs[iy, ix]
            f = fac[iy, ix]
            if c in classes.keys():
                classes[c] += f
            else:
                classes.update({c:f})

        classes_sorted = {k: v for k, v in sorted(classes.items(), key=lambda item: item[1], reverse=True)}
        self.total_area_weighted = (sum(classes_sorted[c] for c in classes_sorted))
        area_correction = (self.get_total_area() / self.total_area_weighted)

        # area_correction: resize to original total size (correction same for all pixels)
        self.classes_corrected = []
        for key in classes_sorted:
            pct = round((int(classes_sorted[key]) / self.total_area_weighted) * 100)
            self.classes_corrected.append({"key":key,"class":self.stuff_classes[key],"pixels":(round(int(classes_sorted[key]) * area_correction)),"percentage":pct})

    def do_cubic_weighted_weights(self,matrix_shape):
        idx = np.indices(matrix_shape)
        xval = idx[1].reshape(-1)
        yval = idx[0].reshape(-1)

        dx = 2*(xval+0.5)/matrix_shape[1] - 1
        dy = 2*(yval+0.5)/matrix_shape[0] - 1

        fac = (dx**2+dy**2+1)**-1.5

        return fac
