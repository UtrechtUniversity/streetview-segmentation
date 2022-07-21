import logging, os
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

def print_areas(classes,total_area,model_metadata,logger):
    logger.info(f'Semantic classes sorted by area (total {total_area:,}px)')
    for key in classes:
        pct = round((int(classes[key]) / total_area) * 100)
        logger.info(f'{key:>2}; {model_metadata.as_dict()["stuff_classes"][key]}: {classes[key]:,}px ({pct}%)')

def print_areas_corrected(classes,total_area,logger):
    logger.info(f'Semantic classes sorted by area, corrected (total {total_area:,}px)')
    for row in classes:
        logger.info(f'{row[0]:>2}; {row[1]}: {row[2]:,}px ({row[3]}%)')

class SegmentationAreasCalculator:

    semantic_outputs = None
    model_metadata = None
    classes = None
    classes_corrected = None
    total_area = None
    total_area_weighted = None

    def set_outputs(self,outputs):
        self.semantic_outputs = outputs["sem_seg"].argmax(0).cpu().detach().numpy()

    def set_model_metadata(self,model_metadata):
        self.model_metadata = model_metadata

    def set_total_area(self):
        self.total_area = (sum(self.classes[c] for c in self.classes))

    def get_classes(self):
        return self.classes  

    def get_classes_sorted(self):
        return {k: v for k, v in sorted(self.classes.items(), key=lambda item: item[1], reverse=True)}

    def get_classes_corrected(self):
        return self.classes_corrected  

    def get_total_area(self):
        return self.total_area

    def calculate_areas(self):
        class_ids, counts = np.unique(self.semantic_outputs, return_counts=True)
        self.classes = dict(zip(class_ids, counts))
        self.set_total_area()

    def calculate_areas_corrected(self):
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

        self.classes_corrected = []
        for key in classes_sorted:
            pct = round((int(classes_sorted[key]) / self.total_area_weighted) * 100)
            self.classes_corrected.append([key,self.model_metadata.as_dict()["stuff_classes"][key],(round(int(classes_sorted[key]) * area_correction)),pct])

    def do_cubic_weighted_weights(self,matrix_shape):
        idx = np.indices(matrix_shape)
        xval = idx[1].reshape(-1)
        yval = idx[0].reshape(-1)

        dx = 2*(xval+0.5)/matrix_shape[1] - 1
        dy = 2*(yval+0.5)/matrix_shape[0] - 1

        fac = (dx**2+dy**2+1)**-1.5

        return fac


