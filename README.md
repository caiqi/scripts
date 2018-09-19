try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
import os


def _load_label(anno_path):
    """Parse xml file and return labels."""
    root = ET.parse(anno_path).getroot()
    size = root.find('size')
    width = float(size.find('width').text)
    height = float(size.find('height').text)
    label = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text.strip().lower()
        xml_box = obj.find('bndbox')
        xmin = (float(xml_box.find('xmin').text))
        ymin = (float(xml_box.find('ymin').text))
        xmax = (float(xml_box.find('xmax').text))
        ymax = (float(xml_box.find('ymax').text))
        xmax = min(xmax, width)
        ymax = min(ymax, height)
        try:
            _validate_label(xmin, ymin, xmax, ymax, width, height)
        except AssertionError as e:
            raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
        label.extend([xmin, ymin, xmax, ymax, cls_name])
    return label


def _validate_label(xmin, ymin, xmax, ymax, width, height):
    """Validate labels."""
    assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
    assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
    assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
    assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)


from tqdm import tqdm

if __name__ == '__main__':
    annotation_file = open("anno.txt").readlines()
    annotation_file = [m.strip() for m in annotation_file]
    all_data = []
    for m in tqdm(annotation_file):
        label = _load_label(m)
        all_data.append(m.strip().replace("xml", "JPEG").replace("Annotations", "Data") + " " + " ".join(
            [str(k) for k in label]) + "\n")
    with open("all_annotation_from_xml.txt", "w") as g:
        for m in all_data:
            g.write(m.strip() + "\n")
