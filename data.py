

class CityScapeDetection(VisionDataset):
    """Pascal VOC detection Dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, defaut None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indicies. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extreamly large.
    """
    CLASSES = ['car']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits="",
                 transform=None, index_map=None, preload_label=True, min_dataset_size=-1):
        super(CityScapeDetection, self).__init__(root)
        self._im_shapes = {}
        self.min_dataset_size = min_dataset_size

        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        # self._image_path = os.path.join('{}')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = self._splits
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        # img_path = self._image_path.format(*img_id)
        img_path = img_id[0]
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        with open(os.path.join(self._root, splits), 'r') as f:
            for line in f.readlines():
                line = line.strip().split(" ")
                path = os.path.join(self._root, line[0])
                others = line[1:]
                ids.append([path] + others)
        if self.min_dataset_size > 0:
            print("{}: padding from : {} to {}".format(self._splits, len(ids), self.min_dataset_size))
            while (len(ids)) < self.min_dataset_size:
                ids = ids + ids
            ids = ids[:self.min_dataset_size]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        annotation_data = img_id[1:]
        annotation_data = np.array([float(k) for k in annotation_data])
        annotation_data = np.reshape(annotation_data, newshape=(-1, 5))
        annotation_data = np.concatenate((annotation_data, np.zeros(shape=(annotation_data.shape[0], 1))), axis=1)
        return np.array(annotation_data)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
