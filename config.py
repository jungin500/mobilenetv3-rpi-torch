import numpy as np
from box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 500
image_mean = np.array([127, 127, 127])  # RGB layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(19, 16, SSDBoxSizes(5, 20), [1, 2]),
    SSDSpec(10, 32, SSDBoxSizes(20, 95), [1, 2]),
    SSDSpec(5, 64, SSDBoxSizes(95, 130), [1, 2]),
    SSDSpec(3, 100, SSDBoxSizes(130, 195), [1, 2]),
    SSDSpec(2, 150, SSDBoxSizes(195, 250), [1, 2]),
    SSDSpec(1, 300, SSDBoxSizes(250, 330), [1, 2])
]


priors = generate_ssd_priors(specs, image_size)