from enum import Enum, auto

import tensorflow as tf
import cv2
from numpy.core.multiarray import ndarray
from tensorflow import Tensor


class CorrectionType(Enum):
    LINEAR = auto()
    EXPONENTIAL = auto()


class ImgProcessor:
    def __init__(self, img_path):
        img = cv2.imread(img_path)
        img = img[..., ::-1]  # BGR to RGB
        self.img = tf.convert_to_tensor(img, tf.float32)
        self.hsv = tf.image.rgb_to_hsv(img)

    @staticmethod
    def rgb_to_display(img: Tensor) -> ndarray:
        return tf.cast(img, tf.uint8).numpy()

    @classmethod
    def hsv_to_display(cls, img: Tensor) -> ndarray:
        rgb = tf.image.hsv_to_rgb(img)
        return cls.rgb_to_display(rgb)

    def _linear_correction(self, **kwargs):
        hs, v = (self.hsv[..., :2],
                 self.hsv[..., -1:])  # -1: saves last dimension
        i_max = tf.reduce_max(v)
        i_min = tf.reduce_min(v)
        v = tf.math.divide(tf.math.subtract(v, i_min),
                           tf.math.subtract(i_max, i_min)) * 255
        return tf.concat((hs, v), axis=-1)

    def _exponential_correction(self, c: float = 1, p: float = 0.9):
        if c > 1:
            c = 1
        hs, v = (self.hsv[..., :2],
                 self.hsv[..., -1:])
        c_tensor = tf.constant(c, dtype=tf.float32)
        p_tensor = tf.constant(p, dtype=tf.float32)
        v = v / 255
        v = tf.pow(v, p_tensor)
        v = tf.multiply(c_tensor, v) * 255
        return tf.concat((hs, v), axis=-1)

    def correction(self, correction_type: CorrectionType, **kwargs):
        methods = {
            CorrectionType.LINEAR: self._linear_correction,
            CorrectionType.EXPONENTIAL: self._exponential_correction
        }
        return methods[correction_type](**kwargs)

    @staticmethod
    def gaussian_kernel(size: int,
                        mean: float,
                        std: float,
                        ):
        distribution = tf.compat.v1.distributions.Normal(mean, std)
        values = distribution.prob(
            tf.range(start=-size, limit=size + 1, dtype=tf.float32))
        kernel = tf.einsum('i,j->ij', values, values)  # outer product
        return kernel / tf.reduce_sum(kernel)

    @staticmethod
    def box_kernel(size: int):
        kernel = tf.ones(shape=(size, size), dtype=tf.float32)
        return kernel / size ** 2

    @staticmethod
    def unsharp_masking_kernel(lambda_: int = 1):
        tf1 = tf.constant([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=tf.float32)
        tf2 = tf.constant([[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                          dtype=tf.float32)
        return tf1 + lambda_ * tf2

    @staticmethod
    def _sobel_kernel_x():
        return tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]],
                           dtype=tf.float32)

    @staticmethod
    def _sobel_kernel_y():
        return tf.constant([[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
                           dtype=tf.float32)

    def apply_filter(self, kernel_func, **kernel_kwargs):
        kernel = kernel_func(**kernel_kwargs)
        kernel = tf.tile(kernel[..., tf.newaxis, tf.newaxis], [1, 1, 3, 1])
        point_wise_filter = tf.eye(3, batch_shape=[1, 1])
        return tf.nn.separable_conv2d(tf.expand_dims(self.img, axis=0),
                                      kernel,
                                      point_wise_filter,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')[0]

    def sobel_edges(self):
        edges_x = self.apply_filter(self._sobel_kernel_x)
        edges_y = self.apply_filter(self._sobel_kernel_y)
        return edges_x, edges_y
