import numpy as np
import cv2
import math

__author__ = 'rons'


class SorfExtractor:
    def __init__(self):
        pass

    @staticmethod
    def get_sorf(image_path, image=None):
        luminance, image_hsv = SorfExtractor.get_luminance_and_colorcomp(image_path, image)
        luminance = SorfExtractor.convert2norm_double(luminance)
        i0 = luminance
        number_of_levels = 4
        blurred_pyramid, pad_r, pad_c = SorfExtractor.generate_blurred_pyramid(i0, number_of_levels)
        sorfpyramid, center_srnd_pyramid = SorfExtractor.generate_sorf_pyramid(blurred_pyramid, number_of_levels)
        s0 = sorfpyramid[0]
        [m, n] = s0.shape
        s0 = s0[:(m - pad_r), :(n - pad_c)]
        s0 = 255 * s0
        s0 = s0.astype('uint8')

        # return sorfpyramid, center_srnd_pyramid, pad_r, pad_c
        return s0

    @staticmethod
    def get_luminance_and_colorcomp(image_path, bgr_image=None):
        if bgr_image is None:
            image = cv2.imread(image_path)
        else:
            image = bgr_image
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        luminance = image_hsv[:, :, 2]
        return luminance, image_hsv

    @staticmethod
    def convert2norm_double(luminance):
        if np.max(luminance) < 1:
            return luminance.astype('float')
        else:
            return luminance.astype('float') / 255

    @staticmethod
    def generate_blurred_pyramid(luminance, number_of_levels):
        padded_image, pad_r, pad_c = SorfExtractor.pad_image(luminance, number_of_levels)

        blurred_pyramid = []
        blurred_pyramid.append(padded_image)

        for pyr_ind in range(number_of_levels):
            pyr_down = cv2.pyrDown(blurred_pyramid[pyr_ind])
            blurred_pyramid.append(pyr_down)
        # Todo: add smart stop condition
        return blurred_pyramid, pad_r, pad_c

    @staticmethod
    def pad_image(luminance, number_of_levels):
        pad_factor = math.pow(2, number_of_levels + 1)
        [row, col] = luminance.shape
        pad_r = 0
        pad_c = 0
        if row % pad_factor != 0:
            pad_r = pad_factor - row % pad_factor
        if col % pad_factor != 0:
            pad_c = pad_factor - col % pad_factor
        padded_image = cv2.copyMakeBorder(luminance, 0, pad_r, 0, pad_c, cv2.BORDER_REPLICATE)
        return padded_image, pad_r, pad_c

    @staticmethod
    def generate_sorf_pyramid(blurred_pyramid, number_of_levels):
        sorfpyramid = []
        center_srnd_pyramid = []
        # generate center surround pyramid
        for ind in range(number_of_levels):
            curr_pyr_step = blurred_pyramid[ind]
            expend_curr_pyr = cv2.pyrUp(blurred_pyramid[ind + 1])
            center_srnd = abs(curr_pyr_step - expend_curr_pyr)
            p2_center_srnd = np.power(center_srnd, 2)
            center_srnd = cv2.blur(p2_center_srnd, (3, 3)) / (cv2.blur(center_srnd, (3, 3)) + np.finfo(float).eps)
            center_srnd_pyramid.append(center_srnd)

        # generate sorf pyramid
        sorfpyramid.append(center_srnd_pyramid[len(center_srnd_pyramid) - 1])
        temp = 0.5 * center_srnd_pyramid[len(center_srnd_pyramid) - 2] + 0.5 * cv2.pyrUp(
            center_srnd_pyramid[len(center_srnd_pyramid) - 1])
        sorfpyramid.insert(0, temp)

        for i in reversed(range(number_of_levels - 2)):
            t1 = center_srnd_pyramid[i]
            t2 = cv2.pyrUp(center_srnd_pyramid[i + 1])
            t3 = cv2.pyrUp(cv2.pyrUp(center_srnd_pyramid[i + 2]))
            temp = (t1 + t2 + t3) / 3
            sorfpyramid.insert(0, temp)

        for ind in range(number_of_levels):
            sorfpyramid[ind] = SorfExtractor.norm_and_convert_image_2_double(sorfpyramid[ind])
        return sorfpyramid, center_srnd_pyramid

    @staticmethod
    def norm_and_convert_image_2_double(im):
        min_val = np.min(im.ravel())
        max_val = np.max(im.ravel())
        return (im.astype('float') - min_val) / (max_val - min_val)



















