import cv2
import numpy as np
from SORF import SorfExtractor

__author__ = 'rons'


class FeaturesExtractor:
    def __init__(self):
        pass

    @staticmethod
    def change2gray_if(image):
        if image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = image
        return gray_img

    @staticmethod
    def get_sift_points_and_features(image, max_num_of_sift=100, is_use_sorf=False):
        gray_img = FeaturesExtractor.change2gray_if(image)
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=max_num_of_sift)
        if is_use_sorf:
            sorf_ex = SorfExtractor()
            # sorfpyramid, center_srnd_pyramid, pad_r, pad_c = sorf_ex.get_sorf("", image)
            # s0 = sorfpyramid[0]
            # [m,n] = s0.shape
            # s0 = s0[:(m-pad_r), :(n-pad_c)]
            # s0 = 255 * s0
            # s0 = s0.astype('uint8')
            s0 = sorf_ex.get_sorf("", image)
            keypoints = detector.detect(s0)
            (keypoints, descriptors) = detector.compute(gray_img, keypoints)
        else:
            (keypoints, descriptors) = detector.detectAndCompute(gray_img, None)
        # print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
        return keypoints, descriptors

    @staticmethod
    def get_matched_sift_points_and_features_from_image_pair(image1, image2):

        keypoints1, descriptors1 = FeaturesExtractor.get_sift_points_and_features(image1)
        keypoints2, descriptors2 = FeaturesExtractor.get_sift_points_and_features(image2)
        bf = cv2.BFMatcher()
        # Match descriptors with 2 nearest neighbours
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good = []
        nndist = 0.75
        for m, n in matches:
            if m.distance < nndist * n.distance:
                good.append(m)

        matched_keypoints1 = [keypoints1[good[i].queryIdx] for i in range(len(good))]
        matched_descriptors1 = np.array([descriptors1[good[i].queryIdx] for i in range(len(good))])
        matched_keypoints2 = [keypoints2[good[i].trainIdx] for i in range(len(good))]
        matched_descriptors2 = np.array([descriptors2[good[i].trainIdx] for i in range(len(good))])
        valid_matches = bf.match(matched_descriptors1, matched_descriptors2)
        return (matched_keypoints1, matched_keypoints2), (matched_descriptors1, matched_descriptors2), valid_matches

    @staticmethod
    def get_surf_points_and_features(image):
        gray_img = FeaturesExtractor.change2gray_if(image)
        detector = cv2.xfeatures2d.SURF_create()
        (keypoints, descriptors) = detector.detectAndCompute(gray_img, None)
        print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
        return keypoints, descriptors

    @staticmethod
    def get_matched_surf_points_and_features_from_image_pair(image1, image2):

        keypoints1, descriptors1 = FeaturesExtractor.get_surf_points_and_features(image1)
        keypoints2, descriptors2 = FeaturesExtractor.get_surf_points_and_features(image2)
        bf = cv2.BFMatcher()
        # Match descriptors with 2 nearest neighbours
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good = []
        nndist = 0.75
        for m, n in matches:
            if m.distance < nndist * n.distance:
                good.append(m)

        matched_keypoints1 = [keypoints1[good[i].queryIdx] for i in range(len(good))]
        matched_descriptors1 = np.array([descriptors1[good[i].queryIdx] for i in range(len(good))])
        matched_keypoints2 = [keypoints2[good[i].trainIdx] for i in range(len(good))]
        matched_descriptors2 = np.array([descriptors2[good[i].trainIdx] for i in range(len(good))])
        valid_matches = bf.match(matched_descriptors1, matched_descriptors2)
        return (matched_keypoints1, matched_keypoints2), (matched_descriptors1, matched_descriptors2), valid_matches

    @staticmethod
    def get_brisk_points_and_features(image):
        gray_img = FeaturesExtractor.change2gray_if(image)
        detector = cv2.BRISK_create()
        (keypoints, descriptors) = detector.detectAndCompute(gray_img, None)
        # print("keypoints: {}, descriptors: {}".format(len(keypoints), descriptors.shape))
        return keypoints, descriptors

    @staticmethod
    def get_matched_brisk_points_and_features_from_image_pair(image1, image2):

        keypoints1, descriptors1 = FeaturesExtractor.get_brisk_points_and_features(image1)
        keypoints2, descriptors2 = FeaturesExtractor.get_brisk_points_and_features(image2)
        matches = FeaturesExtractor.sift_matcher()
        matched_keypoints1 = [keypoints1[matches[i].queryIdx] for i in range(len(matches))]
        matched_descriptors1 = np.array([descriptors1[matches[i].queryIdx] for i in range(len(matches))])
        matched_keypoints2 = [keypoints2[matches[i].trainIdx] for i in range(len(matches))]
        matched_descriptors2 = np.array([descriptors2[matches[i].trainIdx] for i in range(len(matches))])
        valid_matches = bf.match(matched_descriptors1, matched_descriptors2)
        return (matched_keypoints1, matched_keypoints2), (matched_descriptors1, matched_descriptors2), valid_matches

    @staticmethod
    def get_hough_lines(image):
        gray_img = FeaturesExtractor.change2gray_if(image)
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
        min_votes = 100
        lines = cv2.HoughLines(edges, 1, np.pi / 180, min_votes)
        # draw lines
        image_with_lines = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        for idx in range(lines.shape[0]):
            rho = lines[idx][0, 0]
            theta = lines[idx][0, 1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("lines", image_with_lines)
        return lines

    @staticmethod
    def get_prob_hough_lines(image):
        gray_img = FeaturesExtractor.change2gray_if(image)
        edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)

        min_line_length = 100
        max_line_gap = 10
        min_votes = 50
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, min_votes, min_line_length, max_line_gap)
        # draw lines
        image_with_lines = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        for row_idx in range(lines.shape[0]):
            x1 = lines[row_idx][0, 0]
            y1 = lines[row_idx][0, 1]
            x2 = lines[row_idx][0, 2]
            y2 = lines[row_idx][0, 3]
            cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("prob lines", image_with_lines)
        return lines

    @staticmethod
    def get_line_segments(image):
        gray_img = FeaturesExtractor.change2gray_if(image)
        line_detector = cv2.createLineSegmentDetector()
        lines = line_detector.detect(gray_img)

        # draw lines
        image_with_lines = cv2.copyMakeBorder(image, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
        try:
            retval, newimage = line_detector.compareSegments(gray_img.shape, lines[0], lines[2], image_with_lines)
            cv2.imshow("comp im", newimage)
        except Exception as e:
            print (e.message)
            pass
        ind = 0
        for l_s in lines:
            image_with_lines = line_detector.drawSegments(gray_img, lines[0])
            cv2.imshow("im" + str(ind), image_with_lines)
            ind += 1
        return lines

    @staticmethod
    def sift_matcher(descriptors1, descriptors2, nndist=0.75):
        bf = cv2.BFMatcher()
        # Match descriptors with 2 nearest neighbours
        matches = bf.knnMatch(descriptors1, descriptors2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < nndist * n.distance:
                good_matches.append(m)
        return good_matches
