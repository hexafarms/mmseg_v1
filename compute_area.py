import numpy as np
import cv2


def compute_area(graph_cut_result: np.ndarray, thres=30):
    '''
    graph_cut_result : mask after graphcut
    thres : minimum area of detectable plants (less than this value is regards as noise)

    area: the number of pixel
    '''
    kernel = np.ones((21, 21), np.uint8)

    output = cv2.morphologyEx(graph_cut_result.astype(
        'uint8'), cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(output,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


    area = 0
    filtered_contours = []
    for contour in contours:
        c_area = cv2.contourArea(contour)
        if c_area > thres:
            area += c_area
            filtered_contours.append(contour)


    return area
