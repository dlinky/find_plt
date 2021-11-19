import copy

import cv2
import numpy as np


def empty(self):
    pass


def find_plt(img, labels):
    windowName = 'find_plt'
    cv2.namedWindow(windowName)
    cv2.createTrackbar('param1', windowName, 10, 255, empty)
    cv2.createTrackbar('param2', windowName, 20, 255, empty)
    cv2.createTrackbar('minDist', windowName, 10, 100, empty)
    cv2.createTrackbar('minRadius', windowName, 3, 50, empty)
    cv2.createTrackbar('maxRadius', windowName, 10, 100, empty)
    labels = np.uint8(labels)

    while True:
        if cv2.waitKey(10) == 27:
            cv2.destroyAllWindows()
            break
        temp = copy.deepcopy(img)
        param1 = cv2.getTrackbarPos('param1', windowName)
        param2 = cv2.getTrackbarPos('param1', windowName)
        minDist = cv2.getTrackbarPos('minDist', windowName)
        minRadius = cv2.getTrackbarPos('minRadius', windowName)
        maxRadius = cv2.getTrackbarPos('maxRadius', windowName)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, param1=param1+1, param2=param2+1, dp=1, minDist=minDist+1, minRadius=minRadius+1, maxRadius=maxRadius+1)
        if circles is not None:
            for circle in circles[0,:]:
                if labels[int(circle[1])][int(circle[0])] == 0:
                    cv2.circle(temp, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)
                else:
                    cv2.circle(temp, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)

        cv2.imshow(windowName, temp)
    if circles is not None:
        bboxes = []
        for circle in circles[0, :]:
            if labels[int(circle[0])][int(circle[1])] == 0:
                bbox = ['Platelets', int(circle[0]-circle[2]), int(circle[1]-circle[2]), int(circle[0]+circle[2]), int(circle[1]+circle[2])]
                bboxes.append(bbox)
    return bboxes


def post_process(img):
    """
    이미지 전처리
    1. hue, value로 adaptive theshold하여, 둘다 만족하는 픽셀만 취함
    2. flood fill로 안쪽 채움
    3. morphology operation(open)
    """
    print('post_process', end='', flush=True)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    threshold_block_size = len(v)//3 + (len(v)//3+1)%2
    threshed_value = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, threshold_block_size, 0)
    threshed_hue = cv2.adaptiveThreshold(h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, threshold_block_size, 0)
    threshed = threshed_value
    # threshed = cv2.bitwise_and(threshed_hue, threshed_value)

    # Fill Flood
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        flood_filled = cv2.drawContours(threshed, [c], 0, (255, 255, 255), -1)

    # Morphology Operation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_open = cv2.morphologyEx(flood_filled, cv2.MORPH_OPEN, kernel, iterations=2)

    print(', ', end='', flush=True)
    return morph_open


def cvt_stats_to_bboxes(stats):
    bboxes = []
    for line in stats:
        print(line)
        bboxes.append(['Undefined', line[1], line[2], line[1]+line[3], line[2]+line[4]])
    return bboxes


def label_img(img):
    """
    전처리된 이미지를 레이블링하고, 각 레이블마다 픽셀 정보를 담은 리스트를 반환
    :param img: 전처리된 이미지
    :return: 레이블 리스트
    """
    print('ccl', end='', flush=True)
    _, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(img, 4, cv2.CV_16U, cv2.CCL_WU)
    bboxes = cvt_stats_to_bboxes(stats)
    print(len(bboxes))
    cv2.imshow('label', np.uint8(labels))
    cv2.waitKey(0)

    print(', ', end='', flush=True)
    return labels, bboxes


def print_box(boxes, img):
    """
    bounding box 이미지에 출력해서 저장
    """
    print('print_box', end='', flush=True)
    names = ['WBC', 'RBC', 'Platelets']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for i, item in enumerate(boxes):
        color = colors[names.index(item[0])]
        xmin = item[1]
        ymin = item[2]
        xmax = item[3]
        ymax = item[4]
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.imshow('final', img)
    print(', ', end='', flush=True)


img = cv2.imread('sample.jpg')
threshed = post_process(img)
labels, bboxes = label_img(threshed)
print('cvt_stats')
plts = find_plt(img, labels)
bboxes.extend(plts)
