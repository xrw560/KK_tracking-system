# -*-coding: utf-8 -*-
import numpy as np


def NMS1(dets, thresh):
    x1, y1, x2, y2, scores = [dets[:, i] for i in range(5)]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # 像素点个数,所以需要加1
    order = scores.argsort()[::-1]  # 按得分倒序排序后的索引
    print(order)
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h  # 矩形交集
        iou = inter / (areas[i] + areas[order[1:]] - inter)  # iou
        index = np.where(iou <= thresh)[0]
        order = order[index + 1]  # 由于order[1:]，排除了第一个[0]，此处需要补回
    return keep


def NMS2(lists, thre):
    if len(lists) == 0:
        return {}
    lists = np.array(lists)
    res = []
    x1, y1, x2, y2, score = [lists[:, i] for i in range(5)]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    # get sorted index in ascending order
    idxs = np.argsort(score)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        res.append(i)

        xmin = np.maximum(x1[i], x1[idxs[:last]])
        ymin = np.maximum(y1[i], y1[idxs[:last]])
        xmax = np.minimum(x2[i], x2[idxs[:last]])
        ymax = np.minimum(y2[i], y2[idxs[:last]])
        w = np.maximum(0, xmax - xmin + 1)
        h = np.maximum(0, ymax - ymin + 1)
        inner_area = w * h
        iou = inner_area / (area[i] + area[idxs[:last]] - inner_area)

        idxs = np.delete(idxs, np.concatenate(([last], np.where(iou > thre)[0])))
        # here "where" will return us a tuple
        # [0] means to extract array from a tuple
    return res


if __name__ == "__main__":
    dets = [
        [50, 51, 60, 61, 0.9],
        [55, 56, 65, 66, 0.8],
        [57, 58, 70, 71, 0.85],
        [90, 91, 97, 98, 0.6],
        [92, 93, 99, 100, 0.92]
    ]
    print(NMS1(np.array(dets), 0.3))
    print(NMS2(dets, 0.3))
