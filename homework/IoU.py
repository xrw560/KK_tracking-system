# -*-coding: utf-8 -*-

def compu_iou(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    # 计算每个矩形的面积
    S_rec1 = (cx2 - cx1) * (cy2 - cy1)  # C的面积
    S_rec2 = (gx2 - gx1) * (gy2 - gy1)  # G的面积

    # 计算相交矩形
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


if __name__ == "__main__":
    rec1 = [661, 27, 679, 47]
    rec2 = [662, 27, 682, 47]
    iou = compu_iou(rec1, rec2)
    print(iou)
