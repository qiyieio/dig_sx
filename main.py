import cv2
import numpy as np
import os

#cv2显示图像
def show(image, window_name):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)

#图像预处理
#读取本地图片
image = cv2.imread('imgs/img01.png')
show(image, "image")
print(image.shape)
#裁剪
a,b,c = image.shape
xs = int(0.45*b)
xe = int(0.75*b)
ys = int(0.31*a)
ye = int(0.7*a)
image = image[ys:ye,xs:xe]
#灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# show(gray, "gray")
#中值滤波
# blur = cv2.medianBlur(gray, 3)
# show(blur, "blur")
#二值化
threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# threshold1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)[1]
# show(threshold, "threshold")
#边缘检测
canny = cv2.Canny(threshold, 100, 150)
# show(canny, "canny")
#边缘膨胀
kernel = np.ones((2, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations=5)
# show(dilate, "dilate")
#轮廓检测
contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()
res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 3)
# show(res, "res")
#膨胀2
kernel = np.ones((1, 3), np.uint8)
dilate2 = cv2.dilate(dilate, kernel, iterations=5)
# show(dilate2, "dilate2")

contours, hierarchy = cv2.findContours(dilate2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
resize_copy = image.copy()
res3 = cv2.drawContours(resize_copy, contours, -1, (255, 0, 0), 2)
show(res3, "res3")
##############################预处理结束

# 筛选轮廓区域，这里以面积为例
min_area = 100  # 最小面积
max_area = 1000  # 最大面积
filtered_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if min_area < area < max_area:
        filtered_contours.append(contour)

# 在图像上绘制筛选后的轮廓
res4 = cv2.drawContours(resize_copy, filtered_contours, -1, (255, 0, 0), 2)
# show(res4, "res4")

# 还可以获取筛选后轮廓的外接矩形区域
# for contour in filtered_contours:
#     x, y, w, h = cv2.boundingRect(contour)
#     cv2.rectangle(resize_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
# show(resize_copy, "res_with_rectangles")
#轮廓排序函数
def sort_contours(contours):
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                           key=lambda b: b[1][1], reverse=False))
    return contours

# 对轮廓进行排序
sorted_contours = sort_contours(contours)

# 获取第一个轮廓
if sorted_contours:
    first_contour = sorted_contours[0]
    # 获取该轮廓的外接矩形
    x, y, w, h = cv2.boundingRect(first_contour)
    # 裁剪区域
    first_contour_region = image[y:y + h, x:x + w]
    # 显示
    show(first_contour_region, "First Contour Region")
    # 保存
    cv2.imwrite("first_contour_region.png", first_contour_region)
else:
    print("未检测到轮廓")

a = os.system("tesseract.exe first_contour_region.png out  -l eng --psm 7")
print(a)

with open('out.txt', 'r') as file:
    line = file.readline()
    print('识别到学号:',line)
