import sys
import cv2
import numpy as np
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

def show(image, window_name):
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


class ImageProcessingUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.first_contour_region = None

    def initUI(self):
        # 创建布局
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # 创建按钮
        self.select_button = QPushButton('选择图片', self)
        self.select_button.clicked.connect(self.select_image)
        button_layout.addWidget(self.select_button)

        self.recognize_button = QPushButton('识别', self)
        self.recognize_button.clicked.connect(self.recognize_image)
        self.recognize_button.setEnabled(False)
        button_layout.addWidget(self.recognize_button)

        self.view_region_button = QPushButton('查看学号区域', self)
        self.view_region_button.clicked.connect(self.view_region)
        self.view_region_button.setEnabled(False)
        button_layout.addWidget(self.view_region_button)

        # 创建图像显示标签
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        # 创建结果显示标签
        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        # 添加布局
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.image_label)
        main_layout.addWidget(self.result_label)

        # 设置布局
        self.setLayout(main_layout)

        # 设置窗口属性
        self.setWindowTitle('图片识别 UI')
        self.setGeometry(300, 300, 800, 600)

    def select_image(self):
        global file_path
        file_path, _ = QFileDialog.getOpenFileName(self, '选择图片', '', '图像文件 (*.png *.jpg *.jpeg)')
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
            self.recognize_button.setEnabled(True)
            self.view_region_button.setEnabled(False)
            self.result_label.setText('')

    def recognize_image(self):
        if self.image is not None:
            # 图像预处理
            a, b, c = self.image.shape
            xs = int(0.45 * b)
            xe = int(0.75 * b)
            ys = int(0.31 * a)
            ye = int(0.7 * a)
            image = self.image[ys:ye, xs:xe]
            # 灰度化
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # show(gray, "gray")
            # 中值滤波
            # blur = cv2.medianBlur(gray, 3)
            # show(blur, "blur")
            # 二值化
            threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            # threshold1 = cv2.threshold(gray,120,255,cv2.THRESH_BINARY_INV)[1]
            # show(threshold, "threshold")
            # 边缘检测
            canny = cv2.Canny(threshold, 100, 150)
            # show(canny, "canny")
            # 边缘膨胀
            kernel = np.ones((2, 3), np.uint8)
            dilate = cv2.dilate(canny, kernel, iterations=5)
            # show(dilate, "dilate")
            # 轮廓检测
            contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image_copy = image.copy()
            res = cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 3)
            # show(res, "res")
            # 膨胀2
            kernel = np.ones((1, 3), np.uint8)
            dilate2 = cv2.dilate(dilate, kernel, iterations=5)
            # show(dilate2, "dilate2")

            contours, hierarchy = cv2.findContours(dilate2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            resize_copy = image.copy()
            res3 = cv2.drawContours(resize_copy, contours, -1, (255, 0, 0), 2)
            # show(res3, "res3")
            # 筛选轮廓区域

            min_area = 100  # 最小面积
            max_area = 1000  # 最大面积
            filtered_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if min_area < area < max_area:
                    filtered_contours.append(contour)

            # 对轮廓进行排序
            def sort_contours(contours):
                boundingBoxes = [cv2.boundingRect(c) for c in contours]
                (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
                                                        key=lambda b: b[1][1], reverse=False))
                return contours

            sorted_contours = sort_contours(contours)

            # 获取第一个轮廓
            if sorted_contours:
                first_contour = sorted_contours[0]
                x, y, w, h = cv2.boundingRect(first_contour)
                self.first_contour_region = image[y:y + h, x:x + w]
                cv2.imwrite("first_contour_region.png", self.first_contour_region)

                a = os.system("tesseract.exe first_contour_region.png out  -l eng --psm 7")
                if a == 0:
                    with open('out.txt', 'r') as file:
                        line = file.readline()
                        self.result_label.setText(f'识别到学号: {line}')
                        self.view_region_button.setEnabled(True)
                else:
                    self.result_label.setText('识别失败')
            else:
                self.result_label.setText('未检测到轮廓')

    def view_region(self):
        a = cv2.imread('first_contour_region.png')
        cv2.imshow('a',a)
        cv2.waitKey(0)

    def display_image(self, image):
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        qImg = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(qImg)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(),
                                                 Qt.KeepAspectRatio))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = ImageProcessingUI()
    ui.show()
    sys.exit(app.exec_())