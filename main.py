import cv2
import numpy as np
import os
from collections import defaultdict
from PIL import Image, ImageDraw
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt


'''
参考:
https://zhuanlan.zhihu.com/p/55479744
https://zhuanlan.zhihu.com/p/109971789
进行人脸对齐
'''


# global var
# 对齐后图像眼睛位置，归一化坐标
desiredLeftEye = (0.3, 0.3)
desiredFaceWidth = 256
desiredFaceHeight = 256

# 原始图像数据集路径和标签路径
images_path = r'.\data\images'
ldmk_path = r'.\data\ldmk'
# 对齐后图片和标签输出路径
output_image_path = r'.\data\aligned_images'
output_ldmk_path = r'.\data\aligned_ldmk'


# landmark可视化，标记为绿点
def visualize_landmark(img_array, landmarks):
    # # origin_img = Image.fromarray(img_array)
    # origin_img = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))      # cv2读出来的，默认是BGR的通道，而Image处理的是RGB
    # draw = ImageDraw.Draw(origin_img)
    # for facial_feature in landmarks.keys():
    #     # 函数调用,ImageDraw.point(xy, fill=None),xy–Sequence of either 2-tuples like [(x, y), (x, y), ...] or numeric values like [x, y, x, y, ...]
    #     draw.point(landmarks[facial_feature], fill=(255, 255, 0))
    #
    # imshow(origin_img)
    # plt.show()
    # draw.point()画出的点太小

    # 拷贝，避免在输出图片上标记绿点
    origin_img = img_array.copy()
    for facial_feature in landmarks.keys():
        # circle(img, center, radius, color, thickness=None, lineType=None, shift=None)
        cv2.circle(origin_img, center=(int(landmarks[facial_feature][0]), int(landmarks[facial_feature][1])), radius=3, color=(0, 255, 0), thickness=-1)
    imshow(Image.fromarray(cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)))     # cv2读出来的，默认是BGR的通道，而Image处理的是RGB
    plt.show()


# 人脸对齐
# （1）计算左右眼中心坐标连线与水平方向的夹角
# （2）将图片旋转对应角度
# （3）通过两眼间距进行放缩

def align_face(img_array, landmarks):
    left_eye_center = landmarks['left_eye_center']
    right_eye_center = landmarks['right_eye_center']
    # 以左右眼中心连线中点进行旋转
    eye_center = ((right_eye_center[0] + left_eye_center[0])//2, (right_eye_center[1] + left_eye_center[1])//2)
    # calculate angle
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))    # 弧度转角度

    # 以两眼间距计算缩放比例
    desiredRightEye = (1-desiredLeftEye[0], desiredLeftEye[1])
    DistEyes = np.sqrt((dx ** 2) + (dy ** 2))
    desiredDistEyes = (desiredRightEye[0] - desiredLeftEye[0]) * desiredFaceWidth
    scale = desiredDistEyes / DistEyes

    # the rotation matrix for rotating and scaling the face，仿射变换矩阵
    M = cv2.getRotationMatrix2D(eye_center, angle, scale)

    # update the translation component of the matrix
    # 旋转、放缩后，需要偏移图像，使得眉心位于预定位置
    # 为将两眼连线中心（眉心）平移到对齐后图像的预定位置，对M矩阵添加平移变换，再使用M矩阵进行仿射变换
    tX = desiredFaceWidth/2                         # 眉心在宽度的中间值，desiredFaceWidth = desiredFaceHeight = 256
    tY = desiredFaceHeight * desiredLeftEye[1]      # 眉心高度同左眼高度
    M[0, 2] += (tX - eye_center[0])
    M[1, 2] += (tY - eye_center[1])

    rotated_img = cv2.warpAffine(img_array, M, (desiredFaceWidth, desiredFaceHeight))

    # 注意：以上是图片旋转放缩，landmark未旋转放缩
    rotated_landmarks = defaultdict(tuple)
    for facial_feature in landmarks.keys():
        landmark = landmarks[facial_feature]
        # 构造输入向量
        A = np.array([landmark[0], landmark[1], 1])
        # 由M矩阵进行仿缩变换得到输出向量
        B = np.dot(M, A)
        # 取整,四舍五入
        B = np.around(B)
        # 变换后landmask
        rotated_landmark = (B[0], B[1])
        rotated_landmarks[facial_feature] = rotated_landmark

    return rotated_img, rotated_landmarks, eye_center, angle


def main(tag):
    files = os.listdir(images_path)     # 读入文件夹
    num_img = len(files)                # 统计文件夹中的文件个数
    print('There are ' + str(num_img) + ' images in the set')                      # 打印文件个数
    for file in files:
        # 图片路径
        n = file[6:11]
        img_path = images_path + r'\train_' + n + '.jpg'
        image_array = cv2.imread(img_path)                      # cv2.imread是BGR通道

        # 标签路径
        attri_path = ldmk_path + r'\train_' + n + '_manu_attri.txt'
        f = open(attri_path, encoding="utf-8")
        attributes = f.readlines()
        f.close()

        # 参考https://console.faceplusplus.com.cn/documents/5671270对于landmarks进行命名
        face_landmarks_dict = {
            'left_eye_center': tuple(map(float, attributes[0][:-1].split('\t'))),
            'right_eye_center': tuple(map(float, attributes[1][:-1].split('\t'))),
            'nose_tip': tuple(map(float, attributes[2][:-1].split('\t'))),
            'mouth_left_corner': tuple(map(float, attributes[3][:-1].split('\t'))),
            'mouth_right_corner': tuple(map(float, attributes[4][:-1].split('\t')))
            # 'gender': attributes[5][:-1],   # 0-male 1-female
            # 'race': attributes[6][:-1],     # 3种，白人，黑人，亚裔
            # 'age': attributes[7][:-1],      # 0-70岁，5 ranges
        }

        if tag:
            # 在原始图像上显示5个特征点
            visualize_landmark(img_array=image_array, landmarks=face_landmarks_dict)

        # 进行人脸对齐
        aligned_face, rotated_landmarks, eye_center, angle = align_face(img_array=image_array, landmarks=face_landmarks_dict)

        if tag:
            # 显示对齐后图像
            imshow(Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)))
            plt.show()
            # 显示对齐后图像及仿缩变换后的landmask
            visualize_landmark(img_array=aligned_face, landmarks=rotated_landmarks)

        # 保存对齐后图像
        aligned_image_path = output_image_path + r'\aligned_train_' + n + '.jpg'
        cv2.imwrite(aligned_image_path, aligned_face)

        # 保存对齐后标签
        aligned_label_path = output_ldmk_path + r'\aligned_train_' + n + '_manu_attri.txt'
        with open(aligned_label_path, "a") as f:
            f.seek(0)       # 移动指针到开头
            f.truncate()    # 清空文件, 防止多次运行重复写入

            # 逐行写入
            for item in rotated_landmarks.items():
                s = str(item[1][0]) + '\t' + str(item[1][1]) + '\n'
                f.write(s)
        f.close()

        print('image ' + n + ' is aligned!')


# True = 可视化
main(True)





# 注意：以上图片旋转放缩，landmark未旋转放缩
# def rotate(origin, point, angle, row):
#     x1, y1 = point
#     x2, y2 = origin
#     y1 = row - y1
#     y2 = row - y2
#     angle = math.radians(angle)
#     x = x2 + math.cos(angle)*(x1 - x2) - math.sin(angle) * (y1 - y2)
#     y = y2 + math.sin(angle)*(x1 - x2) + math.cos(angle) * (y1 - y2)
#     y = row - y
#     return int(x), int(y)
#
#
# def rotate_landmarks(landmarks, eye_center, angle, row):
#     rotated_landmarks = defaultdict(list)
#     for facial_feature in landmarks.keys():
#         for landmark in landmarks[facial_feature]:
#             rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
#             rotated_landmarks[facial_feature].append(rotated_landmark)
#     return rotated_landmarks
#
#
# rotated_landmarks = rotate_landmarks(landmarks=face_landmarks_dict, eye_center=eye_center, angle=angle, row=image_array.shape[0])
# visualize_landmark(img_array=aligned_face, landmarks=rotated_landmarks)


# # 人脸裁切
# # 根据旋转后的landmark剪裁人脸到固定尺寸
#
# def corp_face(image_array, size, landmarks):
#     left_eye_center = landmarks['left_eye_center'][0]
#     right_eye_center = landmarks['right_eye_center'][0]
#     eye_center = tuple(map(lambda x: 0.5*(x[0]+x[1]), zip(left_eye_center, right_eye_center)))
#     left_edge, right_edge = (eye_center[0] - size/2, eye_center[0] + size/2)
#
#     mouth_left_corner = landmarks['mouth_left_corner'][0]
#     mouth_right_corner = landmarks['mouth_right_corner'][0]
#     mouth_center = tuple(map(lambda x: 0.5*(x[0]+x[1]), zip(mouth_left_corner, mouth_right_corner)))
#
#     mid_part_dy = mouth_center[1] - eye_center[1]
#     top_edge = eye_center[1] - mid_part_dy*30/35        # 30/35 ??
#     bottom_edge = mouth_center[1] + mid_part_dy
#
#     pil_img = Image.fromarray(image_array)
#     top_edge, bottom_edge, left_edge, right_edge = [int(i) for i in [top_edge, bottom_edge, left_edge, right_edge]]
#     cropped_img = pil_img.crop((left_edge, top_edge, right_edge, bottom_edge))
#     cropped_img = np.array(cropped_img)
#     return cropped_img, left_edge, top_edge
#
#
# cropped_face, left_edge, top_edge = corp_face(image_array=aligned_face, size=280, landmarks=rotated_landmarks)
# cropped_img = Image.fromarray(cropped_face)
# imshow(cropped_img)
# plt.show()
