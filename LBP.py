from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# settings for LBP
radius = 4
n_points = 8 * radius
# 乘8源于8-邻域法
# Method = 'uniform'
Method = 'uniform'
# 不同method区别见https://blog.csdn.net/hyk_1996/article/details/79619269
# 使用等价模式特征，可以有效进行数据降维，而对模型性能却无较大影响

# 读取图像
image = cv2.imread(r'.\data\aligned_images\aligned_train_00136.jpg')

#显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(image1)

# 转换为灰度图显示
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(132)
plt.imshow(image, cmap='gray')

# 处理
lbp = local_binary_pattern(image, n_points, radius, Method)

plt.subplot(133)
plt.imshow(lbp, cmap='gray')
plt.show()


# 统计直方图,直接统计
'''
对于八采样点的LBP算子来说，特征值范围为0~255，对每个特征值进行统计，比如得到特征值为1的LBP值有多少个、特征值为245的LBP值有多少个等等。
这样就形成了一个直方图，该直方图有256个bin，即256个分量，也可以把该直方图当做一个长度为256的向量。

如果直接使用该向量的话，那么对八采样点的LBP算子来说，一张图片至多会形成一个256长度的一个向量，这样位置信息就全部丢失了，会造成很大的精度问题。
所以在实际中还会再有一个技巧，就是先把图像分成若干个区域，对每个区域进行统计得到直方图向量，
再将这些向量整合起来形成一个大的向量。下图就将一张人脸图像分成了7x7的子区域。
'''


# 对于Uniform LBP共有2+2(p-1)+(p-1)(p-2)=p(p-1)+2种模式可能
# mode_num = n_points * (n_points - 1) + 2

# scikit-image 文档给出的lbp uniform模式个数计算方法 = p + 2, lbp中元素的范围0-25，共26 = 24 + 2种
mode_num = n_points + 2

# lbp中元素的范围0-25
x = np.arange(mode_num)
# 统计个数
y = np.zeros(mode_num)
for i in lbp:
    for j in i:
        y[int(j)] = y[int(j)] + 1

y = y/(lbp.shape[0]*lbp.shape[1])
print(y)

plt.bar(x, y)
plt.show()


# np.set_printoptions(threshold = 1e6)
# def Lbp(image):
#     l = np.zeros(256, dtype=int)
#     for a in range(6):
#         for b in range(18):
#             l[image[a][b]] = l[image[a][b]] + 1
#     lbp2 = np.array(l)
#     lbp2 = lbp2.reshape(1, -1)
#     return lbp2
#
#
# LBP_zhifangtu = Lbp(image)
# print(LBP_zhifangtu)






# reference
'''
https://blog.csdn.net/heroacool/article/details/53516032

https://www.jianshu.com/p/8d96ceb45f74

https://blog.csdn.net/zouxy09/article/details/7929531

https://senitco.github.io/2017/06/12/image-feature-lbp/

https://blog.csdn.net/u012679707/article/details/80671941

https://blog.csdn.net/qianqing13579/article/details/49406563

https://github.com/huangchuchuan/SVM-LBP-picture-classifier

https://www.geek-share.com/detail/2723050451.html
'''

