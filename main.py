import argparse
from skimage.measure import compare_ssim
from pylab import *
import cv2
from numpy import average, dot, linalg

# global point1, point2, cut_img, img


def de_mean(x):  # 均值作差
    xmean = mean(x)
    return [xi - xmean for xi in x]


# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)  # 计算像素平均值
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def on_mouse(event, x, y, flags, param):
    global point1, point2, cut_img, img
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        point1 = (x, y)
        cv2.circle(img2, point1, 10, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (0, 0, 255), 5)
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        cut_img = img[min_y:min_y + height, min_x:min_x + width]


def get_cut(img_1):
    global img, cut_img

    img = img_1.copy()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)

class metric:

    def __init__(self, source, target):
        self.__source_img = source
        self.__target_img = target
        self.__source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        self.__target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    def covariance(self):  # 对数平方/总个数-1,协方差
        n = len(self.__source_gray.flatten())

        return dot(de_mean(self.__source_gray.flatten()), de_mean(self.__target_gray.flatten())) / (n - 1)

    def SSIM(self):
        (score, diff) = compare_ssim(self.__source_gray, self.__target_gray, full=True)
        diff = (diff * 255).astype("uint8")

        return score

    # 计算图片的余弦距离
    def image_similarity_vectors_via_numpy(self):
        image1 = self.__source_gray
        image2 = self.__target_gray
        images = [image1, image2]
        vectors = []
        norms = []
        for image in images:
            vector = []
            for pixel_tuple in image.flatten():
                vector.append(average(pixel_tuple))
            vectors.append(vector)
            # linalg=linear（线性）+algebra（代数），norm则表示范数
            # 求图片的范数？？
            norms.append(linalg.norm(vector, 2))
        a, b = vectors
        a_norm, b_norm = norms
        # dot返回的是点积，对二维数组（矩阵）进行计算
        res = dot(a / a_norm, b / b_norm)
        return res

    def hist_similar(self):
        # 计算图img的直方图
        H1 = cv2.calcHist([self.__source_img], [1], None, [256], [0, 256])
        H1 = cv2.normalize(H1, H1, 0, 1, cv2.NORM_MINMAX, -1)  # 对图片进行归一化处理

        # 计算图img2的直方图
        H2 = cv2.calcHist([self.__target_img], [1], None, [256], [0, 256])
        H2 = cv2.normalize(H2, H2, 0, 1, cv2.NORM_MINMAX, -1)

        # 利用compareHist（）进行比较相似度
        similarity = cv2.compareHist(H1, H2, 0)
        return similarity

    # 计算汉明距离
    def Hamming_distance(self):
        # 调整到8*8
        img1 = cv2.resize(self.__source_gray, (8, 8))
        img2 = cv2.resize(self.__target_gray, (8, 8))

        # 获取哈希
        hash1 = getHash(img1)
        hash2 = getHash(img2)

        num = 0
        for index in range(len(hash1)):
            if hash1[index] != hash2[index]:
                num += 1
        return num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='a.jpg', help='source img')
    parser.add_argument('--target', type=str, default='b.bmp', help='target img')


    opt = parser.parse_args()

    global point1, point2, cut_img, img
    source = opt.source
    target = opt.target
    source_img = cv2.imread(source)
    target_img = cv2.imread(target)
    if source_img.shape == target_img.shape:#git 修改实例
        m = metric(source_img,target_img)   #git修改1.2
    else:
        source_img = cv2.resize(source_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        target_img = cv2.resize(target_img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)

        get_cut(source_img)
        source_img = cut_img

        get_cut(target_img)
        target_img = cut_img

        target_img = cv2.resize(target_img,(source_img.shape[1],source_img.shape[0]))
        m = metric(source_img,target_img)

        score = m.SSIM()
        print("SSIM: {}".format(score))

        cosin = m.image_similarity_vectors_via_numpy()
        print('余弦相似度', cosin)

        hist_score = m.hist_similar()
        print("calc: {}".format(hist_score))

        Hamming_distance = m.Hamming_distance()
        print("Hamming_distance: {}".format(Hamming_distance))