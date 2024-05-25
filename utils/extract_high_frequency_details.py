import numpy as np
from scipy.fftpack import fft2, ifft2
import cv2  # 用于读取和显示图像，如果需要的话

def extract_high_frequency_details(image_gray):
    """
    提取图像的高频细节
    :param image_gray: 灰度图像，np.array格式
    :return: 高频细节增强的图像
    """
    # 快速傅里叶变换到频域
    fft_image = fft2(image_gray)
    
    # 获取图像的尺寸
    rows, cols = image_gray.shape
    
    # 创建一个中心在图像中心的掩码，用于保留高频部分
    # 假设低频在中心，我们将远离中心的区域视为高频信号
    mask = np.zeros((rows, cols), dtype=np.uint8)
    radius = min(rows, cols) // 2 - 5  # 这里的5是一个示例值，可以根据需要调整以控制高频区域的大小
    center = (cols // 2, rows // 2)
    cv2.circle(mask, center, radius, 1, -1)  # 使用cv2画圆来创建掩码
    
    # 应用掩码到频域图像上，只保留高频部分
    fft_image_masked = fft_image * mask
    
    # 快速傅里叶逆变换回空间域
    high_freq_image = np.abs(ifft2(fft_image_masked))
    
    return high_freq_image

def extract_high_frequency_details_from_RGB(image_path=None, show=False):
    # 示例：读取图像并转换为灰度图
    image_path = r'F:\EIProject\DerainCycleGAN\datasets\rainy_Rain100L\trainB\norain-001.png'  # 请替换为你的图像路径
    image_color = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

    # 提取高频细节
    high_freq_detail_img = extract_high_frequency_details(image_gray)
    
    if show:
        # 显示原图和处理后的图像，如果需要
        cv2.imshow('Original Gray Image', image_gray)
        cv2.imshow('High Frequency Detail Image', high_freq_detail_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    return high_freq_detail_img
if '__name__' == '__main__':
    extract_high_frequency_details_from_RGB(image_path=None,show=True)
    