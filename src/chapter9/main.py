import imageio
import imageio.v2 as imageio
import matplotlib. pyplot as plt
import numpy as np

#RGB图片变为Gray图片
def rgb2gray(rgb):
    """
    rgb 2 gray
    Args:
        rgb image
    Returns:
        gray image
    """

    gray = rgb[:, : ,0]* 0.299 + rgb[:,:, 1] * 0.587 + rgb[:, :, 2] * 0.114
    return gray

# 1 read image
image = np. array(imageio.imread("moon.jpg")[:, :, 0:3])
# 2:convert rgb to gray image
image = rgb2gray(image)
# 3: dark area become, bright area become dark
invert_image = 255 - image

#二值化函数
def thre_bin(gray_image, threshold=170):
    """
    binary image
    Aras :
        gray image:image with gray scale
        threshold:the split standard
    Returns:
        bin image
    """
    threshold_image = np. zeros(shape= (image.shape[0], image.shape[1]), dtype= np. uint8)
    # loop for every pixel
    for i in range(gray_image. shape [0]) :
        for j in range(gray_image.shape[1]) :
            if gray_image[i][j] > threshold:
                threshold_image[i][j] = 1
            else:
                threshold_image[i][j] = 0
    return threshold_image

# 腐蚀
kernel: np.ones(shape=(5, 5))
def erode_bin_image(bin_image, kernel):
    """
    erode bin image
    Args:
       bin.image: image with 0, 1 pixel value
    Returns:
        erode image
    """
    kernel_size = kernel.shape[0]
    bin_image: np.array(bin_image)
    if (kernel_size%2 == 0) or kernel_size<1 :
        raise ValueError("kernel size must be odd and bigger than 1")
    if (bin_image. max() != 1) or (bin_image.min() != 0) :
        raise ValueError("input image s pixel value must be 0 or 1")
    d_image = np. zeros (shape=bin_image. shape)
    center_move = int ((kernel_size-1)/2)
    for i in range(center_move, bin_image. shape[0]-kernel_size+1):
        for j in range(center_move, bin_image. shape[1] -kernel_size+1):
            d_image[i, j] = np. min(bin_image[i-center_move:i+center_move,
                                    j-center_move:j+center_move] )
    return d_image

#膨胀
kernel = np. ones (shape=(13, 13))
def dilate_bin_image(bin_image, kernel):
    """"
    dilate bin image
    Args:
        bin_image: image with 0,1 pixel value
    Returns:
        dilate image
    """
    kernel_size = kernel. shape[0]
    bin_image = np. array(bin_image)
    if (kernel_size%2 == 0) or kernel. size<1:
        raise ValueError("kernel size must be odd and bigger than 1")
    if (bin_image. max() != 1) or (bin_image. min() != 0):
        raise ValueError("input image s pixel value must be 0 or 1")
    d_image = np. zeros (shape=bin_image. shape)
    center_move = int((kernel_size - 1)/2)
    for i in range(center_move, bin_image. shape[0]-kernel_size+1):
        for j in range(center_move, bin_image.shape[1]-kernel_size+1):
            d_image[i, j] = np. max(bin_image[i-center_move:i+center_move, j-center_move:j+center_move])
    return d_image

#程序运行
bin_image = thre_bin(invert_image)
plt. imshow(bin_image, cmap= "gray" )
erode_bin_image = erode_bin_image(bin_image,kernel)
# plt. imshow(erode_bin_image, cmap= "gray" )
dilate_bin_image = dilate_bin_image(bin_image,kernel)
# plt. imshow(dilate_bin_image, cmap= "gray" )

# #二值化图片显示
# plot_image = [image, invert_image]
# plot_title = ["original image" , "invert image"]
# plt. figure()
# for i in range(1, len(plot_image)+1):
#     plt. subplot(1, len(plot_image), i)
#     plt. imshow(plot_image[i-1], cmap= "gray")
#     plt. title(plot_title[i-1])
# plt. show()

#二值化图片显示
plot_image = [image, invert_image,erode_bin_image,dilate_bin_image]
plot_title = ["original image" , "invert image","erode image","dilate image"]
plt. figure()
for i in range(1, len(plot_image)+1):
    plt. subplot(1, len(plot_image), i)
    plt. imshow(plot_image[i-1], cmap= "gray")
    plt. title(plot_title[i-1])
plt. show()

