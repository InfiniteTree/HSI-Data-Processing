import numpy as np
import matplotlib.pyplot as plt
import cv2

import RemoveBG
import ReadData

def knn(img, iter, k):
    img = img.reshape(-1,3) # Rshape for reducing the loops
    #print(img)
    img_new = np.column_stack((img, np.ones(img_row*img_col))) # Add a new column
 
    # step 1 randomly choose clustering center point
    cluster_orientation = np.random.choice(img_row*img_col, k, replace=False) # k positions of index
    cluster_center = img_new[cluster_orientation, :] # get the pixle value according to the cluster_center position
 
    # Iteration
    distance = [ [] for i in range(k)] # [ [], [], [], [], []]生成list,每个元素是一个列向量，该列向量保存的是所有像素距离中心j的距离
    for i in range(iter):
        # (2) 计算所有像素与聚类中心j的颜色距离 step 2. Calculate all the value distances between the pixels and the cluster_center
        print("Iteration time：%d" % i)
        for j in range(k):
            distance[j] = np.sqrt(np.sum(np.square(img_new - cluster_center[j]), axis=1)) # data_new.shape = (269180,4)，一行的和
 
        # (3) 在当前像素与k个中心的颜色距离中，找到最小那个中心，更新图像所有像素label
        # np.array(distance).shape = (5, 269180) ，返回一列中最小值对应的索引,范围是 [0, 4], 代表不同的label
        orientation_min_dist = np.argmin(np.array(distance), axis=0)   # np.array(distance).shape = (5, 269180) 一列中最小值
        img_new[:, 3] = orientation_min_dist # shape = (269180, ), 将返回的索引列向量赋值给第4维，即保存label的第3列
        # (4) 更新第j个聚类中心
        for j in range(k):
            # np.mean(r,g,b,label)，属性和label都求个平均值
            one_cluster = img_new[img_new[:, 3] == j] # 找到所有label为j的像素,其中img_new.shape = (269180,4)
            cluster_center[j] = np.mean(one_cluster, axis=0) # 通过img_new[:, 3] == j找到所有label为j的行索引(?, 4)，
            # 求一列均值，这样mean_r ,mean_g_, mean_b, mean_label,一次循环得到(1,4)
    #print("cluster_center:",cluster_center)
    return cluster_center, img_new
 
def get_img(HSI_info, band1, band2, band3):
    samples =  HSI_info[0]
    lines = HSI_info[1]
    channels = HSI_info[2]
    HSI = HSI_info[3]
    HSI_npArray = np.array(HSI)
    img = np.zeros((lines,samples,3))
    # Show RGB img
    #Window = int(channels/30) # Default value: 10
    Window = 10
    for k in range(2*Window):
        img[:,:,0] += HSI_npArray[:,band1-Window+k,:]
        img[:,:,1] += HSI_npArray[:,band2-Window+k,:]
        img[:,:,2] += HSI_npArray[:,band3-Window+k,:]
    img[:,:,0] /= 2*Window 
    img[:,:,1] /= 2*Window
    img[:,:,2] /= 2*Window
    
    return img


if __name__ == "__main__":
    HSI_info = ReadData.ReadData()
    channels = HSI_info[2]
    print("---------------Start to get the Single img channel------------------\n")
    img = get_img(HSI_info,110,61,35)
    plantPos = RemoveBG.getPlant_pos(HSI_info) # a list of plant position
    '''
    print("---------------Start to get the background Pos-----------------\n")
    for i in range(HSI_info[0]):
        for j in range (HSI_info[1]):
            totalPos.append([i,j])
    print("---------------Got the Total Pos------------------\n")
    for i in range(HSI_info[0]):
        for j in range (HSI_info[1]):
            if [i,j] in totalPos and [i,j] not in plantPos:
                BGPos.append([i,j])
    print("---------------Start to remove the background------------------\n")

    emptyImage = np.zeros((HSI_info[0],HSI_info[1], 3), np.uint8)+255
    for i in range(HSI_info[0]):
        for j in range(HSI_info[1]):
            if [i,j] in plantPos:
                emptyImage[i][j] = img[i][j]
    print("---------------Finish to remove the background----------------\n")
    img = emptyImage
    print(img)
    '''
    #img = plt.imread('C:/Users/AlexChen/Desktop/MilletHill/m-CTP/m-CTP_Analysis/result4.jpg')
    #print(img)
    #plt.imshow(img.astype(np.uint8))
    #plt.show()
    
    img_row = img.shape[0]
    img_col = img.shape[1]

    print("---------------Start to do KNN method----------------\n")

    labels_result = knn(img, 50, 6)
    labels_centers = labels_result[0]
    labels_vector = labels_result[1]
    
    labels_img = labels_vector[:,3].reshape(img_row, img_col)

    print("labels_center is",labels_centers)
    #   print(labels_img)
    print("The shape of labels_img is",labels_img.shape)
    # Store the position(row, col) of the leaves in the img 

    plt.imshow(labels_img)
    plt.savefig("knn_cluster_result_windowsize=10.jpg")
    plt.show()



