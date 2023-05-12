import numpy as np
import matplotlib.pyplot as plt
import cv2

import RemoveBG
import RemoveSD
import ReadData


def knn(img, iter, k):
    img_row = img.shape[0]
    img_col = img.shape[1]
    img = img.reshape(-1,3) # Rshape for reducing the loops

    # Remove background pixels from the clustered image
    bg_indices = np.where(np.all(img == [0, 0, 0], axis=-1))[0]
    img_new = img[np.logical_not(np.isin(np.arange(img_row*img_col), bg_indices)), :]
    print(img_new.shape)

    img_new = np.column_stack((img, np.ones(img_row*img_col))) # Add a new column

    # step 1. Randomly choose clustering center points
    cluster_orientation = np.random.choice(img_row*img_col, k, replace=False) # k positions of index
    cluster_center = img_new[cluster_orientation, :] # get the pixel value according to the cluster_center position

    # Iteration
    distance = [ [] for j in range(k)] # [ [], [], [], [], []]create a list,each element represents a column vector ，which stores its distance to the center pixel j
    for i in range(iter):
        # step 2. Calculate all the value distances between the pixels and the cluster_center
        print("Iteration time: %d" % i)
        for j in range(k):
            distance[j] = np.sqrt(np.sum(np.square(img_new - cluster_center[j]), axis=1))

        # step 3. Among all the centers points, find the closest one to the current pixel and update its pixel value same as the center
        orientation_min_dist = np.argmin(np.array(distance), axis=0)   # the min value in the column vector
        img_new[:, 3] = orientation_min_dist # return the 3th column of the labels

        # step 4. Update the jth clustering center
        for j in range(k):
            one_cluster = img_new[img_new[:, 3] == j] # find all the pixels with label j
            cluster_center[j] = np.mean(one_cluster, axis=0) # get the mean rgb values of pixels in one_cluster

    #print(img_new.shape[0],img_new.shape[1])
    return cluster_center, img_new

def get_img(HSI_info, band1, band2, band3):
    samples =  HSI_info[2]
    lines = HSI_info[0]
    channels = HSI_info[1]
    HSI = HSI_info[3]
    HSI_npArray = np.array(HSI)
    img = np.zeros((lines,samples,3))
    # Show RGB img
    #Window = int(channels/30) # Default value: 10
    Window = 10
    for k in range(2*Window):
        img[:,:,0] = HSI_npArray[:,band1-Window+k,:]
        img[:,:,1] = HSI_npArray[:,band2-Window+k,:]
        img[:,:,2] = HSI_npArray[:,band3-Window+k,:]

    return img


if __name__ == "__main__":
    HSI_info = ReadData.Read()
    wavelengths = HSI_info[4]
    lines = HSI_info[0]
    channels= HSI_info[1]
    samples = HSI_info[2]
    PixelSum = lines * samples
    '''
    ### Level 1
    Level_1 = RemoveBG.getPlantPos(HSI_info)
    HSI_1 = Level_1[0]
    BG_Counter = Level_1[2]
    proportion_1 = float((PixelSum - BG_Counter)/PixelSum)
    HSI_info_L1= [lines, channels, samples, HSI_1]

    ### Level 2
    set_value = [0, 0, 0]
    Level_2 = RemoveSD.RemoveSD(HSI_info_L1,set_value, proportion_1)
    HSI_2 = Level_2[0]
    HSI_info_L2= [lines, channels, samples, HSI_2]

    
    print("---------------Start to get the Single img channel------------------")
    #img = get_img(HSI_info_L2,109,59,34)
    '''
    img = cv2.imread("C:/Users/AlexChen/Desktop/HSI-Data-Processing/figures/Level2_img.jpg") 
    print("---------------Start to do KNN method----------------")
    labels_result = knn(img, 50, 4)
    labels_centers = labels_result[0]

    labels_vector = labels_result[1]
    img_row = img.shape[0]
    img_col = img.shape[1]
    labels_img = labels_vector[:,3].reshape(img_row, img_col)

    plt.imshow(labels_img)
    plt.savefig("figures/new_knn_cluster.jpg")
    plt.show()
    
