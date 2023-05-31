import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
from sklearn.cluster import KMeans

import ReadData

def knn(img, iter, k):
    img = get_HSVimg(img)[0]
    img_row = img.shape[0]
    img_col = img.shape[1]
    img = img.reshape(-1,3) # Rshape for reducing the loops

    # Remove background pixels from the clustered image
    bg_indices = np.where(np.all(img == [0, 0, 0], axis=-1))[0]
    img_new = img[np.logical_not(np.isin(np.arange(img_row*img_col), bg_indices)), :]
    #print(img_new.shape)

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

def get_RGBimg(HSI_info, band1, band2, band3):
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


def get_HSVimg(RGBimg):
    #rgb_image = RGBimg.convert('RGB')
    rgb_array = np.array(RGBimg)

    rgb_array = rgb_array / 255.0
    h, s, v = np.vectorize(colorsys.rgb_to_hsv)(rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2])
    h = h * 360
    s = s * 255
    v = v * 255
    hsv_array = np.dstack((h, s, v)).astype(np.uint8)
    hsv_image = Image.fromarray(hsv_array, mode='HSV')
    '''
    # Perform k-means clustering
    hsv_flat = hsv_array.reshape((-1, 3))
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(hsv_flat)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    # Reshape labels back to the original image shape
    labels = labels.reshape(hsv_array.shape[:2])
    '''
    return hsv_array, hsv_image

if __name__ == "__main__":
    test = 1
    match test:
        case 1:
            HSI_info = ReadData.Read()
            wavelengths = HSI_info[4]
            lines = HSI_info[0]
            channels= HSI_info[1]
            samples = HSI_info[2]
            PixelSum = lines * samples

            #img = cv2.imread("figures/wheat/pre_processing/Level2_img_new.jpg") 
            img = cv2.imread("figures/wheat/test.jpg")
            print("---------------Start to do KNN method----------------")
            knn_center_num = 2
            iter_time = 50
            labels_result = knn(img, iter_time, knn_center_num) # Clustering for 3 classes: 1) Useless info; 2) Normal blades info; 3) abnormal blades;
            labels_centers = labels_result[0]

            labels_vector = labels_result[1]
            img_row = img.shape[0]
            img_col = img.shape[1]
            labels_img = labels_vector[:,3].reshape(img_row, img_col)
            
            plt.imshow(labels_img)
            plt.savefig("figures/wheat/Clustering/HSV_knn_cluster_n={0}.jpg".format(knn_center_num))
            plt.show()

            # Get the indices of all pixels labeled as "abnormal blades" (class 2)
            PixelSum = img_row * img_col
            for i in range(knn_center_num):
                print("When labels_img ==",i)
                abnormal_indices = np.where(labels_img == i)
                # Convert the indices to (x,y) coordinates
                abnormal_pixels = np.column_stack((abnormal_indices[0], abnormal_indices[1]))
                # Get the proportion of the abnormal pixels
                abnormal_proportion = len(abnormal_pixels) / PixelSum
                #print(abnormal_pixels)
                print(abnormal_proportion)
            
            normal_indices = np.where(labels_img == 1)
            abnormal_indices = np.where(labels_img == 2)

            # Convert the indices to (x,y) coordinates
            normal_pixels = np.column_stack((normal_indices[0], normal_indices[1]))
            abnormal_pixels = np.column_stack((abnormal_indices[0], abnormal_indices[1]))

        case 2:
            #img = Image.open("figures/wheat/pre_processing/Level2_img_new.jpg")
            '''
            img = Image.open("figures/wheat/test.jpg")
            print(type(img))
            '''
            img = cv2.imread("figures/wheat/test.jpg")
            print(type(img))
            
            print(img.shape)
            print(img.shape[0])
            get_HSVimg(img)



