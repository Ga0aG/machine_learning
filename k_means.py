import numpy as np
import copy
import matplotlib.pyplot as plt

def kmeans_wave(n, k, data):  # n为迭代次数， k为聚类数目， data为输入数据
    data_new = copy.deepcopy(data)
    data_new = np.column_stack((data_new, np.ones(len(data))))  
    center_point = np.random.choice(len(data), k, replace=False)    
    center = data_new[center_point,:]
    distance = [[] for i in range(k)]   
    for i in range(n):
         for j in range(k):
             distance[j] = np.sqrt(np.sum(np.square(data_new - np.array(center[j])), axis=1))     # 更新距离
         data_new[:,3] = np.argmin(np.array(distance), axis=0)   # 将最小距离的类别标签作为当前数据的类别
         for l in range(k):
             center[l] = np.mean(data_new[data_new[:,3]==l], axis=0)   # 更新聚类中心

    return data_new



pic = plt.imread('webwxgetmsgimg.jpeg')
data = pic.reshape(-1, 3)
data_new = kmeans_wave(100,5,data)
print(data_new.shape)
pic_new = data_new[:,3].reshape(300,400)   # 将多个标签展示出来
plt.imsave('k_means.png',pic_new)
