import numpy as np
np.set_printoptions(threshold=np.inf)  ## print all values of matrix without reduction
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.spatial import distance     ## calculate the distance between two points

iris = datasets.load_iris()
iris_data = iris.data
iris_label = iris.target
# print(iris_data)
# print(iris_label)

column = [2,3]
iris_data = iris_data[:,column]
# print(iris_data)


# print(iris_label.shape[0])
for i in range(iris_label.shape[0]):
    if iris_label[i] == 0:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='red',s=50,alpha=0.6)
    elif iris_label[i] == 1:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='green',s=50,alpha=0.6)
    elif iris_label[i] == 2:
        plt.scatter(iris_data[i,0],iris_data[i,1],color='blue',s=50,alpha=0.6)
# plt.show()


K = 5
class_num = 3 # Divided into three categories
class_count = [0,0,0] # Ballot box to see which one is the highest
test_point =[5, 1.7]
dis_array = []


# Calculate all distances
for i in range(iris_label.shape[0]):
    dst = distance.euclidean(test_point,iris_data[i, :])
    dis_array.append(dst)
# print(dis_array)


idx_sort = np.argsort(dis_array)[0:K] # Only return the first five
# print(idx_sort)

for i in range(K): # Just need to know the top K nearest ones
    label = iris_label[ idx_sort[i] ] # Find out which category the top five are in and vote
    class_count[label] += 1  
# print(class_count)

result = np.argsort(class_count)[-1] # argsort is sorted from small to large. -1 means grabbing the largest index.
print(result)

if result == 0:
    plt.scatter(test_point[0], test_point[1],
                color='red',s=150,alpha=1,marker='^')
elif result == 1:
    plt.scatter(test_point[0], test_point[1],
                color='green',s=150,alpha=1,marker='^')
elif result == 2:
    plt.scatter(test_point[0], test_point[1],
                color='blue',s=150,alpha=1,marker='^')
    
plt.grid()
plt.show()























    
























# for i in range(K): # Just need to know the top K nearest ones
#     label = iris_label[ idx_sort[i] ] # Find out which category the top five are in and vote
#     class_count[label] += 1    
    
# result = np.argsort(class_count)[-1] # argsort is sorted from small to large. -1 means grabbing the largest index.
# print(result)

# if result == 0:
#     plt.scatter(test_point[0], test_point[1],
#                 color='red',s=150,alpha=1,marker='^')
# elif result == 1:
#     plt.scatter(test_point[0], test_point[1],
#                 color='green',s=150,alpha=1,marker='^')
# elif result == 2:
#     plt.scatter(test_point[0], test_point[1],
#                 color='blue',s=150,alpha=1,marker='^')
    
# plt.grid()
# plt.show()