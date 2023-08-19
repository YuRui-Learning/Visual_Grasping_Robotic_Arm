from sklearn.linear_model import LinearRegression
import pca
import pandas as pd
import matplotlib.pyplot as plt
from numpy import *
import numpy as np



# 读取CSV文件
df = pd.read_csv('data.csv')
# 打印DataFrame
datamatrix = mat(np.array(df))
# pca
lowdata,recondata =pca.pca(datamatrix,1)
# get cx cy
cx = datamatrix[:,0].mean()
cy = datamatrix[:,1].mean()


def linearmodel(recondata):
    # inear model fit
    model = LinearRegression()
    x = np.array(recondata[:, 0])
    y = np.array(recondata[:, 1])
    model.fit(x, y)
    # 获取拟合参数
    slope = model.coef_[0]
    slope_v = -1 / slope
    return slope,slope_v

slope,slope_v = linearmodel(recondata)
k_dict = {"++":0,
          "+-":0,
          "--":0,
          "-+":0}

def line1(x,slope,cx,cy):
    return slope * (x - cx) + cy

def line2(x,slope_v,cx,cy):
    return slope_v * (x - cx) + cy

data = np.array(datamatrix)
dis_center = {"++":[],
              "+-":[],
              "--":[],
              "-+":[]}
index_center = {"++":[],
              "+-":[],
              "--":[],
              "-+":[]}
for id,data_item in enumerate(data):
    diff_line1 = line1(data_item[0],slope,cx,cy) - data_item[1]
    diff_line2 = line2(data_item[0],slope_v,cx,cy) - data_item[1]
    strflag = ""
    if diff_line1 < 0:
        strflag += "+"
    else:
        strflag += "-"
    if diff_line2 < 0:
        strflag += "+"
    else:
        strflag += "-"
    x_dis = data_item[0] - cx  # 横向距离
    y_dis = data_item[1] - cy  # 纵向距离
    distance = x_dis ** 2 + y_dis ** 2
    dis_center[strflag].append(distance)
    index_center[strflag].append(id)
    strflag += str(id)

    # ax.annotate(strflag, xy=(data_item[0], data_item[1]), xytext=(data_item[0], data_item[1]))


# plt
def plot(datamatrix,slope):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datamatrix[:,0].flatten().A[0],datamatrix[:,1].flatten().A[0],marker='^',s=90)
    x_min,x_max = min(datamatrix[:,0])-200,max(datamatrix[:,0])+200
    y_min,y_max = min(datamatrix[:,1])-200,max(datamatrix[:,1])+200
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])

    x = np.linspace(500, 1500, 10)
    # 计算y的值
    y = slope * (x - cx) + cy
    # 绘制直线
    plt.plot(x, y)


    x = np.linspace(500, 1500, 10)
    # get vertival
    slope_v = -1 / slope
    y = slope_v * (x - cx) + cy
    # 绘制直线
    plt.plot(x, y)

function_dict = {}





slope,slope_v = linearmodel(recondata)
plot(datamatrix,slope)

k_dict["++"],k_dict["+-"],k_dict["--"],k_dict["-+"] = slope,slope_v,slope,slope_v

for key in dis_center.keys():
    max_dict_index = dis_center[key].index(max(dis_center[key]))
    max_index = index_center[key][max_dict_index]
    a1,b = data[max_index]
    k = k_dict[key]
    x = np.linspace(500, 1500, 10)
    y = k * (x - a1) + b
    plt.plot(x, y)

plt.show()