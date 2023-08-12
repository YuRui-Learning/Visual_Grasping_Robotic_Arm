import numpy as np
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [1, 2, 3]])

zero_matrix = np.zeros(len(matrix))
new_matrix = np.insert(matrix, len(matrix[0]), zero_matrix, axis=1)
new_matrix = np.insert(new_matrix, len(new_matrix[0]), zero_matrix, axis=1)
print(new_matrix)

import numpy as np

# 创建示例矩阵
matrix = np.array([[12, 8, 21],
                   [231, 5, 8],
                   [213, 6, 9]])

# 按第一列数据进行排序
sorted_matrix = matrix[matrix[:, 2].argsort()]

print(sorted_matrix)

my_dict = {'key1': True, 'key2': True, 'key3': True}

all_true = all(value for value in my_dict.values())

if all_true:
    print("字典中的所有值都为 True")
else:
    print("字典中的值存在 False 或其他非真值")