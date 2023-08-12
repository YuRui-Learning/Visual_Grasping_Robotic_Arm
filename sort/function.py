import numpy as np

'''
先外后内，先大后小，然后根据排序依次遍历
所以排序优先级：内外圈参数 > 大小 > id
'''
def send_sort(data):
    count = 0 # 遍历排完序的结果，以免每次重复遍历
    store_list = []
    sorted_matrix = data[data[:, 7].argsort()] # 先根据id排
    sorted_matrix = sorted_matrix[sorted_matrix[:, 4].argsort()]  # 再根据大小排
    sorted_matrix = sorted_matrix[sorted_matrix[:, 8].argsort()]  # 再根据内外圈排
    print(sorted_matrix)
    for inout_index in range(2): # 内外圈参数
        for label_index in range(3):
            cache_list = []
            # 先看排完序的第一个是否符合要求再去遍历，否则直接跳出该列
            if count < len(data) and sorted_matrix[count][8] == inout_index and sorted_matrix[count][4] == label_index:
                # while时候要防止越界
                while(count < len(data) and sorted_matrix[count][8] == inout_index and sorted_matrix[count][4] == label_index ):
                    cache_list.append(sorted_matrix[count])
                    count += 1
            # 该类不适合分析下一类
            else:
                continue
            epoch_len = len(cache_list)  # 该圈数和label的所有结果
            fast_step = epoch_len // 2
            slow_step = epoch_len // 4
            flag = {} # 用来寄存标志位
            for i in range(epoch_len): flag[i] = False
            store_index = 0 # 从0开始存储
            i = 0
            while i < epoch_len:
                if i%2 ==0:
                    store_list.append(cache_list[store_index])
                    flag[store_index] = True
                    store_index += fast_step
                    store_index %= epoch_len
                    while flag[store_index] is True:
                        store_index += 1
                        store_index %= epoch_len
                        all_true = all(value for value in flag.values())
                        if all_true:
                            break
                else:
                    store_list.append(cache_list[store_index])
                    flag[store_index] = True
                    store_index += slow_step
                    store_index %= epoch_len
                    while flag[store_index] is True:
                        store_index += 1
                        store_index %= epoch_len
                        all_true = all(value for value in flag.values())
                        if all_true:
                            break
                i += 1
    return store_list
