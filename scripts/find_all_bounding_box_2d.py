from PIL import Image
import argparse
import numpy as np
from numpy import ndarray
from typing import *
from collections import deque
import logging
import numpy as np
from scipy.ndimage import label, find_objects
import torch

def find_bounding_boxes_and_split_labels(binary_array) -> List[List]:
    """ List[List] in xyxy form """
    # 标记连通域
    labels, num_features = label(binary_array)

    # 找到每个连通域的切片
    regions = find_objects(labels)

    # 打印每个连通域的边界框
    bounding_boxes = []
    for i, region in enumerate(regions):
        # region 是一个 slice 对象，分别表示行和列的范围
        row_slice, col_slice = region
        bounding_boxes.append([row_slice.start, col_slice.start, row_slice.stop - 1, col_slice.stop - 1]) 
    
    outputs = np.zeros_like(labels)
    outputs = np.expand_dims(outputs, axis=0)
    outputs = np.repeat(outputs, repeats=num_features, axis=0)
    for i in range(num_features):
        outputs[i][labels == (i + 1)] = 1

    return bounding_boxes, outputs



def find_bounding_boxes(binary_array: ndarray) -> List[List]:
    """ List[List] in xyxy form """
    # 标记连通域
    labels, num_features = label(binary_array)

    # 找到每个连通域的切片
    regions = find_objects(labels)

    # 打印每个连通域的边界框
    bounding_boxes = []
    for i, region in enumerate(regions):
        # region 是一个 slice 对象，分别表示行和列的范围
        row_slice, col_slice = region
        bounding_boxes.append([row_slice.start, col_slice.start, row_slice.stop - 1, col_slice.stop - 1]) 
    
    return bounding_boxes


def find_bounding_boxes_and_masks(binary_array: ndarray) -> Tuple[List[List], List[ndarray]]:  
    """ List[Tuple] in xyxy form """
    bounding_boxes = []
    masks = []
    dx, dy = (0, 1, 0, -1), (1, 0, -1, 0)
    st = np.zeros_like(binary_array)
    q = deque()
    cnt = 0
    n, m = binary_array.shape
    for i in range(n):
        for j in range(m):
            if not st[i][j] and binary_array[i][j]: 
                min_x = min_y = 1e9
                max_x = max_y = 0
                mask = np.zeros_like(binary_array).astype(np.uint8)
                q.append([i, j])
                st[i, j] = 1
                while q:
                    x, y = q.popleft()
                    mask[x, y] = 1
                    min_x = min(min_x, x); min_y = min(min_y, y)
                    max_x = max(max_x, x); max_y = max(max_y, y)
                    for k in range(4):
                        xx, yy = x + dx[k], y + dy[k]
                        if xx >= 0 and xx < n and yy >= 0 and yy < m and not st[xx][yy] and binary_array[xx][yy]:
                            q.append([xx, yy])
                            st[xx, yy] = 1
            
                cnt += 1
                bounding_boxes.append([min_x, min_y, max_x, max_y])    
                masks.append(mask)
            
                
    print(f"Find {cnt} bounding box!")

    return bounding_boxes, masks


if __name__ == "__main__":
    i = torch.Tensor([[1, 0, 0, 1], [1, 1, 1, 1]])
    find_bounding_boxes_and_split_labels(i)
    