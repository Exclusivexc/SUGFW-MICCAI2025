import os
s = "/media/ubuntu/maxiaochuan/SAM_adaptive_learning/sam_data/Promise12/images/train"
print(len(os.listdir(s)), len(os.listdir(s.replace("train", "valid"))), len(os.listdir(s.replace("train", "test"))))