# import torch
# import random
#
# class RandomZeroReplace(object):
#     def __init__(self, replace_prob):
#         self.replace_prob = replace_prob
#
#     def __call__(self, sequence):
#         replaced_sequence = [
#             0 if random.random() < self.replace_prob else element
#             for element in sequence
#         ]
#         return replaced_sequence
#
# # Example usage
# original_sequence = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# replace_probability = 0.5
# random_zero_replace = RandomZeroReplace(replace_probability)
# replaced_sequence = random_zero_replace(original_sequence)
# print(replaced_sequence)
#
#
# print('best model save....')


import torch

# 加载模型文件
model_path = 'path_to_your_model_file.pt'  # 替换为你的模型文件路径
model_path = 'G:\\Learn\\Sample Code\\SEAD-Net\\experiments\\2023-11-08_09-10-32 - 1\\checkpoint_finetune.pt'
model_path_1 = 'G:\\Learn\\Sample Code\\SEAD-Net\\experiments\\2023-11-08_09-10-32 - 1\\checkpoint.pt'
state = torch.load(model_path)
state_1 = torch.load(model_path_1)

# 修改键名
if 'moco' in state_1:
    state_copy = state_1.copy()  # 创建state的副本
    state_copy['sclm'] = state_copy.pop('moco')
    # state['sclm'] = state.pop('moco')


# 保存修改后的模型文件
# torch.save(state, model_path)
torch.save(state_copy, model_path_1)