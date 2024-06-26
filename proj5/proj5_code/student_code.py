from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm


def center(a_tensor):
    squeeze_atensor = torch.squeeze(a_tensor)
    begin = end = 15
    center_tensor = squeeze_atensor[:, begin:-end, begin:-end]
    return torch.unsqueeze(center_tensor, -1)


def commplex_exp_torch(phase, dtype=torch.complex128):
    """
    通过commplex_exp_torch函数将phase转换为复数，这个复数的实部是cos(phase)，虚部是sin(phase)
    
    Useful functions:
    -   type(torch.float64)
    -   torch.complex()
    -   torch.cos()
    -   torch.sin()
    """
    return torch.complex(torch.cos(phase), torch.sin(phase))


def ifftshift2d_tf(a_tensor):
    """
    通过ifftshift2d_tf实现tensor的第二和第三维度的逆傅里叶移位(ifftshift)操作

    Useful functions:
    -   torch.index_select()
    """
    # (B, H, W, C)
    import math
    _, H, W, _ = a_tensor.size()
    H_split, W_split = math.ceil(H / 2), math.ceil(W / 2)
    a_tensor_up, a_tensor_down = a_tensor.split([H_split, H - H_split], dim=1)
    a_tensor = torch.cat([a_tensor_down, a_tensor_up], dim=1)
    a_tensor_left, a_tensor_right = a_tensor.split([W_split, W - W_split], dim=2)
    a_tensor = torch.cat([a_tensor_right, a_tensor_left], dim=2)
    return a_tensor


def transp_ifft2d(a_tensor, dtype=torch.complex64):
    """
    通过transp_ifft2d将tensor的第二、第三维度进行逆傅里叶变换

    Useful functions:
    -   torch.fft.ifft2()
    """
    # (B, H, W, C)
    return torch.fft.ifft2(a_tensor, dim=(1, 2))


def transp_fft2d(a_tensor, dtype=torch.complex64):
    """
    通过transp_fft2d将tensor的第二、第三维度进行傅里叶变换

    Useful functions:
    -   torch.fft.fft2()
    """
    # (B, H, W, C)
    return torch.fft.fft2(a_tensor, dim=(1, 2))


def psf2otf(input_filter, output_size):
    fh, fw, _, _ = input_filter.size()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = F.pad(torch.permute(input_filter, (2, 3, 0, 1)), [pad_top, pad_bottom,
                                                                   pad_left, pad_right, 0, 0, 0, 0], mode="constant")
        
        padded = torch.permute(padded, (2, 3, 0, 1))
    else:
        padded = input_filter
    return padded





class MyDataset(Dataset):
    def __init__(self, train=True, transform=None):
        train_data_path = '../data/mnist10-train-data.npy'
        train_label_path = '../data/mnist10-train-label.npy'

        test_data_path = '../data/mnist10-test-data.npy'
        test_label_path = '../data/mnist10-test-label.npy'

        self.train = train
        self.transform = transform
        
        self.train_data = np.load(train_data_path)
        self.train_label = np.load(train_label_path)

        self.test_data = np.load(test_data_path)
        self.test_label = np.load(test_label_path)

    def __getitem__(self, index):
        if self.train:
            self.label = self.train_label[index]
            self.data = self.train_data[index]

        else:
            self.label = self.test_label[index]
            self.data = self.test_data[index]
        
        return self.data, self.label
    
    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.test_data.shape[0]
        


otf_size = 142
padamt = 80

# 图像划分为12块，去掉中间的两块后剩下的10块对应10个数字，再将15块上下左右裁去10个像素，剩下大小为18*34
def img_split(img):
    splitted_1d = torch.stack(torch.chunk(img, 4, dim=1), 0)
    splitted = torch.concat(torch.chunk(splitted_1d, 3, dim=3), 0)
    
    result = torch.stack((center(splitted[0]),
                         center(splitted[1]),
                         center(splitted[2]),
                         center(splitted[3]),
                         center(splitted[5]),
                         center(splitted[6]),
                         center(splitted[8]),
                         center(splitted[9]),
                         center(splitted[10]),
                         center(splitted[11])), 0)
    
    # 均值池化
    result = result.mean(dim=(2,3,4))
    result = torch.transpose(result, 0, 1)
    print(result.shape)
    
    return result

class onn(nn.Module):
    def __init__(self):
        super(onn, self).__init__()

        # height map
        self.height_map_var = torch.randn([otf_size, otf_size, 1, 1])
        self.height_map_var = self.height_map_var.div(1000)
        self.height_map_var = nn.Parameter(self.height_map_var)

        # parameters
        self.refractive_index = 1.5
        self.delta_N = self.refractive_index - 1.000277

        self.wave_lengths = 550e-9
        self.wave_nos = 2. * np.pi / self.wave_lengths

    def forward(self, x):
        #####################################################
        # height_map是一个随机的高度图，大小为[142, 142, 1, 1]
        # 用于调控光场相位，是DOE的制造参数，通过梯度下降优化
        height_map = torch.square(self.height_map_var)
        phi = self.wave_nos * self.delta_N * height_map
        phase_shifts = commplex_exp_torch(phi)
        atf = phase_shifts


        #####################################################
        # [bs, w, h, c]
        x = torch.reshape(x, [-1, 32, 32, 1])
        paddings = (0,0, padamt,padamt, padamt,padamt, 0,0)
        x = F.pad(x, paddings, "constant", 0)

        input_img = x
        img_shape = input_img.shape


        #####################################################
        target_side_length = 2 * img_shape[1]

        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img1 = F.pad(input_img, (0, 0, pad_top, pad_bottom, pad_left, pad_right, 0, 0), "constant", 0)
        img_shape = img1.shape



        #####################################################
        output_img1 = transp_fft2d(img1)
        output_img1 = ifftshift2d_tf(output_img1)

        otf1 = psf2otf(atf, output_size=img_shape[1:3])
        otf1 = otf1.transpose(0,1)
        otf1 = otf1.transpose(0,2)
        otf1 = otf1.to(torch.complex64)

        img_fft1 = output_img1.to(torch.complex64)
        result1 = transp_ifft2d(img_fft1 * otf1)
        result1 = torch.abs(result1).to(torch.float32) 


        output_img1 = result1[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

        #####################################################
        # return img_split(output_img1)
        return output_img1