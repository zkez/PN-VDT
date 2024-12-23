import os
import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import rearrange_3dct_to_2d


class LungNoduleDataset(Dataset):
    def __init__(self, csv_data, data_dir, normalize=False, transform=None):
        """
        Args:
            csv_data (pd.DataFrame): 包含 ['Subject ID', 'study_yr', 'label'] 等信息的DataFrame
            data_dir (str): 存放 npy 数据文件的目录。
            normalize (bool): 是否对图像进行归一化。
            transform (callable): 对图像进行数据增强的transform。
        """
        self.csv_data = csv_data
        self.subject_ids = self.csv_data['Subject ID'].unique()
        self.data_dir = data_dir
        self.normalize = normalize
        self.transform = transform

        # 过滤有效样本
        valid_subject_ids = []
        for subject_id in self.csv_data['Subject ID'].unique():
            subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]

            # 获取每个时间点的文件路径
            T0_path = os.path.join(data_dir, f"{subject_id}_T0.npy")
            T1_path = os.path.join(data_dir, f"{subject_id}_T1.npy")
            T2_path = os.path.join(data_dir, f"{subject_id}_T2.npy")

            # 检查 CSV 数据是否齐全以及文件是否存在
            T0_row = not subject_data[subject_data['study_yr'] == 'T0'].empty
            T1_row = not subject_data[subject_data['study_yr'] == 'T1'].empty
            T2_row = not subject_data[subject_data['study_yr'] == 'T2'].empty
            files_exist = os.path.exists(T0_path) and os.path.exists(T1_path) and os.path.exists(T2_path)

            if T0_row and T1_row and T2_row and files_exist:
                valid_subject_ids.append(subject_id)

        self.subject_ids = valid_subject_ids
        print(f"Filtered dataset: {len(self.subject_ids)} valid samples remain")

    def __len__(self):
        return len(self.subject_ids)

    def normalize_image(self, image):
        """
        Normalize image to zero mean and unit variance.

        Args:
            image (numpy array): Image to be normalized.

        Returns:
            numpy array: Normalized image.
        """
        # mean = np.mean(image)
        # std = np.std(image)
        # if std > 0:
        #     image = (image - mean) / std
        # else:
        #     image = image - mean
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        return image

    def load_image(self, file_path):
        """
        Load an image from a given file path.

        Args:
            file_path (str): Path to the image file.

        Returns:
            numpy array: Loaded image.
        """
        return np.load(file_path).astype(np.float32)

    def __getitem__(self, idx):
        if idx >= len(self.subject_ids):
            raise IndexError(f"Index {idx} is out of range")

        subject_id = self.subject_ids[idx]
        subject_data = self.csv_data[self.csv_data['Subject ID'] == subject_id]

        # 获取三个时间点的数据行
        T0_row = subject_data[subject_data['study_yr'] == 'T0']
        T1_row = subject_data[subject_data['study_yr'] == 'T1']
        T2_row = subject_data[subject_data['study_yr'] == 'T2']

        # 如果任一时间点的数据不存在，跳过此样本
        if T0_row.empty or T1_row.empty or T2_row.empty:
            raise ValueError(f"Missing T0 or T1 or T2 data for subject {subject_id}")

        # 构建文件路径
        T0_path = os.path.join(self.data_dir, f"{subject_id}_T0.npy")
        T1_path = os.path.join(self.data_dir, f"{subject_id}_T1.npy")
        T2_path = os.path.join(self.data_dir, f"{subject_id}_T2.npy")

        # 加载数据
        T0_image = self.load_image(T0_path)
        T1_image = self.load_image(T1_path)
        T2_image = self.load_image(T2_path)

        # 任意缺失则抛出异常或返回None以跳过
        if T0_image is None or T1_image is None or T2_image is None:
            raise ValueError(f"Missing npy file for subject {subject_id}")

        label = T2_row.iloc[0]['label']
        label = int(label)

        if self.normalize:
            T0_image = self.normalize_image(T0_image)
            T1_image = self.normalize_image(T1_image)
            T2_image = self.normalize_image(T2_image)

        if self.transform is not None:
            T0_image = self.transform(T0_image)
            T1_image = self.transform(T1_image)
            T2_image = self.transform(T2_image)

        # 添加channel维度 [C=1, D, H, W]
        T0_3Dimage = torch.tensor(T0_image, dtype=torch.float32).unsqueeze(0) # [1, D, H, W]
        T1_3Dimage = torch.tensor(T1_image, dtype=torch.float32).unsqueeze(0)
        T2_3Dimage = torch.tensor(T2_image, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.float32)

        # 将3D CT拼接成2D CT
        T0_image = rearrange_3dct_to_2d(T0_3Dimage)
        T1_image = rearrange_3dct_to_2d(T1_3Dimage)
        T2_image = rearrange_3dct_to_2d(T2_3Dimage)

        video_FCHW = torch.cat([T0_image, T1_image, T2_image], dim=0).unsqueeze(1) # [3, C, H, W]
        video_CFHW = video_FCHW.permute(1, 0, 2, 3)

        return video_FCHW, video_CFHW, label
    