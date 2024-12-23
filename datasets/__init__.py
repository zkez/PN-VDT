import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .dataloader import LungNoduleDataset

def get_dataset(config):
    csv_data = pd.read_csv(config.csv_path)
    data_dir = config.data_dir
    subject_ids = csv_data['Subject ID'].unique()
    train_ids, val_ids = train_test_split(subject_ids, test_size=config.spilt_size, random_state=42)
    train_data = csv_data[csv_data['Subject ID'].isin(train_ids)]
    val_data = csv_data[csv_data['Subject ID'].isin(val_ids)]
        
    train_dataset = LungNoduleDataset(train_data, data_dir, normalize=True)
    val_dataset = LungNoduleDataset(val_data, data_dir, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size, shuffle=False)

    return train_loader, val_loader
