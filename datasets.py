import torch
from torch.utils.data import Dataset
import pandas as pd
import os


class DigitalTwinDataset(Dataset):
    '''
        the dataset classs takes a directory with following structure:
        root_dir:
                -healthy
                -faulty
                    -fault_at_1
                    -fault_at_2
                    -fault_at_3
    '''

    def __init__(self, root_dir, sample_length=500, device=torch.device("cpu"), shuffle=True):

        super(DigitalTwinDataset, self).__init__()
        self.root_dir = root_dir
        self.sample_length = sample_length
        self.device = device
        self.shuffle = shuffle

        self.files = self._get_files()

        self.samples = self._get_samples()

        self.data, self.labels = self._get_data()
        if self.shuffle:

            # Shuffle the data and labels using the same permutation
            if len(self.data) != len(self.labels):
                raise ValueError("Data and labels must have the same length.")

            self.permutation = torch.randperm(len(self.data))
            self.data = [self.data[i] for i in self.permutation]
            self.labels = [self.labels[i] for i in self.permutation]

    def _get_files(self):
        root_dir = self.root_dir
        files = {}
        healthy_list = os.listdir(os.path.join(root_dir, 'healthy'))
        files['healthy'] = [os.path.join(
            root_dir, 'healthy', x) for x in healthy_list]
        faulty_folders = os.listdir(os.path.join(root_dir, 'faulty'))

        for folder in faulty_folders:
            faulty_list = []

            this_fault_list = os.listdir(
                os.path.join(root_dir, 'faulty', folder))
            this_fault_list = [os.path.join(
                root_dir+f"faulty/{folder}", x) for x in this_fault_list]
            faulty_list = faulty_list+this_fault_list
            files[folder] = faulty_list

        return files

    def _get_samples(self):

        samples = {x: [] for x in self.files.keys()}

        for folder in self.files.keys():

            for files in self.files[folder]:

                df = pd.read_csv(files)
                x1 = torch.tensor(df['X_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y1 = torch.tensor(df['Y_1 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x2 = torch.tensor(df['X_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y2 = torch.tensor(df['Y_2 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                x3 = torch.tensor(df['X_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)
                y3 = torch.tensor(df['Y_3 (Mean)'],
                                  dtype=torch.float64).reshape(-1, 1)

                data = torch.cat((x1, y1, x2, y2, x3, y3), 1)

                number_of_rows, _ = data.shape

                for x in range(0, number_of_rows, self.sample_length):
                    if x+self.sample_length <= number_of_rows:
                        samples[folder].append(data[x:x+self.sample_length])

        return samples

    def _get_data(self):
        data = []
        labels = []
        for label in self.samples:
            for labeled_data in self.samples[label]:
                if label == 'healthy':
                    labels.append(torch.tensor(0,  dtype=torch.float64))
                elif label == 'fault_at_2':
                    labels.append(torch.tensor(2, dtype=torch.float64))
                elif label == 'fault_at_1':
                    labels.append(torch.tensor(1, dtype=torch.float64))
                else:
                    labels.append(torch.tensor(3, dtype=torch.float64))

                data.append(labeled_data)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }

        return sample

    def split(self, ratio):
        # Calculate the sizes of each split based on the ratio
        split_size = int(ratio * len(self))

        # Splitting the data and labels
        split_one_data = self.data[:split_size]
        split_one_labels = self.labels[:split_size]

        split_two_data = self.data[split_size:]
        split_two_labels = self.labels[split_size:]

        # Create new instances of the dataset for the two splits
        split_one_dataset = DigitalTwinDataset(
            root_dir=self.root_dir,
            sample_length=self.sample_length,
            device=self.device,
            shuffle=self.shuffle
        )
        split_one_dataset.data = split_one_data
        split_one_dataset.labels = split_one_labels

        split_two_dataset = DigitalTwinDataset(
            root_dir=self.root_dir,
            sample_length=self.sample_length,
            device=self.device,
            shuffle=self.shuffle
        )
        split_two_dataset.data = split_two_data
        split_two_dataset.labels = split_two_labels

        return split_one_dataset, split_two_dataset


if __name__ == '__main__':

    test_data_path = "datasets/digital_twins/test/"
    data = DigitalTwinDataset(test_data_path)

    print(len(data.files))
    print(len(data.samples))
