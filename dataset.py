import torch

class Dataset():
    def __init__(self, batch_size, negatives = False):
        self.dataset = torch.tensor([
            [0.,8.],[1.,8.],[2.,8.],[3.,8.],[4.,8.],
            [0.,7.],[1.,7.],[2.,7.],[3.,7.],[4.,7.],
            [0.,6.],[1.,6.],[2.,6.],[3.,6.],[4.,6.],
            [0.,5.],[1.,5.],[2.,5.],[3.,5.],[4.,5.],
            [0.,4.],[1.,4.],[2.,4.],[3.,4.],[4.,4.],
            [0.,3.],[1.,3.],[2.,3.],[3.,3.],[4.,3.],
            [0.,2.],[1.,2.],[2.,2.],[3.,2.],[4.,2.],
            [0.,1.],[1.,1.],[2.,1.],[3.,1.],[4.,1.]
        ])

        if negatives: 
            self.labels = torch.tensor(  [-1.,-1.,-1.,-1.,-1.,
                         -1.,-1.,-1.,-1.,-1.,
                         -1.,-1.,-1.,-1.,-1.,
                         -1.,-1.,1.,-1.,-1.,
                         1.,1.,1.,-1.,1.,
                         1.,1.,-1.,1.,1.,
                         1.,1.,1.,1.,1.,
                         1.,1.,1.,1.,1.])
        else:

            self.labels = torch.tensor(
            [0.,0.,0.,0.,0.,
            0.,0.,0.,0.,0.,
            0.,0.,0.,0.,0.,
            0.,0.,1.,0.,0.,
            1.,1.,1.,0.,1.,
            1.,1.,0.,1.,1.,
            1.,1.,1.,1.,1.,
            1.,1.,1.,1.,1.]
        )


        
        self.batch_size = batch_size

        self.train_data = []
        self.train_data0 = []
        self.train_data1 = []

        for i in range(len(self.dataset)):
            self.train_data.append([self.dataset[i], self.labels[i]])
            if self.labels[i] == (0 if not negatives else -1):
                self.train_data0.append([self.dataset[i], self.labels[i]])
            elif self.labels[i] == 1:
                self.train_data1.append([self.dataset[i], self.labels[i]])

    def get_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size = self.batch_size,
            shuffle = True
        )
    
    def get_separated_dataloaders(self):
        dl1 = torch.utils.data.DataLoader(
            self.train_data1,
            batch_size = self.batch_size,
            shuffle = True
        )
        dl0 = torch.utils.data.DataLoader(
            self.train_data0,
            batch_size = self.batch_size,
            shuffle = True
        )

        return dl1, dl0
