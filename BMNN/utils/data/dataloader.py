from torch.utils.data import DataLoader
class AQDataloader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AQDataloader, self).__init__(*args, **kwargs)