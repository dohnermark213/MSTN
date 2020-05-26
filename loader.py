import torch
import os 
from torchvision import datasets, transforms
from torch.utils.data  import DataLoader, Dataset


tt = transforms.Lambda(lambda x: print(x.size()))
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

def amazon_loader(args):
	amazon_data = datasets.ImageFolder(
			office_31['amazon'],
			transform=office_transform
		)
	train_size = int(0.8 * len(amazon_data))
	test_size = len(amazon_data) - train_size
	train, test = torch.utils.data.random_split(amazon_data, [train_size, test_size])

	train_loader = DataLoader(train, batch_size= args.batch_size, shuffle=True)
	test_loader = DataLoader(test, batch_size= args.batch_size, shuffle=True)
	return train_loader, test_loader

    class TransferLoader:
    def __init__(self, source, target):
        self.source = source
        self.target = target

        #self.func = func

    def __len__(self):
        return min(5,len(self.source)-1, len(self.target)-1)

    def __iter__(self):
        s = iter(self.source)
        t = iter(self.target)
        for _ in range(len(self)):
            yield (*s.next(), *t.next())


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


