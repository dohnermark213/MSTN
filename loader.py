import torch
import os 
from torchvision import datasets, transforms
from torch.utils.data  import DataLoader, Dataset


tt = transforms.Lambda(lambda x: print(x.size()))
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])

office_31 = {'amazon': 'office_31/amazon/images/',
             'dslr': 'office_31/dslr/images/',
             'webcam': 'office_31/webcam/images/'}

office_transform =  transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456,  0.406], [0.229, 0.224, 0.225])
                    ])

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

def webcam_loader(args): 
    webcam_data = datasets.ImageFolder(
                    office_31['webcam'],
                    transform=office_transform
                )
    train_size = int(0.8*len(webcam_data))
    test_size = len(webcam_data) - train_size 
    train, test = torch.utils.data.random_split(webcam_data, [train_size, test_size])

    train_loader =  DataLoader(train, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=True)
    return  train_loader, test_loader


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

class ImageLoader(Dataset):
    def __init__(self, root,  label, transform=None, loader=default_loader):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img =  self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
        
        return  img, target 

    def __len__(self):
        return  len(self.imgs)

    def split_data(data):
        train_loader=DataLoader(data,  batch_size = args.batch_size, shuffle=True)
        test_loader=DataLoader(data, batch_size = args.batch_size,  shuffle=True)
        return train_loader, test_loader 

    def office_loader(args, subset):
        data = ImageLoader('dataset/office/'+subset+'/','dataset/office/'+subset+'_label.txt')
        return split_data(data)


    def clef_loader(args, subsetm):
        data = ImageLoader('dataset/imageCLEF/'+subset+'/','dataset/imageCLEF/'+subset+'List.txt')
        return split_data(data)

    def  clef_c_loader(args):
        return office_loader(args, 'c')


