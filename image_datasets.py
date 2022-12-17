import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, CIFAR10
from torchvision import transforms
from torchvision.datasets.folder import *

class FilterableImageFolder(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            valid_classes: List = None
    ):
        self.valid_classes = valid_classes
        super(FilterableImageFolder, self).__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        #增加了这下面这句
        if self.valid_classes is not None:
            classes = [valid_class for valid_class in classes if valid_class in self.valid_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def get_loader(
        data_dir,
        batch_size,
        image_size,
        random_flip=True,
        is_gray=False
):
    """
        For a dataset, create a generator over (images, kwargs) pairs.
        :param data_dir: a dataset directory.
        :param batch_size: the batch size of each returned pair.
        :param image_size: the size to which images are resized.
        :param random_flip: if True, randomly flip the images for augmentation.
        :param is_gray: if True, convert images to grayscale
        """
    tf = [transforms.Resize(image_size), transforms.ToTensor()]
    if random_flip:
        tf = [transforms.RandomHorizontalFlip()] + tf
    if is_gray:
        tf = [transforms.Grayscale(1)] + tf
    tf = transforms.Compose(tf)
    dataset = FilterableImageFolder(
        root=data_dir, transform=tf)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


if __name__ == "__main__":
    loader = get_loader("./dataset/images", 1, 256, is_gray=True)
    # tf = [transforms.ToTensor()]
    # tf = transforms.Compose(tf)
    # train_set = CIFAR10("./CIFAR10", train=True, download=False, transform=tf)
    # loader = DataLoader(
    #     train_set,
    #     batch_size=1,
    #     num_workers=0,
    # )
    for x, c in loader:
        print(c.cpu().numpy())
        # from torchvision.utils import save_image
        # save_image(x, "./a.png")
        break


