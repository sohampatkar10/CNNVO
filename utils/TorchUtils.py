from torchvision import transforms, utils


class ToTensor:
    """
    Convert ndarrays in sample to Tensors.
    (Shamelessly stolen from `https://pytorch.org/tutorials/beginner/data_loading_tutorial.html`)
    """

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image)
