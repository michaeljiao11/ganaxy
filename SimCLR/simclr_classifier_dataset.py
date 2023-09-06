class SUPERGALAXYDataGen(Dataset):
    def __init__(self, root, phase, num_classes):
        self.root = root
        self.phase = phase
        self.num_classes = num_classes
        self.randomcrop = transforms.RandomResizedCrop(32, (0.8, 1.0))
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])

        # Load the ImageFolder dataset and store it in self.ds
        self.ds = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image, label = self.ds[idx]

        if self.phase == 'train':
            image = self.randomcrop(image)

        image = self.preprocess(image)

        return image, label

    def preprocess(self, frame):
        frame = frame * 255.0  # Undo the ToTensor normalization
        frame = frame.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        frame = Image.fromarray(np.uint8(frame))  # Convert to PIL Image
        frame = frame.resize((32, 32))  # Resize the image to (32, 32)
        frame = np.array(frame)  # Convert back to NumPy array
        frame = frame.transpose((2, 0, 1))  # Convert to (C, H, W) format
        frame = frame / 255.0  # Normalize to [0, 1] range

        return frame
