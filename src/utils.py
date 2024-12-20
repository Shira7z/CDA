import torchvision.transforms as T

# Function to get transformations
def get_transform(image_size):
    return T.Compose([
        T.ToTensor(),
        T.Resize((image_size, image_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
