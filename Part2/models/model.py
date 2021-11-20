import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import timm

def build_model(n_classes, device, args):
    if args.model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif args.model_type.startswith('efficientnet'):
        model = EfficientNet.from_pretrained("efficientnet-b" + str(args.model_type.split('efficientnet')[1]), num_classes=n_classes)
    elif args.model_type == "cspresnext50":
        model = timm.create_model('cspresnext50', pretrained=True, num_classes=n_classes)
    model = model.to(device)
    return model