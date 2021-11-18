import torch
import torchvision

def build_model(n_classes, device, args):
    model = torchvision.models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, n_classes)
    model.to(device)
    return model