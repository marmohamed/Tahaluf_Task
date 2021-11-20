import torch
import torchvision
from efficientnet_pytorch import EfficientNet
import timm

def build_model(n_classes, device, student, args):
    model_type = args.model_type
    if student:
      model_type = args.student_model_type
    if model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = torch.nn.Linear(num_ftrs, n_classes)
    elif model_type.startswith('efficientnet'):
        model = EfficientNet.from_pretrained("efficientnet-b" + str(model_type.split('efficientnet')[1]), num_classes=n_classes)
    elif model_type == "cspresnext50":
        model = timm.create_model('cspresnext50', pretrained=True, num_classes=n_classes)
    model = model.to(device)
    return model