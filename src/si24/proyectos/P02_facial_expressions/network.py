import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from torchvision.models import resnet18, ResNet18_Weights

# El F sirve para aplicar simplemente operaciones como ReLU o el MaxPooling
# Por el otro lado, en los layers si tenemos pesos, por lo que se usa el nn.Module

file_path = pathlib.Path(__file__).parent.absolute()

def build_backbone(model='resnet18', weights='imagenet', freeze=True, last_n_layers=2):
    if model == 'resnet18':
        backbone = resnet18(pretrained=weights == 'imagenet')
        if freeze:
            for param in backbone.parameters():
                param.requires_grad = False
        return backbone
    else:
        raise Exception(f'Model {model} not supported')

class Network(nn.Module):
    def __init__(self, input_dim: int, n_classes: int) -> None:  # Manera B
        super().__init__() 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.conv1 = nn.Conv2d(1, out_channels=32, kernel_size=3) #dimension 46x46
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, out_channels=64, kernel_size=3) #dimension 44x44
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) # dimension 22x22

        self.conv3 = nn.Conv2d(64, out_channels=128, kernel_size=3) #dimension 20x20
        self.relu3 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) # dimension 10x10

        self.conv4 = nn.Conv2d(128, out_channels=256, kernel_size=3) #dimension 8x8
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1) # dimension 4x4

        out_channels = 256
        h_out = 4
        w_out = 4

        self.fc1 = nn.Linear(out_channels * h_out * w_out, 1024)
        self.relu5 = nn.ReLU()

        self.fc2 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU()

        self.fc3 = nn.Linear(512, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
    

        self.to(self.device)

 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # TODO: Define la propagacion hacia adelante de tu red ✅

        x = self.conv1(x)
        print("CONVOLUCION 1 HECHA: ", x.size())
        x = F.relu(x)
        #Segunda capa conv
        x = self.conv2(x)
        print("CONVOLUCION 2 HECHA: ", x.size())
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        print("MAX POOLING 1 HECHA: ", x.size())
        #Tercera capa conv
        x = self.conv3(x)
        print("CONVOLUCION 3 HECHA: ", x.size())
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        #Cuarta capa conv
        x = self.conv4(x)
        print("CONVOLUCION 4 HECHA: ", x.size())
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        print("MAX POOLING 2 HECHA: ", x.size())


        #Flatten y primera fully connected
        x = torch.flatten(x, 1) # B, C, H, W
        print("FLATTEN HECHO: ", x.size())
        x = self.fc1(x)
        x = F.relu(x) #x
        print("FULLY CONNECTED 1 HECHA: ", x.size())
        #Segunda fully connected
        x = self.fc2(x)
        x = F.relu(x)
        print("FULLY CONNECTED 2 HECHA: ", x.size())
        #Tercera fully connected y ultima
        x = self.fc3(x)
        logits = x
        print("FULLY CONNECTED 3 HECHA: ", x.size())
        print("logits: ", logits.size())


        #return x, logits, proba #Logits: Raw outputs from final layer, aqui habia return x
        return logits
    
    def forward_inference(self, x: torch.Tensor) -> torch.Tensor: 

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv4(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(x)

        x = torch.flatten(x) # C, H, W
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        logits = x
        print("logits: ", logits.size())
        print("logits info: ", logits)
        
        return logits

    def predict(self, x):
        with torch.inference_mode():
            return self.forward_inference(x)

    def save_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        torch.save(self.state_dict(), models_path)

    def load_model(self, model_name: str):
        models_path = file_path / 'models' / model_name
        self.load_state_dict(torch.load(models_path))