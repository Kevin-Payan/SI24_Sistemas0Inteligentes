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

        # TODO: Calcular dimension de salida ✅
        # TODO: Define las capas de tu red ✅

        #Primera convolucion
        self.conv1 = nn.Conv2d(1, out_channels=24, kernel_size=3) #dimension 46x46
        self.relu1 = nn.ReLU()
        print("Primera convolucion declarada.")

        #Segunda convolucion
        self.conv2 = nn.Conv2d(24, out_channels=18, kernel_size=3) #dimension 44x44
        self.relu2 = nn.ReLU()
        print("Segunda convolucion declarada.")

        #Tercera convolucion
        self.conv3 = nn.Conv2d(18, out_channels=24, kernel_size=5) #dimension 40x40
        self.relu3 = nn.ReLU()
        print("Tercera convolucion declarada.")

        #Cuarta convolucion
        self.conv4 = nn.Conv2d(24, out_channels=36, kernel_size=7) #dimension 34x34
        self.relu4 = nn.ReLU()

        #Primera fully connected
        h_out = 34
        w_out = 34

        self.fc1 = nn.Linear(36 * h_out * w_out, 72)
        self.relu5 = nn.ReLU()
        print("Primera fully connected declarada.")

        #Segunda fully connected y ultima
        self.fc2 = nn.Linear(72, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        print("Segunda y ultima fully connected declarada.")

        self.to(self.device)

 
    def calc_out_dim(self, in_dim, kernel_size, stride=1, padding=0):
        out_dim = math.floor((in_dim - kernel_size + 2*padding)/stride) + 1
        return out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # TODO: Define la propagacion hacia adelante de tu red ✅
        #Primera capa conv
        x = self.conv1(x)
        x = F.relu(x)
        print("CONVOLUCION 1 HECHA: ", x.size())
        #Segunda capa conv
        x = self.conv2(x)
        x = F.relu(x)
        print("CONVOLUCION 2 HECHA: ", x.size())
        #Tercera capa conv
        x = self.conv3(x)
        x = F.relu(x)
        print("CONVOLUCION 3 HECHA: ", x.size())
        #Cuarta capa conv
        x = self.conv4(x)
        x = F.relu(x)
        print("CONVOLUCION 4 HECHA: ", x.size())
        #Flatten y primera fully connected

        x = torch.flatten(x)

        print("FLATTEN HECHO: ", x.size())
        #print("Esto: ", nn.Flatten(x, 1))
        #print("Deberia ser lo mismo que: ", x.reshape(-1, 36 * 34 * 34))
        x = self.fc1(x)
        x = F.relu(x)
        print("FULLY CONNECTED 1 HECHA: ", x.size())
        #Segunda y ultima fully connected
        x = self.fc2(x)
        logits = x
        print("FULLY CONNECTED 2 HECHA: ", x.size())
        print("logits: ", logits.size())


        #return x, logits, proba #Logits: Raw outputs from final layer, aqui habia return x
        return logits

    def predict(self, x):
        with torch.inference_mode():
            return self.forward(x)

    def save_model(self, model_name: str):
        '''
            Guarda el modelo en el path especificado
            args:
            - net: definición de la red neuronal (con nn.Sequential o la clase anteriormente definida)
            - path (str): path relativo donde se guardará el modelo
        '''
        models_path = file_path / 'models' / model_name
        # TODO: Guarda los pesos de tu red neuronal en el path especificado
        torch.save(model_name, models_path)

    def load_model(self, model_name: str):
        '''
            Carga el modelo en el path especificado
            args:
            - path (str): path relativo donde se guardó el modelo
        '''
        # TODO: Carga los pesos de tu red neuronal
        models_path = file_path / 'models' / model_name
        torch.load(models_path)
