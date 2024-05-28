import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text, to_torch
from dataset import EMOTIONS_MAP
from glob import glob
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size=48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized


def predict(img_title_paths):
    """
    Hace la inferencia de las imagenes
    args:
    - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    """
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("arq7_exp1")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        logits = modelo.predict(transformed.cuda())
        pred = torch.argmax(logits, -1).item()
        pred_label = EMOTIONS_MAP[pred]

        #EMOTIONS_MAP = {  0: "Enojo",  1: "Disgusto",  2: "Miedo", 3: "Alegria", 4: "Tristeza", 5: "Sorpresa", 6: "Neutral"}

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)


if __name__ == "__main__":
    # Direcciones relativas a este archivo
    img_paths = [
        "./test_imgs/happy.png",
        "./test_imgs/happy_2.png",
        "./test_imgs/happy_3.png",
        "./test_imgs/angry.png",
        "./test_imgs/impressed.png",
        "./test_imgs/neutral.png",
        "./test_imgs/neutral_2.png",
        "./test_imgs/sad.png",
        "./test_imgs/sad_2.png",
        "./test_imgs/sad_3.png",
        "./test_imgs/happy_4.png",
        "./test_imgs/happy_adolfo.png",
        "./test_imgs/happy_caro.png",
        "./test_imgs/happy_kevin.png",
        "./test_imgs/happy_ord.png",
        "./test_imgs/angry_adolfo.png",
        "./test_imgs/angry_caro.png",
        "./test_imgs/angry_kevin.png",
        "./test_imgs/angry_ord.png",
        "./test_imgs/scary_adolfo.png",
        "./test_imgs/scary_caro.png",
        "./test_imgs/scary_kevin.png",
        "./test_imgs/scary_ord.png",
        "./test_imgs/sad_adolfo.png",
        "./test_imgs/sad_caro.png",
        "./test_imgs/sad_kevin.png",
        "./test_imgs/sad_ord.png",
        "./test_imgs/dislike_adolfo.png",
        "./test_imgs/dislike_caro.png",
        "./test_imgs/dislike_kevin.png",
        "./test_imgs/dislike_ord.png",
        "./test_imgs/impressed_adolfo.png",
        "./test_imgs/impressed_caro.png",
        "./test_imgs/impressed_kevin.png",
        "./test_imgs/impressed_ord.png",
        "./test_imgs/neutral_adolfo.png",
        "./test_imgs/neutral_caro.png",
        "./test_imgs/neutral_kevin.png",
        "./test_imgs/neutral_ord.png",
        "./test_imgs/elbicho_siu.jpeg",
        "./test_imgs/elbicho_sin_cejas.jpg",
        "./test_imgs/anda_pasha_bobo.jpeg",
        "./test_imgs/messi_feliz.jpeg"

    ]

    predict(img_paths)
