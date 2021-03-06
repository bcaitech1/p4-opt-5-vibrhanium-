"""Example code for submit.
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import os
import json
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from src.model import Model
from src.utils.common import read_yaml, str2bool
from src.utils.inference_utils import run_model
from src.utils.decompose import decompose
from src.utils.huffman_coding import huffman_decode_model

CLASSES = ['Battery', 'Clothing', 'Glass', 'Metal', 'Paper', 'Paperpack', 'Plastic', 'Plasticbag', 'Styrofoam']

class CustomImageFolder(ImageFolder):
    """ImageFolder with filename."""

    def __getitem__(self, index):
        img_gt = super(CustomImageFolder, self).__getitem__(index)
        fdir = self.imgs[index][0]
        fname = fdir.rsplit(os.path.sep, 1)[-1]
        return (img_gt + (fname,))

def get_dataloader(img_root: str, data_config: str) -> DataLoader:
    """Get dataloader.

    Note:
	Don't forget to set normalization.
    """
    # Load yaml
    data_config = read_yaml(data_config)

    transform_test_args = data_config["AUG_TEST_PARAMS"] if data_config.get("AUG_TEST_PARAMS") else None
    if args.quantized:
        img_size = data_config["IMG_SIZE"]
    else:
        img_size = hyperparam_config["img_size"]

    # Transformation for test
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(dataset=data_config["DATASET"], img_size=img_size)

    dataset = CustomImageFolder(root=img_root, transform=transform_test)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        num_workers=8
    )
    return dataloader

@torch.no_grad()
def inference(model, dataloader, dst_path: str):
    result = {}
    submission_csv = {}

    model = model.to(device)
    model.eval()
    for img, _, fname in tqdm(dataloader):
        img = img.to(device)
        pred, enc_data = run_model(model, img)
        pred = torch.argmax(pred)
        submission_csv[fname[0]] = CLASSES[int(pred.detach())]

    result["macs"] = enc_data
    result["submission"] = submission_csv
    j = json.dumps(result, indent=4)
    
    os.makedirs(dst_path, exist_ok=True)
    save_path = os.path.join(dst_path, 'submission.csv')
    
    with open(save_path, 'w') as outfile:
        json.dump(result, outfile)
    print("Inference complete.")
    print(f"SAVE PATH: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit.")
    parser.add_argument(
        "--dst", default=".", type=str, help="destination path for submit"
    )
    parser.add_argument(
        "--data_config", default="configs/data/taco.yaml", type=str, help="dataconfig used for training."
    )
    parser.add_argument(
	    "--img_root", default="/opt/ml/input/data/test", type=str, help="image folder root. e.g) 'data/test'"
    )
    parser.add_argument(
        "--weight", required=True, type=str, help="model weight path"
    )
    parser.add_argument(
        "--model", required=True, type=str, help="model config path"
    )
    parser.add_argument(
	    "--hyperparam", default=None, type=str, help="hyperparamter config path"
    )    
    parser.add_argument(
        "--decompose", type=str2bool, nargs='?', const=True, default=False, help="whether apply decomposition to convolution layers"
    )
    parser.add_argument(
        "--quantized", type=str2bool, nargs='?', const=True, default=False, help="whether weights is quantized"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # prepare dataloader
    dataloader = get_dataloader(img_root=args.img_root, data_config=args.data_config)

    
    # apply dequantization
    if args.quantized:
        model = getattr(__import__("src.models", fromlist=[""]), args.model)()  # "Squeezenet1_1"
        huffman_decode_model(model, args.weight)

    else:
        hyperparam_config = read_yaml(args.hyperparam)

        # prepare model
        model_instance = Model(args.model, verbose=True)
        model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
        model = model_instance.model

    # apply tensor decomposition to conv layers
    # Applicable conditions: kernel size 3 or higher, groups 1
    if args.decompose:
        model = decompose(model)
        print("===AFTER DECOMPOSITION===")
        print(model)


    # inference
    inference(model, dataloader, args.dst)
