"""Example code for submit.
- Author: Junghoon Kim
- Contact: placidus36@gmail.com
"""
import json
import argparse
import torch
import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.model import Model
from src.augmentation.policies import simple_augment_test
from src.utils.common import read_yaml
from src.utils.inference_utils import run_model

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

    transform_test_args = data_confg["AUG_TEST_PARAMS"] if data_config.get("AUG_TEST_PARAMS") else None
    # Transformation for test
    transform_test = getattr(
        __import__("src.augmentation.policies", fromlist=[""]),
        data_config["AUG_TEST"],
    )(dataset=data_config["DATASET"], img_size=data_config["IMG_SIZE"])

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
    model = model.to(device)
    model.eval()
    submission_csv = {}
    for img, _, fname in dataloader:
        img = img.to(device)
        pred, enc_data = run_model(model, img)
        pred = torch.argmax(pred)
        submission_csv[fname[0]] = CLASSES[int(pred.detach())]

    result["macs"] = enc_data
    result["submission"] = submission_csv
    j = json.dumps(result, indent=4)
    save_path = os.path.join(dst_path, 'submission.csv')
    with open(save_path, 'w') as outfile:
        json.dump(result, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit.")
    parser.add_argument(
        "--dst", default=".", type=str, help="destination path for submit"
    )
    parser.add_argument(
        "--weight", required=True, type=str, help="model weight path"
    )
    parser.add_argument(
        "--model_config", required=True, type=str, help="model config path"
    )
    parser.add_argument(
        "--data_config", required=True, type=str, help="dataconfig used for training."
    )
    parser.add_argument(
	"--img_root", required=True, type=str, help="image folder root. e.g) 'data/test'"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    # prepare datalaoder
    dataloader = get_dataloader(img_root=args.img_root, data_config=args.data_config)

    # prepare model
    model_instance = Model(args.model_config, verbose=True)
    model_instance.model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))

    # inference
    inference(model_instance.model, dataloader, args.dst)
