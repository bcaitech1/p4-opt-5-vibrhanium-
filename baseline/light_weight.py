import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torchvision.models

from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix

from src.utils.huffman_coding import huffman_encode


# https://github.com/mightydeveloper/Deep-Compression-PyTorch
def apply_weight_sharing(module, layer_type: str, bits: int = 5):
    assert layer_type in ['conv2d', 'linear']
    
    dev = module.weight.device
    weight = module.weight.data.cpu().numpy()

    # Conv layer's weight(4 dimension) to 2 dimension
    if layer_type == 'conv2d':
        out_channels, in_channels, kernel_size_H, kernel_size_W = weight.shape
        weight = weight.reshape(out_channels, -1)
    
    shape = weight.shape
    mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

    # Linear initialization
    min_ = min(mat.data)
    max_ = max(mat.data)
    space = np.linspace(min_, max_, num=2**bits)

    # Kmean clustering
    kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, algorithm="full")
    kmeans.fit(mat.data.reshape(-1,1))
    new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
    mat.data = new_weight
    new_weight = torch.from_numpy(mat.toarray()).to(dev)

    # new_weight(2 dimension) to Conv layer's weight(4 dimension)
    if layer_type == 'conv2d':
        new_weight = new_weight.reshape(out_channels, in_channels, kernel_size_H, kernel_size_W)
    
    module.weight.data = new_weight


def apply_huffman_encode(name, param, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    
    if 'weight' in name:
        weight = param.data.cpu().numpy()
        shape = weight.shape
        
        if len(shape) == 4:
            out_channels = weight.shape[0]
            weight = weight.reshape(out_channels, -1)
            shape = weight.shape
        elif len(shape) == 1:
            weight.dump(f'{directory}/{name}')
            return None

        form = 'csr' if shape[0] < shape[1] else 'csc'
        mat = csr_matrix(weight, shape) if shape[0] < shape[1] else csc_matrix(weight, shape)

        # Encode
        t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory)
        t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory)
        t2, d2 = huffman_encode(mat.indptr, name+f'_{form}_indptr', directory)
        
        # Print statistics
        original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
        compressed = t0 + t1 + t2 + d0 + d1 + d2

        print(f"{name:<45} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
    
    elif 'bias' in name:
        # Note that we do not huffman encode bias
        bias = param.data.cpu().numpy()
        bias.dump(f'{directory}/{name}')


def model_optimizer_children(modules, func, *args): 
    try:
        for module in enumerate(modules.children()):
            if isinstance(module, nn.Conv2d):
                func(module, 'conv2d', *args)
            elif isinstance(module, nn.Linear):
                func(module, 'linear', *args)
            else:
                model_optimizer_children(module, func)

    except Exception as e:
        print(f'*error in {modules}*', e)


def model_optimizer_params(model, func, *args):
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        func(name, param, *args)


def test(modules, f):
    for module in modules.children():
        if isinstance(module, nn.Conv2d):
            f.write(str(module))
            f.write(str(module.weight))
        elif isinstance(module, nn.Linear):
            f.write(str(module))
            f.write(str(module.weight))
        else:
            test(module, f)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument(
        "--model", required=True, type=str, help="model config path"
    )
    parser.add_argument(
        "--weight", required=True, type=str, help="model weight path"
    )
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model = getattr(__import__("src.models", fromlist=[""]), args.model)()  # "Squeezenet1_1"
    model = model.to(device)
    try:
        model.load_state_dict(torch.load(args.weight, map_location=torch.device('cpu')))
    except:
        model.load_weight(args.weight)
    quantized_weight_dir = os.path.join(os.path.dirname(args.weight), "quantized_best")
    os.makedirs(quantized_weight_dir, exist_ok=True)

    # quantization 적용 전
    with open(os.path.join(quantized_weight_dir, "before_weight.txt"), "w") as f:
        test(model, f)

    model_optimizer_children(model, apply_weight_sharing, 2)  # weight sharing using 4 groups
    model_optimizer_params(model, apply_huffman_encode, quantized_weight_dir)  # huffman code using codebook

    with open(os.path.join(quantized_weight_dir, "after_weight.txt"), "w") as f:
        test(model, f)
