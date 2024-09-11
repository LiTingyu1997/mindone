import argparse
import torch
import numpy as np
from safetensors import serialize_file


def read_bin_file(bin_file_path):
    # 读取 .bin 文件
    state_dict = torch.load(bin_file_path, map_location='cpu')

    # 检查是否包含元数据
    if 'metadata' in state_dict:
        metadata = state_dict['metadata']
        # 移除元数据，以便将其单独处理
        del state_dict['metadata']
    else:
        metadata = metadata={"format": "pt"}

    return state_dict, metadata


def convert_to_safetensors_format(state_dict):
    # 将 state_dict 转换为 safetensors 格式
    tensor_dict = {}
    for key, tensor in state_dict.items():
        tensor_dict[key] = {
            "dtype": str(tensor.numpy().dtype),
            "shape": list(tensor.shape),
            "data": tensor.numpy().tobytes()
        }
    return tensor_dict


def write_safetensors_file(tensor_dict, metadata, safetensors_file_path):
    # 将数据和元数据写入 .safetensors 文件
    serialize_file(tensor_dict, safetensors_file_path, metadata)


def convert_bin_to_safetensors(bin_file_path, safetensors_file_path):
    # 读取 .bin 文件内容和元数据
    data, metadata = read_bin_file(bin_file_path)

    # 转换为 safetensors 格式
    tensor_dict = convert_to_safetensors_format(data)

    # 写入 .safetensors 文件
    write_safetensors_file(tensor_dict, metadata, safetensors_file_path)
    print(f"Successfully converted {bin_file_path} to {safetensors_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch .bin weights to safetensors format")
    parser.add_argument("--bin_path", type=str, required=True, help="Path to the input .bin file")
    parser.add_argument("--sf_path", type=str, required=True,
                        help="Path to the output .safetensors file")

    args = parser.parse_args()
    convert_bin_to_safetensors(args.bin_path, args.sf_path)

