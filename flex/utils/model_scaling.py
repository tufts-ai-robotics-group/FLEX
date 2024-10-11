# script for scaling the model
from utils.model_uilts import scale_object_new
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, help='input file path')
    parser.add_argument('-o', '--output_name', type=str, help='output name', default=None)
    parser.add_argument('-s', '--scale', nargs='+', type=float, help='scale factor', default=[0.3 ,0.3, 0.3])

    args = parser.parse_args()
    input_path = args.input_path
    output_name = args.output_name
    scale = args.scale
    scale_object_new(input_path, output_name, scale)