# coding=utf8

import onnx

def test(onnx_path):
    onnx.load(onnx_path)


if __name__ == "__main__":
    import sys
    t_path = sys.argv[1]
    test(t_path)
