"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import torch
from net.ShuffleV2 import ShuffleNetV2
import onnx
from onnxsim import simplify
import cv2
import numpy as np

from PIL import Image
from torchvision import datasets, models, transforms
data_transforms = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default="../model/shuffle2_sm.pt", help='weights path')  # from yolov5/models/

    parser.add_argument('--img-size', nargs='+', type=int, default=[128, 128], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # Input
    img = torch.zeros((opt.batch_size, 3, *opt.img_size))  # image size(1,3,320,192) iDetection

    # Load PyTorch model
    model = ShuffleNetV2().to(device)
    model.eval()
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.weights).items()})
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(opt.weights, map_location=torch.device('cpu')).items()})

    y = model(img)  # dry run

    onnx_sim = False
    tf = True
    tflite = True


    if onnx_sim:
        # # ONNX export
        try:
            import onnx

            print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
            f = opt.weights.replace('.pth', '.onnx').replace('.pt', '.onnx')   # filename
            torch.onnx.export(model, img, f, verbose=False, opset_version=10, input_names=['images'],
                              output_names=['classes', 'boxes'] if y is None else ['output'])

            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
            print('ONNX export success, saved as %s' % f)

        except Exception as e:
            print('ONNX export failure: %s' % e)



           ## ONNXSIM export
        try:
            import onnx
            from onnxsim import simplify

            onnx_path = opt.weights.replace('.pth', '.onnx').replace('.pt', '.onnx')   # filename
            onnxsim_path = onnx_path.replace('.onnx', '_sim.onnx')  # filename
            print(onnxsim_path)
            onnx_model = onnx.load(onnx_path)  # load onnx model
            model_simp, check = simplify(onnx_model)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(model_simp, onnxsim_path)
            print('finished exporting ONNXSIM')
        except Exception as e:
            print('ONNXSIM export failure: %s' % e)


    if tf:
        from onnx_tf.backend import prepare
        import onnx
        import tensorflow as tf

        onnx_model = onnx.load("../model/shuffle2_sm_sim.onnx")  # load onnx model
        tf_rep = prepare(onnx_model)  # prepare tf representation
        tf_rep.export_graph("shuffle2_sm_sim.tf")  # export the model


    if tflite:
        # Convert the model
        converter = tf.lite.TFLiteConverter.from_saved_model('shuffle2_sm_sim.tf')  # path to the SavedModel directory
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
            tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
        ]
        tflite_model = converter.convert()

        # Save the model.
        with open('shuffle2_sm_sim.tflite', 'wb') as f:
            f.write(tflite_model)

