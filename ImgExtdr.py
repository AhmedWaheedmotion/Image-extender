import numpy as np
import os
import requests
import imageio
import skimage.transform
from sys import argv
from PIL import Image
import argparse

class D:
    Left, Down, Right, Up = range(4)


def kernel(d):
    return {
       D.Left: left_kernel,
       D.Down: down_kernel,
       D.Right: right_kernel,
       D.Up: up_kernel
    }[d]

extensions = {ext for f in imageio.formats for ext in f.extensions}

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Extend the image to fit a resolution')
    parser.add_argument('img_name', help='path/url of the image to process')
    parser.add_argument('-s', '--scale', help='scale the image before the extension process', action='store_true')
    parser.add_argument('-o', '--output', help='specify the name of the output image', default='result.png')
    args = parser.parse_args()
    name = args.img_name

    # check the desired output name, add .png if it is not valid
    _, ext_name = os.path.splitext(args.output)
    if ext_name in extensions:
        out_name = args.output
    else:
        out_name = args.output + '.png'

    try:
        img = Image.open(name)
    except FileNotFoundError:
        img = Image.open(requests.get(name, stream=True).raw)

    # Desired size
    output_h = 1080
    output_w = 1920

    # load the image
    input = np.array(img)

    # input image dimensions
    input_h, input_w, nb_vals = input.shape

    # scale the image
    if args.scale:
        ratio_h = output_h/input_h
        ratio_w = output_w/input_h
        if ratio_h > ratio_w:
            input = skimage.transform.resize(input, (round(ratio_w * input_h), output_w, nb_vals))
        else:
            input = skimage.transform.resize(input, (output_h, round(ratio_h * input_w), nb_vals))

    # dimensions of the scaled image
    input_h, input_w, nb_vals = input.shape

    result = np.zeros((output_h, output_w, nb_vals))

    # Copy the input image in the middle of the result image
    startx = (output_w - input_w) // 2
    starty = (output_h - input_h) // 2
    for y in range(input_h):
        for x in range(input_w):
            result[starty + y, startx + x] = input[y, x]

    n = 20
    left_kernel = np.array([(j, -i - 1) for i in range(n) for j in range(i - n + 1, n - i)])
    down_kernel = np.array([(i + 1, j) for i in range(n) for j in range(i - n + 1, n - i)])
    right_kernel = np.array([(j, i + 1) for i in range(n) for j in range(i - n + 1, n - i)])
    up_kernel = np.array([(-i - 1, j) for i in range(n) for j in range(i - n + 1, n - i)])


    def convolute(y, x, d, minx=0, maxx=output_w - 1, miny=0, maxy=output_h - 1):
        return np.mean(result[np.clip([y, x] + kernel(d), [miny, minx], [maxy, maxx]).T.tolist()], axis=0)


    for y in reversed(range(starty)):
        for x in range(startx, startx + input_w):
            result[y, x] = convolute(y, x, D.Down, minx=startx, maxx=startx + input_w - 1)

    for y in range(starty + input_h, output_h):
        for x in range(startx, startx + input_w):
            result[y, x] = convolute(y, x, D.Up, minx=startx, maxx=startx + input_w - 1)

    for x in reversed(range(startx)):
        for y in range(output_h):
            result[y, x] = convolute(y, x, D.Right)

    for x in range(startx + input_w, output_w):
        for y in range(output_h):
            result[y, x] = convolute(y, x, D.Left)

    imageio.imwrite(out_name, result[:, :, :3])
