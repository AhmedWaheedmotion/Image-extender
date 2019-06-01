import numpy as np
import os
import requests
import imageio
import skimage.transform
from sys import argv
from PIL import Image
import argparse

from extender import Extender, Kernel, D, triangle, rectangle, square, disc

extensions = {ext for f in imageio.formats for ext in f.extensions}

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Extend the image to fit a resolution')
    parser.add_argument('img_name', help='path/url of the image to process')
    parser.add_argument('-s', '--scale', help='scale the image before the extension process', action='store_true')
    parser.add_argument('-o', '--output', help='specify the name of the output image', default='result.png')
    parser.add_argument('-ow', '--output_width', help='specify the width for the output image', type=int, default=1920)
    parser.add_argument('-oh', '--output_height', help='specify the height for the output image', type=int, default=1080)
    parser.add_argument('-k', '--kernel', help='specify the kernel type used for convolution', choices=['square', 'triangle', 'rectangle', 'disc'], default='triangle')
    parser.add_argument('-kw', '--kernel_width', help='specify the kernel width (default = 21)', type=int, default=21)
    parser.add_argument('-kh', '--kernel_height', help='specify the kernel height (default = 21)', type=int, default=21)
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
    output_h = args.output_height
    output_w = args.output_width

    # load the image
    input = np.array(img)

    # input image dimensions
    input_h, input_w, nb_vals = input.shape

    # scale the image
    if args.scale or input_w > output_w or input_h > output_h:
        ratio_h = output_h/input_h
        ratio_w = output_w/input_w
        if ratio_h > ratio_w:
            input = skimage.transform.resize(input, (round(ratio_w * input_h), output_w, nb_vals), mode='reflect')
        else:
            input = skimage.transform.resize(input, (output_h, round(ratio_h * input_w), nb_vals), mode='reflect')
        input = np.uint8(input * 255)
    

    # dimensions of the scaled image
    input_h, input_w, nb_vals = input.shape

    result = np.zeros((output_h, output_w, nb_vals))

    # Copy the input image in the middle of the result image
    startx = (output_w - input_w) // 2
    starty = (output_h - input_h) // 2
    for y in range(input_h):
        for x in range(input_w):
            result[starty + y, startx + x] = input[y, x]

    kw = args.kernel_width
    kw = kw + 1 - kw % 2
    
    if args.kernel == 'triangle':
        kh = kw
        kernel = triangle(kw, kh)
    elif args.kernel == 'square':
        kh = kw // 2 + 1
        kernel = square(kw, kh)
    elif args.kernel == 'rectangle':
        kh = args.kernel_height
        kernel = rectangle(kw, kh)
    elif args.kernel == 'disc':
        kh = kw // 2 + 1
        kernel = disc(kw, kh)
    else:
        raise ValueError('Unkown kernel')

    extender = Extender(kernel)

    # extend the image un the upper part
    extender.convolute(result, D.Up, [startx, startx + input_w - 1], [starty - 1, 0])
    
    # extend the image un the lower part
    extender.convolute(result, D.Down, [startx, startx + input_w - 1], [starty + input_h, output_h - 1])

    # extend the image un the left part
    extender.convolute(result, D.Left, [startx - 1, 0], [0, output_h - 1])

    # extend the image un the right part
    extender.convolute(result, D.Right, [startx + input_w, output_w - 1], [0, output_h - 1])

    print(result)
    
    imageio.imwrite(out_name, result.astype(np.uint8))

    print()
