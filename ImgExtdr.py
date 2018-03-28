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



extensions = {ext for f in imageio.formats for ext in f.extensions}

if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Extend the image to fit a resolution')
    parser.add_argument('img_name', help='path/url of the image to process')
    parser.add_argument('-s', '--scale', help='scale the image before the extension process', action='store_true')
    parser.add_argument('-o', '--output', help='specify the name of the output image', default='result.png')
    parser.add_argument('-k', '--kernel', help='specify the kernel type used for convolution', choices=['square', 'triangle', 'rectangle'], default='triangle')
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
    output_h = 1080
    output_w = 1920

    # load the image
    input = np.array(img) / 255

    # input image dimensions
    input_h, input_w, nb_vals = input.shape

    print('input_h {} input_w {}'.format(input_h, input_w))

    # scale the image
    if args.scale or input_w > output_w or input_h > output_h:
        ratio_h = output_h/input_h
        ratio_w = output_w/input_w
        if ratio_h > ratio_w:
            input = skimage.transform.resize(input, (round(ratio_w * input_h), output_w, nb_vals), mode='reflect')
        else:
            input = skimage.transform.resize(input, (output_h, round(ratio_h * input_w), nb_vals), mode='reflect')

    # dimensions of the scaled image
    input_h, input_w, nb_vals = input.shape

    result = np.zeros((output_h, output_w, nb_vals))

    # Copy the input image in the middle of the result image
    startx = (output_w - input_w) // 2
    starty = (output_h - input_h) // 2
    for y in range(input_h):
        for x in range(input_w):
            result[starty + y, startx + x] = input[y, x]

    total_work = output_w*output_h - input_w*input_h
    work = 0

    def print_progress():
        print('\rProgress: %2.2f%%' % (100*work/total_work), end='')

    kw = args.kernel_width
    kw = kw + 1 - kw%2
    if args.kernel is 'rectangle':
        kh = args.kernel_height
    else:
        kh = kw

    def square():
        kernel = square = np.ones((kw, kw), dtype=bool)
        return (kernel, kernel, kernel, kernel)

    def triangle():
        left_kernel = np.array([[abs(j - kw//2) + kh - i <= kw//2 + 1 for i in range(kh)] for j in range(kw)])
        down_kernel = np.array([[j + abs(i - kw//2) <= kw//2  for i in range(kw)] for j in range(kh)])
        right_kernel = np.array([[abs(j - kw//2) + i <= kw//2 for i in range(kh)] for j in range(kw)])
        up_kernel = np.array([[kh - j + abs(i - kw//2) <= kw//2 + 1  for i in range(kw)] for j in range(kh)])

        return (left_kernel, down_kernel, right_kernel, up_kernel)

    def rectangle():
        return (np.ones((kw, kh), dtype=bool),
                np.ones((kh, kw), dtype=bool),
                np.ones((kw, kh), dtype=bool),
                np.ones((kh, kw), dtype=bool))

    left_k, down_k, right_k, up_k = {
            'triangle': triangle,
            'square': square,
            'rectangle': rectangle
            }[args.kernel]()
    
    # returns the mean of the pixels in the given direction after applying the filtering kernel
    def convolute(y, x, d, minx=0, maxx=output_w - 1, miny=0, maxy=output_h - 1): 
        if d is D.Left:
            filter = left_k[kw//2 - min(kw//2, y - miny): maxy - y + kw//2 + 1]
            values = result[max(y - kw//2, miny): min(y + kw//2, maxy) + 1, x - kw: x]
            return values[filter].mean(axis=0)

        if d is D.Down:
            filter = down_k[:, kw//2 - min(kw//2, x - minx): maxx - x + kw//2 + 1]
            values =  result[y + 1: y + kw + 1, max(x - kw//2, minx): min(x + kw//2, maxx) + 1]
            return values[filter].mean(axis=0)

        if d is D.Right:
            filter = right_k[kw//2 - min(kw//2, y - miny): maxy - y + kw//2 + 1]
            values = result[max(y - kw//2, miny): min(y + kw//2, maxy) + 1, x + 1: x + kw + 1]
            return values[filter].mean(axis=0)

        if d is D.Up:
            filter = up_k[:, kw//2 - min(kw//2, x - minx): maxx - x + kw//2 + 1]
            values = result[y - kw:  y, max(x - kw//2, minx): min(x + kw//2, maxx) + 1]
            return values[filter].mean(axis=0)

        return 0
    
    # extend the image un the upper part
    for y in reversed(range(starty)):
        for x in range(startx, startx + input_w):
            result[y, x] = convolute(y, x, D.Down, minx=startx, maxx=startx + input_w - 1)
        work += input_w
        print_progress()

    # extend the image un the lower part
    for y in range(starty + input_h, output_h):
        for x in range(startx, startx + input_w):
            result[y, x] = convolute(y, x, D.Up, minx=startx, maxx=startx + input_w - 1)
        work += input_w
        print_progress()

    # extend the image un the left part
    for x in reversed(range(startx)):
        for y in range(output_h):
            result[y, x] = convolute(y, x, D.Right)
        work += output_h
        print_progress()

    # extend the image un the right part
    for x in range(startx + input_w, output_w):
        for y in range(output_h):
            result[y, x] = convolute(y, x, D.Left)
        work += output_h
        print_progress()

    # map back the values from [0,1] to [0,255]
    result *= 255

    imageio.imwrite(out_name, result.astype(np.uint8))

    print()
