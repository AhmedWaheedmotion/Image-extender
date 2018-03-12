import numpy as np
from PIL import Image
import requests
import scipy.misc
import skimage.transform


class D:
    Left, Down, Right, Up = range(4)


def kernel(d):
    return {
       D.Left: left_kernel,
       D.Down: down_kernel,
       D.Right: right_kernel,
       D.Up: up_kernel
    }[d]


if __name__ == '__main__':
    url = 'https://pbs.twimg.com/media/DJUEgzAVYAAHbkB.png'

    # Desired size
    output_h = 1080
    output_w = 1920

    # load the image
    image = np.array(Image.open(requests.get(url, stream=True).raw)) / 255

    # input image dimensions
    input_h, input_w, nb_vals = image.shape

    # scale the image
    """
    ratio_h = output_h/input_h
    ratio_w = output_w/input_h
    if ratio_h > ratio_w:
        image = skimage.transform.resize(image, (round(ratio_w * input_h), output_w, nb_vals))
    else:
        image = skimage.transform.resize(image, (output_h, round(ratio_h * input_w), nb_vals))
    """

    # dimensions of the scaled image
    input_h, input_w, nb_vals = image.shape

    result = np.zeros((output_h, output_w, nb_vals))

    # Copy the input image in the middle of the result image
    startx = (output_w - input_w) // 2
    starty = (output_h - input_h) // 2
    for y in range(input_h):
        for x in range(input_w):
            result[starty + y, startx + x] = image[y, x]

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

    scipy.misc.imsave('result.png', result[:, :, :3])
