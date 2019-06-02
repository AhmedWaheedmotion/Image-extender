import numpy as np

class D:
    Left, Down, Right, Up = range(4)

class Kernel:
    def __init__(self, left_kernel, down_kernel, right_kernel, up_kernel):
        self.kernel = {
            D.Left: left_kernel,
            D.Down: down_kernel,
            D.Right: right_kernel,
            D.Up: up_kernel
        }

        self.kw = right_kernel.shape[0]
        self.kh = right_kernel.shape[1]

    @classmethod
    def fromleftkernel(cls, left_kernel):
        down_kernel = np.rot90(left_kernel, axes=(0,1))
        right_kernel = np.rot90(down_kernel, axes=(0,1))
        up_kernel = np.rot90(right_kernel, axes=(0,1))
        return cls(left_kernel, down_kernel, right_kernel, up_kernel)

    @classmethod
    def fromdownkernel(cls, down_kernel):
        right_kernel = np.rot90(down_kernel, axes=(0,1))
        up_kernel = np.rot90(right_kernel, axes=(0,1))
        left_kernel = np.rot90(up_kernel, axes=(0,1))
        return cls(left_kernel, down_kernel, right_kernel, up_kernel)

    @classmethod
    def fromrightkernel(cls, right_kernel):
        up_kernel = np.rot90(right_kernel, axes=(0,1))
        left_kernel = np.rot90(up_kernel, axes=(0,1))
        down_kernel = np.rot90(left_kernel, axes=(0,1))
        return cls(left_kernel, down_kernel, right_kernel, up_kernel)

    @classmethod
    def fromupkernel(cls, up_kernel):
        left_kernel = np.rot90(up_kernel, axes=(0,1))
        down_kernel = np.rot90(left_kernel, axes=(0,1))
        right_kernel = np.rot90(down_kernel, axes=(0,1))
        return cls(left_kernel, down_kernel, right_kernel, up_kernel)

    def __getitem__(self, key):
        return self.kernel[key]

def square(kw, kh):
    kernel = square = np.ones((kw, kw), dtype=bool)
    return Kernel(kernel, kernel, kernel, kernel)

def triangle(kw, kh):
    left_kernel = np.array([[abs(j - kw//2) + kh - i <= kw//2 + 1 for i in range(kh)] for j in range(kw)])
    return Kernel.fromleftkernel(left_kernel)

def rectangle(kw, kh):
    return Kernel(np.ones((kw, kh), dtype=bool),
                  np.ones((kh, kw), dtype=bool),
                  np.ones((kw, kh), dtype=bool),
                  np.ones((kh, kw), dtype=bool))

def disc(kw, kh):
    left_kernel = np.array([[abs(j - kw//2)**2 + (kh - i - 1)**2 <= (kw//2)**2 for i in range(kh)] for j in range(kw)])
    return Kernel.fromleftkernel(left_kernel)


class Extender:
    def __init__(self, kernel):
        self.kernel = kernel
    
    def convolute(self, arr, d, x_range, y_range):
        kw = self.kernel.kw
        kh = self.kernel.kh
        from_x, to_x = x_range
        from_y, to_y = y_range
        
        if d is D.Up:
            for y in range(from_y, to_y - 1, -1):
                for x in range(from_x, to_x + 1):
                    filter = self.kernel[D.Down][:, kw//2 - min(kw//2, x - from_x): to_x - x + kw//2 + 1]
                    values = arr[y + 1: y + kh + 1, max(x - kw//2, from_x): min(x + kw//2, to_x) + 1]
                    arr[y, x] = values[filter].mean(axis=0)
        
        elif d is D.Right:
            for x in range(from_x, to_x + 1):
                for y in range(from_y, to_y + 1):
                    filter = self.kernel[D.Left][kw//2 - min(kw//2, y - from_y): to_y - y + kw//2 + 1]
                    values = arr[max(y - kw//2, from_y): min(y + kw//2, to_y) + 1, x - kh: x]
                    arr[y, x] = values[filter].mean(axis=0)

        elif d is D.Down:
            for y in range(from_y, to_y + 1):
                for x in range(from_x, to_x + 1):
                    filter = self.kernel[D.Up][:, kw//2 - min(kw//2, x - from_x): to_x - x + kw//2 + 1]
                    values = arr[y - kh:  y, max(x - kw//2, from_x): min(x + kw//2, to_x) + 1]
                    arr[y, x] = values[filter].mean(axis=0)

        elif d is D.Left:
            for x in range(from_x, to_x - 1, -1):
                for y in range(from_y, to_y + 1):
                    filter = self.kernel[D.Right][kw//2 - min(kw//2, y - from_y): to_y - y + kw//2 + 1]
                    values = arr[max(y - kw//2, from_y): min(y + kw//2, to_y) + 1, x + 1: x + kh + 1]
                    arr[y, x] =  values[filter].mean(axis=0)