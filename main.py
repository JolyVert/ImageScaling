import numpy as np
import cv2


def process_image(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)
    return gray_image


def scale_down(image, kernel, padding=0, strides=1):

    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape, yKernShape = kernel.shape
    xImgShape, yImgShape = image.shape

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        image_padded = np.pad(image, pad_width=padding, mode='constant', constant_values=0)
    else:
        image_padded = image

    # Iterate through image
    for x in range(0, xImgShape - xKernShape + 1, strides):
        for y in range(0, yImgShape - yKernShape + 1, strides):
            output[x, y] = np.sum(kernel * image_padded[x:x + xKernShape, y:y + yKernShape])

    return output


def scale_up_linear(image, scale_factor):
    image_height, image_width = image.shape

    # Nowy rozmiar obrazu po skalowaniu
    new_height = int(image_height * scale_factor)
    new_width = int(image_width * scale_factor)

    # Inicjalizuj obraz wynikowy
    result = np.zeros((new_height, new_width), dtype=np.uint8)

    # Interpolacja liniowa
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_factor
            y = j / scale_factor

            x1, y1 = int(x), int(y)
            x2, y2 = min(x1 + 1, image_height - 1), min(y1 + 1, image_width - 1)

            result[i, j] = int(
                image[x1, y1] * (x2 - x) * (y2 - y) +
                image[x1, y2] * (x2 - x) * (y - y1) +
                image[x2, y1] * (x - x1) * (y2 - y) +
                image[x2, y2] * (x - x1) * (y - y1)
            )

    return result


def scale_up_bicubic(image, scale_factor):
    image_height, image_width = image.shape

    # Nowy rozmiar obrazu po skalowaniu
    new_height = int(image_height * scale_factor)
    new_width = int(image_width * scale_factor)

    # Inicjalizuj obraz wynikowy
    result = np.zeros((new_height, new_width), dtype=np.uint8)

    def bicubic_interpolation(x, y):
        x = max(0, min(image.shape[0] - 1, x))
        y = max(0, min(image.shape[1] - 1, y))

        x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
        x_frac, y_frac = x - x_floor, y - y_floor

        coefficients = np.array([
            [1, 0, 0, 0],
            [-0.5, 0.5, 1, 0],
            [1, -2.5, 2, -0.5],
            [-0.5, 1.5, -1.5, 0.5]
        ])

        x_coords = np.array([1, x_frac, x_frac**2, x_frac**3])
        y_coords = np.array([1, y_frac, y_frac**2, y_frac**3])

        pixels = np.zeros((4,), dtype=np.float32)

        for i in range(4):
            for j in range(4):
                x_idx = x_floor + i - 1
                y_idx = y_floor + j - 1
                x_idx = max(0, min(image.shape[0] - 1, x_idx))
                y_idx = max(0, min(image.shape[1] - 1, y_idx))
                pixels[i] += image[x_idx, y_idx] * coefficients[i, j]

        interpolated_value = int(np.dot(x_coords, np.dot(pixels, coefficients.T)))

        return np.clip(interpolated_value, 0, 255)

    # Wykonaj bicubic interpolation
    for i in range(new_height):
        for j in range(new_width):
            x = i / scale_factor
            y = j / scale_factor
            result[i, j] = bicubic_interpolation(x, y)

    return result

def mse(image1, image2):
    # Upewnij się, że obrazy mają takie same wymiary
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    image1 = image1[:min_height, :min_width]
    image2 = image2[:min_height, :min_width]

    return np.mean((image1 - image2)**2)


if __name__ == '__main__':
    image = process_image('img.png')

    scale_factor_down = 0.5
    kernel_down = np.ones((int(1 / scale_factor_down), int(1 / scale_factor_down)), np.float32) / (1 / scale_factor_down ** 2)
    output_down = scale_down(image, kernel_down, padding=1)
    cv2.imwrite('ScaledDown.png', output_down)

    scale_factor_up = 2.0
    output_up_linear = scale_up_linear(image, scale_factor_up)
    cv2.imwrite('ScaledUpLinear.png', output_up_linear)

    output_up_bicubic = scale_up_bicubic(image, scale_factor_up)
    cv2.imwrite('ScaledUpBicubic.png', output_up_bicubic)

    mse_downsampled = mse(image, output_down)
    mse_enlarged_linear = mse(image, output_up_linear)
    mse_enlarged_bicubic = mse(image, output_up_bicubic)

    print(f"MSE dla obrazu przeskalowanego w dół: {mse_downsampled}")
    print(f"MSE dla obrazu przeskalowanego w górę metodą linear: {mse_enlarged_linear}")
    print(f"MSE dla obrazu przeskalowanego w górę metodą bicubic: {mse_enlarged_bicubic}")