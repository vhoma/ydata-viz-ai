import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan, pow
from scipy.ndimage import affine_transform
from typing import List


def Affine2dTranslateMatrix(dx, dy):
    """
    Returns a 2D Affine translation matrix
    :param dx: The translation along axis 0
    :param dy: The translation along axis 1
    """
    Ttranslate = np.array([[1, 0, dx],
                           [0, 1, dy],
                           [0, 0, 1,]], dtype='float32')
    return Ttranslate


def Affine3dTranslateMatrix(dx, dy, dz):
    """
    Returns a 3D Affine translation matrix
    :param dx: The translation along axis 0
    :param dy: The translation along axis 1
    :param dz: The translation along axis 2
    """
    Ttranslate = np.array([[1, 0, 0, dx],
                           [0, 1, 0, dy],
                           [0, 0, 1, dz],
                           [0, 0, 0, 1]], dtype='float32')
    return Ttranslate


def Affine2dRotateMatrix(alpha: float=0):
    """
    Returns a 2D Affine rotation matrix given an angle in Degrees
    :param alpha: The angle to be rotated in degrees
    """
    alpha = np.deg2rad(alpha)
    Trotate = np.array([[cos(alpha), - sin(alpha), 0],
                        [sin(alpha), cos(alpha), 0],
                        [0, 0, 1]], dtype='float32')
    return Trotate


def Affine2dRotateAroundCenterMatrix(alpha: float, shape: List[int]):
    """
    Returns a D Affine rotation matrix around center of image
    :param alpha: The angle to be rotated in degrees
    :param shape: The shape of the array to be rotated around center
    """
    assert len(shape) == 2, "Input array must be 2D"
    Trotate = Affine2dRotateMatrix(alpha)

    alpha = np.deg2rad(alpha)
    betta = atan(shape[1] / shape[0])
    r = pow(pow(shape[0] / 2, 2) + pow(shape[1] / 2, 2), 0.5)
    Dx = r * (cos(betta) - cos(alpha + betta))
    Dy = r * (sin(betta) - sin(alpha + betta))

    Ttranslate = np.array([[1, 0, Dx],
                           [0, 1, Dy],
                           [0, 0, 1]], dtype='float32')
    M = np.dot(Ttranslate, Trotate)
    return affine_transform(I, M)


def Affine3dRotateMatrix(alpha: float=0, axis: int = 0):
    """
    Returns a 3D Affine rotation matrix along axis given an angle in Degrees
    :param alpha: The angle to be rotated in degrees
    :param axis: The axis to rotate around (0 / 1 / 2)
    """
    assert axis in [0, 1, 2], "Axis may be either 0, 1 or 2"
    alpha = np.deg2rad(alpha)
    if axis == 0:
        Trotate = np.array([[1, 0, 0, 0],
                            [0, cos(alpha), sin(alpha), 0],
                            [0, - sin(alpha), cos(alpha), 0],
                            [0, 0, 0, 1]], dtype='float32')
    if axis == 1:
        Trotate = np.array([[cos(alpha), 0, - sin(alpha), 0],
                            [0, 1, 0, 0],
                            [sin(alpha), 0, cos(alpha), 0],
                            [0, 0, 0, 1]], dtype='float32')
    if axis == 2:
        Trotate = np.array([[cos(alpha), - sin(alpha), 0, 0],
                            [sin(alpha), cos(alpha), 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], dtype='float32')
    return Trotate


def Affine2dRotateAroundCenterMatrix(alpha: float, shape: List[int]):
    """
    Returns a 2D Affine rotation matrix around around center of image
    :param alpha: The angle to be rotated in degrees
    :param shape: The shape of the array to be rotated around center
    """
    assert len(shape) == 2, "Input shape must be 2D"
    # get new center location after rotation
    center = [c / 2 for c in shape] + [1]
    Trotate = Affine2dRotateMatrix(alpha)
    new_center = np.dot(Trotate, center)
    diff = center - new_center
    # translate center back to original image center
    Ttranslate = Affine2dTranslateMatrix(*diff[:2])
    return np.dot(Ttranslate, Trotate)


def Affine3dRotateCenterMatrix(alpha: float, shape: List[int], axis: int = 0):
    """
    Returns a 3D Affine rotation matrix around axis, around center of image
    :param alpha: The angle to be rotated in degrees
    :param shape: The shape of the array to be rotated around center
    :param axis: The axis to rotate around (0 / 1 / 2)
    """
    assert len(shape) == 3, "Input shape must be 3D"
    assert axis in [0, 1, 2], "Axis may be either 0, 1 or 2"
    # get new center location after rotation
    center = [c / 2 for c in shape] + [1]
    Trotate = Affine3dRotateMatrix(alpha, axis)
    new_center = np.dot(Trotate, center)
    diff = center - new_center
    # translate center back to original image center
    Ttranslate = Affine3dTranslateMatrix(*diff[:3])
    return np.dot(Ttranslate, Trotate)


def MIP(I: np.ndarray, w: int = 1, axis = 0):
    """
    Returns the Maximum Intensity Projection of input array
    :param I: The input array
    :param w: The window size in pixels (w=0 will result in returning th original array)
    :param axis: The axis to rotate around (0 / 1 / 2)
    """
    assert I.ndim == 3, "Input array must be 3D"
    assert axis in [0, 1, 2], "Axis may be either 0, 1 or 2"
    assert w >= 0, "window size must be equal or larger than 0"
    max_ind = I.shape[axis]
    Inew = np.swapaxes(I, 0, axis)
    Imip = np.zeros(Inew.shape)
    for ind in range(max_ind):
        Imip[ind, :, :] = Inew[max(0, ind - w): min(ind + w, max_ind), :, :].max(0)
    Imip = np.swapaxes(Imip, axis, 0)
    return Imip


if __name__ == '__main__':
    # load an ndarray of a CT scan, correct accordingly
    pth = 'CT.npz'

    f = np.load(pth, allow_pickle=True)
    I = f['I']
    #

    # Example of how to rotate 2D image around its center
    img = np.moveaxis(I[:, :, int(I.shape[2] / 2)], 0, 1)
    alpha = 45
    M = Affine2dRotateAroundCenterMatrix(alpha, img.shape)
    N = Affine2dRotateMatrix(alpha)
    img_rotate_center = affine_transform(img, M)
    img_rotate = affine_transform(img, N)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(img)
    ax[1].imshow(img_rotate)
    ax[2].imshow(img_rotate_center)
    ax[0].title.set_text('original image')
    ax[1].title.set_text('image rotated by {} [deg]'.format(int(alpha)))
    ax[2].title.set_text('image rotated around center by {} [deg]'.format(int(alpha)))
    plt.show()

    # Example of how to rotate 3D image around its center
    Trot = Affine3dRotateCenterMatrix(90, I.shape, 0)
    Irot = affine_transform(I, Trot)


