"""Problem Set 4: Motion Detection"""

import numpy as np
import cv2
import os


# Utility function
def normalize_and_scale(image_in, scale_range=(0, 255)):
    """Normalizes and scales an image to a given range [0, 255].

    Utility function. There is no need to modify it.

    Args:
        image_in (numpy.array): input image.
        scale_range (tuple): range values (min, max). Default set to
                             [0, 255].

    Returns:
        numpy.array: output image.
    """
    image_out = np.zeros(image_in.shape)
    cv2.normalize(image_in, image_out, alpha=scale_range[0],
                  beta=scale_range[1], norm_type=cv2.NORM_MINMAX)

    return image_out


# Assignment code
def gradient_x(image):
    """Computes image gradient in X direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the X direction. Output
                     from cv2.Sobel.
    """
    img = image.copy()
    grad_x = cv2.Sobel(img, -1, 1, 0, ksize = 3, scale = 1/8.0)
    return grad_x

def gradient_y(image):
    """Computes image gradient in Y direction.

    Use cv2.Sobel to help you with this function. Additionally you
    should set cv2.Sobel's 'scale' parameter to one eighth and ksize
    to 3.

    Args:
        image (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].

    Returns:
        numpy.array: image gradient in the Y direction.
                     Output from cv2.Sobel.
    """
    img = image.copy()
    grad_y = cv2.Sobel(img, -1, 0, 1, ksize = 3, scale = 1/8.0)
    return grad_y

def optic_flow_lk(img_a, img_b, k_size, k_type, sigma=1):
    """Computes optic flow using the Lucas-Kanade method.

    For efficiency, you should apply a convolution-based method.

    Note: Implement this method using the instructions in the lectures
    and the documentation.

    You are not allowed to use any OpenCV functions that are related
    to Optic Flow.

    Args:
        img_a (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image with
                             values in [0.0, 1.0].
        k_size (int): size of averaging kernel to use for weighted
                      averages. Here we assume the kernel window is a
                      square so you will use the same value for both
                      width and height.
        k_type (str): type of kernel to use for weighted averaging,
                      'uniform' or 'gaussian'. By uniform we mean a
                      kernel with the only ones divided by k_size**2.
                      To implement a Gaussian kernel use
                      cv2.getGaussianKernel. The autograder will use
                      'uniform'.
        sigma (float): sigma value if gaussian is chosen. Default
                       value set to 1 because the autograder does not
                       use this parameter.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along
                             X-axis, same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along
                             Y-axis, same size and type as U.
    """

      #if blur:
    imgA = img_a.copy()
    imgB = img_b.copy()
    #w1,w2 = 15,15
    #if k_type == 'gaussian':
    #  imgA = cv2.GaussianBlur(imgA, (k_size, k_size), sigma)
    #  imgB = cv2.GaussianBlur(imgB, (k_size, k_size), sigma)
    sobelX = gradient_x(imgA)
    sobelY = gradient_y(imgB)
    It = imgB - imgA
    kernel = None
    #when k_type = 'gaussian'
    if k_type == "gaussian":
      kernel = cv2.getGaussianKernel(k_size, sigma)
      kernel = np.dot(kernel, kernel.T)
    #when k_type = 'uniform'
    else:
      kernel = np.ones((k_size,k_size),dtype = np.float_) /(1.0*k_size*k_size)
    m00 = cv2.filter2D(sobelX*sobelX, -1, kernel)
    m01 = cv2.filter2D(sobelX*sobelY, -1, kernel)
    m10 = cv2.filter2D(sobelY*sobelX, -1, kernel)
    m11 = cv2.filter2D(sobelY*sobelY, -1, kernel)
    n0 = cv2.filter2D(sobelX*It, -1, -kernel)
    n1 = cv2.filter2D(sobelY*It, -1, -kernel)

    detM = (m00 * m11) - (m01 * m10)
    detM[np.where(detM<0.000001)] = 100000000000

    U = ((m11 * n0) - (m01 * n1)) / detM
    V = ((-m10 * n0) + (m00 * n1)) / detM

    return U, V

def reduce_image(image):
    """Reduces an image to half its shape.

    The autograder will pass images with even width and height. It is
    up to you to determine values with odd dimensions. For example the
    output image can be the result of rounding up the division by 2:
    (13, 19) -> (7, 10)

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code
    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].

    Returns:
        numpy.array: output image with half the shape, same type as the
                     input image.
    """
    img = image.copy()
    kernel = np.array([[1,4,6,4,1]]) / 16.0
    kernel = np.dot(kernel.T, kernel)

    reduced = cv2.filter2D(img, -1, kernel)[::2,::2]
    return reduced

def gaussian_pyramid(image, levels):
    """Creates a Gaussian pyramid of a given image.

    This method uses reduce_image() at each level. Each image is
    stored in a list of length equal the number of levels.

    The first element in the list ([0]) should contain the input
    image. All other levels contain a reduced version of the previous
    level.

    All images in the pyramid should floating-point with values in

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        levels (int): number of levels in the resulting pyramid.

    Returns:
        list: Gaussian pyramid, list of numpy.arrays.
    """
    pyramid = [image]

    for i in range(1,levels):
      img = cv2.GaussianBlur(pyramid[-1], (5,5), 0.05)
      pyramid.append(reduce_image(img))

    return pyramid

def create_combined_img(img_list):
    """Stacks images from the input pyramid list side-by-side.

    Ordering should be large to small from left to right.

    See the problem set instructions for a reference on how the output
    should look like.

    Make sure you call normalize_and_scale() for each image in the
    pyramid when populating img_out.

    Args:
        img_list (list): list with pyramid images.

    Returns:
        numpy.array: output image with the pyramid images stacked
                     from left to right.
    """
    new_img = img_list[0]
    h, w = new_img.shape[:2]
    new_img = normalize_and_scale(new_img)
    for img in img_list[1:]:
      (new_h,new_w) = img.shape[:2]
      img = normalize_and_scale(img)
      img = np.concatenate((img, np.zeros((h - new_h, new_w))), axis = 0)
      new_img = np.concatenate((new_img, img), axis = 1)


    return new_img

def expand_image(image):
    """Expands an image doubling its width and height.

    For simplicity and efficiency, implement a convolution-based
    method using the 5-tap separable filter.

    Follow the process shown in the lecture 6B-L3. Also refer to:
    -  Burt, P. J., and Adelson, E. H. (1983). The Laplacian Pyramid
       as a Compact Image Code

    You can find the link in the problem set instructions.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].

    Returns:
        numpy.array: same type as 'image' with the doubled height and
                     width.
    """
    img = image.copy()
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    exp = np.zeros((h*2, w*2))
    exp[::2, ::2] = img

    k = np.array([[1,4,6,4,1]])/8.0

    k = np.dot(k.T,k)

    exp = cv2.filter2D(exp, -1, k)

    return exp

def laplacian_pyramid(g_pyr):
    """Creates a Laplacian pyramid from a given Gaussian pyramid.

    This method uses expand_image() at each level.

    Args:
        g_pyr (list): Gaussian pyramid, returned by gaussian_pyramid().

    Returns:
        list: Laplacian pyramid, with l_pyr[-1] = g_pyr[-1].
    """
    gaussian_pyramid = g_pyr[::-1]
    lap_pyr = [gaussian_pyramid[0]]
    for i, img in enumerate(gaussian_pyramid[:-1]):
      img = expand_image(img)
      h1,w1 = img.shape[:2]
      h2,w2 = gaussian_pyramid[i+1].shape[:2]
      if not h1 == h2:
        #img = img[:h2 - h1, :]
        #img = img[:h2, :]
        img = img[(h1-h2)/2 : (h1+h2)/2, :]
      if not w1 == w2:
        #img = img[:, :w2 - w1]
        #img = img[:, :w2]
        img = img[:, (w1-w2)/2 : (w1+w2)/2]
      lap_pyr.append(gaussian_pyramid[i+1] - img)
    
    return lap_pyr[::-1]

def warp(image, U, V, interpolation, border_mode):
    """Warps image using X and Y displacements (U and V).

    This function uses cv2.remap. The autograder will use cubic
    interpolation and the BORDER_REFLECT101 border mode. You may
    change this to work with the problem set images.

    See the cv2.remap documentation to read more about border and
    interpolation methods.

    Args:
        image (numpy.array): grayscale floating-point image, values
                             in [0.0, 1.0].
        U (numpy.array): displacement (in pixels) along X-axis.
        V (numpy.array): displacement (in pixels) along Y-axis.
        interpolation (Inter): interpolation method used in cv2.remap.
        border_mode (BorderType): pixel extrapolation method used in
                                  cv2.remap.

    Returns:
        numpy.array: warped image, such that
                     warped[y, x] = image[y + V[y, x], x + U[y, x]]
    """
    img = image.copy()
    h = np.shape(img)[0]
    w = np.shape(img)[1]
    map_x, map_y = np.meshgrid(range(w), range(h))
    #map_x = np.ndarray.flatten(U + map_x).astype(np.int_)
    #map_y = np.ndarray.flatten(V + map_y).astype(np.int_)    

    map_x = np.asarray(U + map_x).astype(np.float32)
    map_y = np.asarray(V + map_y).astype(np.float32)    

    #img = cv2.copyMakeBorder(img, h, h, w, w, border_mode)

    warped_img  = cv2.remap(img, map_x, map_y, interpolation, borderMode = border_mode)
    #warped_img = [img[h+y, w+x] for x,y in zip(map_x, map_y)]
    #warped_img = np.asarray(warped_img).reshape(h, w)
    return warped_img

def hierarchical_lk(img_a, img_b, levels, k_size, k_type, sigma, interpolation,
                    border_mode):
    """Computes the optic flow using Hierarchical Lucas-Kanade.

    This method should use reduce_image(), expand_image(), warp(),
    and optic_flow_lk().

    Args:
        img_a (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        img_b (numpy.array): grayscale floating-point image, values in
                             [0.0, 1.0].
        levels (int): Number of levels.
        k_size (int): parameter to be passed to optic_flow_lk.
        k_type (str): parameter to be passed to optic_flow_lk.
        sigma (float): parameter to be passed to optic_flow_lk.
        interpolation (Inter): parameter to be passed to warp.
        border_mode (BorderType): parameter to be passed to warp.

    Returns:
        tuple: 2-element tuple containing:
            U (numpy.array): raw displacement (in pixels) along X-axis,
                             same size as the input images,
                             floating-point type.
            V (numpy.array): raw displacement (in pixels) along Y-axis,
                             same size and type as U.
    """
    GaussianA = gaussian_pyramid(img_a, levels)
    GaussianB = gaussian_pyramid(img_b, levels)
    U = None
    V = None

    for k in range(levels, 0, -1):
      if k == levels:
        U = np.zeros(GaussianA[-1].shape)
        V = np.zeros(GaussianA[-1].shape)
      else:
        h,w = GaussianA[k-1].shape[:2]
        U = expand_image(U)*2.0
        V = expand_image(V)*2.0
        uh, uw = U.shape[:2]
        vh, vw = V.shape[:2]
        if not uh == h:
          #U = U[:h-uh, :]
          #U = U[:h, :]
          U = U[(uh-h)/2 : (uh+h)/2, :]
        if not uw == w:
          #U = U[:, :w-uw]
          #U = U[:, :w]
          U = U[:, (uw-w)/2 : (uw+w)/2]
        if not vh == h:
          #V = V[:h-vh, :]
          #V = V[:h, :]
          V = V[(vh-h)/2 : (vh+h)/2, :]
        if not vw == w:
          #V = V[:, :w-vw]
          #V = V[:, :w]
          V = V[:, (vw-w)/2 : (vw+w)/2]
      warped = warp(GaussianB[k-1], U, V, interpolation, border_mode)
      X, Y = optic_flow_lk(GaussianA[k-1], warped, k_size, k_type, sigma)
      U = U + X
      V = V + Y

    return U, V
