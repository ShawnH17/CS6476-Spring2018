"""Problem Set 4: Motion Detection"""

import cv2
import os
import numpy as np
import imageio

import ps4

# I/O directories
input_dir = "input_images"
vid_dir = "input_videos"
output_dir = "output"


# Utility code
def quiver(u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out

def quiver_img(img, u, v, scale, stride, color=(0, 255, 0)):

    img_out = np.asarray(img, dtype = np.uint8)

    #img_out = np.zeros((v.shape[0], u.shape[1], 3), dtype=np.uint8)

    for y in xrange(0, v.shape[0], stride):

        for x in xrange(0, u.shape[1], stride):

            cv2.line(img_out, (x, y), (x + int(u[y, x] * scale),
                                       y + int(v[y, x] * scale)), color, 1)
            cv2.circle(img_out, (x + int(u[y, x] * scale),
                                 y + int(v[y, x] * scale)), 1, color, 1)
    return img_out

# Functions you need to complete:

def scale_u_and_v(u, v, level, pyr):
    """Scales up U and V arrays to match the image dimensions assigned 
    to the first pyramid level: pyr[0].

    You will use this method in part 3. In this section you are asked 
    to select a level in the gaussian pyramid which contains images 
    that are smaller than the one located in pyr[0]. This function 
    should take the U and V arrays computed from this lower level and 
    expand them to match a the size of pyr[0].

    This function consists of a sequence of ps4.expand_image operations 
    based on the pyramid level used to obtain both U and V. Multiply 
    the result of expand_image by 2 to scale the vector values. After 
    each expand_image operation you should adjust the resulting arrays 
    to match the current level shape 
    i.e. U.shape == pyr[current_level].shape and 
    V.shape == pyr[current_level].shape. In case they don't, adjust
    the U and V arrays by removing the extra rows and columns.

    Hint: create a for loop from level-1 to 0 inclusive.

    Both resulting arrays' shapes should match pyr[0].shape.

    Args:
        u: U array obtained from ps4.optic_flow_lk
        v: V array obtained from ps4.optic_flow_lk
        level: level value used in the gaussian pyramid to obtain U 
               and V (see part_3)
        pyr: gaussian pyramid used to verify the shapes of U and V at 
             each iteration until the level 0 has been met.

    Returns:
        tuple: two-element tuple containing:
            u (numpy.array): scaled U array of shape equal to 
                             pyr[0].shape
            v (numpy.array): scaled V array of shape equal to 
                             pyr[0].shape
    """

    # TODO: Your code here
    """
    (uh, uw) = u.shape[:2]
    (vh, vw) = v.shape[:2]
    (h, w) = pyr[level].shape[:2]

    if not uh == h:
        U = U[:h-uh, :]
    if not uw == w:
        U = U[:, :w-uw]
    if not vh == h:
        V = V[:h-vh, :]
    if not vw == w:
        V = V[:, :w-vw]
    """

    for lev in range(level, 0, -1):
        img = pyr[lev]
        (uh, uw) = u.shape[:2]
        (vh, vw) = v.shape[:2]
        (h, w) = img.shape[:2]        
        if not uh == h:
            u = u[:h-uh, :]
        if not uw == w:
            u = u[:, :w-uw]
        if not vh == h:
            v = v[:h-vh, :]
        if not vw == w:
            v = v[:, :w-vw]     
        u,v = ps4.optic_flow_lk(ps4.expand_image(img)*2)
    
    img = pyr[0]
    (uh, uw) = u.shape[:2]
    (vh, vw) = v.shape[:2]
    (h, w) = img.shape[:2]  

    if not uh == h:
        u = u[:h-uh, :]
    if not uw == w:
        u = u[:, :w-uw]
    if not vh == h:
        v = v[:h-vh, :]
    if not vw == w:
        v = v[:, :w-vw]     


    return u,v

def part_1a():

    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    shift_r5_u5 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                          'ShiftR5U5.png'), 0) / 255.
    #shift_0 = cv2.GaussianBlur(shift_0, (15, 15), 0.5)
    #shift_r2 = cv2.GaussianBlur(shift_r2, (15, 15), 0.5)
    #shift_r5_u5 = cv2.GaussianBlur(shift_r5_u5, (15, 15), 0.5)
    #shift_0 = cv2.medianBlur(shift_0, 5)
    #shift_r2 = cv2.medianBlur(shift_r2, 5)
    #shift_r5_u5 = cv2.medianBlur(shift_r5_u5, 5)
    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 25  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=10, stride=25)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-1.png"), u_v)

    # Now let's try with ShiftR5U5. You may want to try smoothing the
    # input images first.

    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 1  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r5_u5, k_size, k_type, sigma)

    # Flow image
    u_v = quiver(u, v, scale=10, stride=25)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-a-2.png"), u_v)


def part_1b():
    """Performs the same operations applied in part_1a using the images
    ShiftR10, ShiftR20 and ShiftR40.

    You will compare the base image Shift0.png with the remaining
    images located in the directory TestSeq:
    - ShiftR10.png
    - ShiftR20.png
    - ShiftR40.png

    Make sure you explore different parameters and/or pre-process the
    input images to improve your results.

    In this part you should save the following images:
    - ps4-1-b-1.png
    - ps4-1-b-2.png
    - ps4-1-b-3.png

    Returns:
        None
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.
    # Optional: smooth the images if LK doesn't work well on raw images
    shift_0 = cv2.GaussianBlur(shift_0, (15, 15), 0.05)
    shift_r10 = cv2.GaussianBlur(shift_r10, (15, 15), 0.05)
    shift_r20 = cv2.GaussianBlur(shift_r20, (15, 15), 0.05)
    shift_r40 = cv2.GaussianBlur(shift_r40, (15, 15), 0.05)



    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 3  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r10, k_size, k_type, sigma)
    # Flow image
    u_v = quiver(u, v, scale=10, stride=25)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-1.png"), u_v)

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r20, k_size, k_type, sigma)
    # Flow image
    u_v = quiver(u, v, scale=10, stride=25)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-2.png"), u_v)

    # Optional: smooth the images if LK doesn't work well on raw images
    k_size = 45  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(shift_0, shift_r40, k_size, k_type, sigma)
    # Flow image
    u_v = quiver(u, v, scale=10, stride=25)
    cv2.imwrite(os.path.join(output_dir, "ps4-1-b-3.png"), u_v)    
    


def part_2():

    yos_img_01 = cv2.imread(os.path.join(input_dir, 'DataSeq1',
                                         'yos_img_01.jpg'), 0) / 255.

    # 2a
    levels = 4
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_01_g_pyr_img = ps4.create_combined_img(yos_img_01_g_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-a-1.png"),
                yos_img_01_g_pyr_img)

    # 2b
    yos_img_01_l_pyr = ps4.laplacian_pyramid(yos_img_01_g_pyr)

    yos_img_01_l_pyr_img = ps4.create_combined_img(yos_img_01_l_pyr)
    cv2.imwrite(os.path.join(output_dir, "ps4-2-b-1.png"),
                yos_img_01_l_pyr_img)


def part_3a_1():
    yos_img_01 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_01.jpg'), 0) / 255.
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_01_g_pyr = ps4.gaussian_pyramid(yos_img_01, levels)
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_01_g_pyr[level_id],
                             yos_img_02_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_02_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_02_warped = ps4.warp(yos_img_02, u, v, interpolation, border_mode)

    diff_yos_img_01_02 = yos_img_01 - yos_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-1.png"),
                ps4.normalize_and_scale(diff_yos_img_01_02))


def part_3a_2():
    yos_img_02 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_02.jpg'), 0) / 255.
    yos_img_03 = cv2.imread(
        os.path.join(input_dir, 'DataSeq1', 'yos_img_03.jpg'), 0) / 255.

    levels = 1  # Define the number of pyramid levels
    yos_img_02_g_pyr = ps4.gaussian_pyramid(yos_img_02, levels)
    yos_img_03_g_pyr = ps4.gaussian_pyramid(yos_img_03, levels)

    level_id = 0  # TODO: Select the level number (or id) you wish to use
    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    u, v = ps4.optic_flow_lk(yos_img_02_g_pyr[level_id],
                             yos_img_03_g_pyr[level_id],
                             k_size, k_type, sigma)

    u, v = scale_u_and_v(u, v, level_id, yos_img_03_g_pyr)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    yos_img_03_warped = ps4.warp(yos_img_03, u, v, interpolation, border_mode)

    diff_yos_img = yos_img_02 - yos_img_03_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-3-a-2.png"),
                ps4.normalize_and_scale(diff_yos_img))


def part_4a():
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r10 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR10.png'), 0) / 255.
    shift_r20 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR20.png'), 0) / 255.
    shift_r40 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                        'ShiftR40.png'), 0) / 255.

    shift_0 = cv2.GaussianBlur(shift_0, (65, 65), 0.05)
    shift_r10 = cv2.GaussianBlur(shift_r10, (65, 65), 0.05)
    shift_r20 = cv2.GaussianBlur(shift_r20, (65, 65), 0.05)
    shift_r40 = cv2.GaussianBlur(shift_r40, (65, 65), 0.05)

    levels = 5  # TODO: Define the number of levels
    k_size = 7  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 2  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    u10, v10 = ps4.hierarchical_lk(shift_0, shift_r10, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u10, v10, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-1.png"), u_v)

    # You may want to try different parameters for the remaining function
    # calls.
    u20, v20 = ps4.hierarchical_lk(shift_0, shift_r20, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)

    u_v = quiver(u20, v20, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-2.png"), u_v)

    u40, v40 = ps4.hierarchical_lk(shift_0, shift_r40, levels, k_size, k_type,
                                   sigma, interpolation, border_mode)
    u_v = quiver(u40, v40, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-a-3.png"), u_v)


def part_4b():
    urban_img_01 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban01.png'), 0) / 255.
    urban_img_02 = cv2.imread(
        os.path.join(input_dir, 'Urban2', 'urban02.png'), 0) / 255.

    levels = 5  # TODO: Define the number of levels
    k_size = 25  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values

    mc01 = cv2.GaussianBlur(mc01, (15, 15), 0.05)
    
    u, v = ps4.hierarchical_lk(urban_img_01, urban_img_02, levels, k_size,
                               k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-1.png"), u_v)

    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    urban_img_02_warped = ps4.warp(urban_img_02, u, v, interpolation,
                                   border_mode)

    diff_img = urban_img_01 - urban_img_02_warped
    cv2.imwrite(os.path.join(output_dir, "ps4-4-b-2.png"),
                ps4.normalize_and_scale(diff_img))


def part_5a():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    shift_0 = cv2.imread(os.path.join(input_dir, 'TestSeq',
                                      'Shift0.png'), 0) / 255.
    shift_r2 = cv2.imread(os.path.join(input_dir, 'TestSeq', 
                                       'ShiftR2.png'), 0) / 255.
    k_size = 25  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    u, v = ps4.optic_flow_lk(shift_0, shift_r2, k_size, k_type, sigma)

    img_0 = ps4.normalize_and_scale(shift_0)
    img_02 = ps4.warp(shift_0, -0.2*u, -0.2*v, interpolation,border_mode)
    img_04 = ps4.warp(shift_0, -0.4*u, -0.4*v, interpolation,border_mode)
    img_06 = ps4.warp(shift_0, -0.6*u, -0.6*v, interpolation,border_mode)
    img_08 = ps4.warp(shift_0, -0.8*u, -0.8*v, interpolation,border_mode)

    img_02 = ps4.normalize_and_scale(img_02)
    img_04 = ps4.normalize_and_scale(img_04)
    img_06 = ps4.normalize_and_scale(img_06)
    img_08 = ps4.normalize_and_scale(img_08)


    img_1 = ps4.normalize_and_scale(shift_r2)
    img_row1 = np.concatenate((img_0, img_02, img_04), axis = 1)
    img_row2 = np.concatenate((img_06, img_08, img_1), axis = 1)
    img_all = np.concatenate((img_row1, img_row2), axis = 0)
    images = [img_0, img_02, img_04, img_06, img_08, img_1]
    imageio.mimsave(os.path.join(output_dir, "ps4-5-1-a-1.gif"), images)

    # Flow image
    u_v = quiver(u, v, scale=3, stride=10)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-00.png"), img_0)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-02.png"), img_02)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-04.png"), img_04)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-06.png"), img_06)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-08.png"), img_08)
    #cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-10.png"), img_1)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-a-1.png"), img_all)



def part_5b():
    """Frame interpolation

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """
    mc01 = cv2.imread(os.path.join(input_dir, 'MiniCooper',
                                      'mc01.png'), 0) / 255.
    mc02 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 
                                       'mc02.png'), 0) / 255.
    mc03 = cv2.imread(os.path.join(input_dir, 'MiniCooper', 
                                       'mc03.png'), 0) / 255.    
    mc01 = cv2.GaussianBlur(mc01, (45, 45), 0.05)
    mc02 = cv2.GaussianBlur(mc02, (45, 45), 0.05)
    mc03 = cv2.GaussianBlur(mc03, (45, 45), 0.05)

    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    levels = 4

    u, v = ps4.hierarchical_lk(mc01, mc02, levels, k_size, k_type, sigma, interpolation, border_mode)

    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-1-quiver.png"), u_v)

    mc_001 = ps4.normalize_and_scale(mc01)
    mc_002 = ps4.warp(mc01, -0.2*u, -0.2*v, interpolation,border_mode)
    mc_004 = ps4.warp(mc01, -0.4*u, -0.4*v, interpolation,border_mode)
    mc_006 = ps4.warp(mc01, -0.6*u, -0.6*v, interpolation,border_mode)
    mc_008 = ps4.warp(mc01, -0.8*u, -0.8*v, interpolation,border_mode)
  
    mc_200 = ps4.normalize_and_scale(mc02)

    mc_002 = ps4.normalize_and_scale(mc_002)
    mc_004 = ps4.normalize_and_scale(mc_004)
    mc_006 = ps4.normalize_and_scale(mc_006)
    mc_008 = ps4.normalize_and_scale(mc_008)

    k_size = 10  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 2  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    levels = 6


    u, v = ps4.hierarchical_lk(mc02, mc03, levels, k_size, k_type, sigma, interpolation, border_mode)
    
    #print "U shape:{}".format(u.shape)

    #print "V shape:{}".format(v.shape)

    #print "img shape: {}".format(mc01.shape)
    u_v = quiver(u, v, scale=3, stride=10)
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-2-quiver.png"), u_v)
    mc_201 = ps4.normalize_and_scale(mc02)
    mc_202 = ps4.warp(mc02, -0.2*u, -0.2*v, interpolation,border_mode)
    mc_204 = ps4.warp(mc02, -0.4*u, -0.4*v, interpolation,border_mode)
    mc_206 = ps4.warp(mc02, -0.6*u, -0.6*v, interpolation,border_mode)
    mc_208 = ps4.warp(mc02, -0.8*u, -0.8*v, interpolation,border_mode)
    mc_300 = ps4.normalize_and_scale(mc03)

    mc_202 = ps4.normalize_and_scale(mc_202)
    mc_204 = ps4.normalize_and_scale(mc_204)
    mc_206 = ps4.normalize_and_scale(mc_206)
    mc_208 = ps4.normalize_and_scale(mc_208)

    mc01_02_row1 = np.concatenate((mc_001, mc_002, mc_004), axis = 1)
    mc01_02_row2 = np.concatenate((mc_006, mc_008, mc_200), axis = 1)
    mc01_02_all = np.concatenate((mc01_02_row1, mc01_02_row2), axis = 0)
    images_01_02 = [mc_001, mc_002, mc_004, mc_006, mc_008, mc_200]
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-1.png"), mc01_02_all)
    imageio.mimsave(os.path.join(output_dir, "ps4-5-1-b-1.gif"), images_01_02)

    mc02_03_row1 = np.concatenate((mc_201, mc_202, mc_204), axis = 1)
    mc02_03_row2 = np.concatenate((mc_206, mc_208, mc_300), axis = 1)
    mc02_03_all = np.concatenate((mc02_03_row1, mc02_03_row2), axis = 0)
    images_02_03 = [mc_201, mc_202, mc_204, mc_206, mc_208, mc_300]
    cv2.imwrite(os.path.join(output_dir, "ps4-5-1-b-2.png"), mc02_03_all)    
    imageio.mimsave(os.path.join(output_dir, "ps4-5-1-b-2.gif"), images_02_03)

def part_6():
    """Challenge Problem

    Follow the instructions in the problem set instructions.

    Place all your work in this file and this section.
    """

    video = os.path.join(vid_dir, "ps4-my-video.mp4")
    frame_gen = video_frame_generator(video)
    #for i in range(150):
    frame1 = frame_gen.next()

    frame2 = frame_gen.next()

    h,w = frame1.shape[:2] 

    out_path = os.path.join(output_dir, "video_out.mp4")
    video_out = mp4_video_writer(out_path, (w, h), fps = 40)

    k_size = 15  # TODO: Select a kernel size
    k_type = "uniform"  # TODO: Select a kernel type
    sigma = 0.5  # TODO: Select a sigma value if you are using a gaussian kernel
    interpolation = cv2.INTER_CUBIC  # You may try different values
    border_mode = cv2.BORDER_REFLECT101  # You may try different values
    levels = 5
    print "img shape: {}".format(frame1.shape)
    cv2.imwrite(os.path.join(output_dir, "frame1.png"), frame1)
    frame_num = 1
    while frame2 is not None and frame1 is not None and frame_num <= 1000:
        print "Processing Frame {}".format(frame_num)
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)/255.0
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)/255.0
        img1 = cv2.GaussianBlur(img1, (15, 15), 0.05)
        img2 = cv2.GaussianBlur(img2, (15, 15), 0.05)        
        #img1 = ps4.normalize_and_scale(img1)
        #img2 = ps4.normalize_and_scale(img2)

        u, v = ps4.hierarchical_lk(img1, img2, levels, k_size, k_type, sigma, interpolation, border_mode)
        quiver_image = quiver_img(frame1, u, v, scale = 3, stride = 10)
        #print "u shape: {}".format(u.shape)
        #print "v shape: {}".format(v.shape)
        if frame_num == 50:
            u_v = quiver_img(frame1, u, v, scale=3, stride=10)
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-1.png"), u_v)    
        elif frame_num == 100:
            u_v = quiver_img(frame1, u, v, scale=3, stride=10)
            cv2.imwrite(os.path.join(output_dir, "ps4-6-a-2.png"), u_v)          

        frame1 = frame2
        frame2 = frame_gen.next()

        video_out.write(quiver_image)
        frame_num += 1
    
    # Flow image
    #u_v = quiver(u, v, scale=3, stride=10)
    #cv2.imwrite(os.path.join(output_dir, "ps4-6-a-1.png"), u_v)
    #cv2.imwrite(os.path.join(output_dir, "ps4-6-a-2.png"), u_v)

def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None

def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.
http://ac.qq.com/ComicView/index/id/541812/cid/185  

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.cv.CV_FOURCC(*'MP4V')
    filename = filename.replace('.mp4', '.avi')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


if __name__ == "__main__":
    part_1a()
    part_1b()
    part_2()
    part_3a_1()
    part_3a_2()
    part_4a()
    part_4b()
    part_5a()
    part_5b()
    part_6()
