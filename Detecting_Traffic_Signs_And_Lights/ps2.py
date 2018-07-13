"""
CS6476 Problem Set 2 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import math

import numpy as np

def mask_red_on(img):
    #Use HSV to define the state of the traffic light
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_red_on = np.array([0,50,150])
    upper_red_on = np.array([10,255,255])
    mask_red_on_0 = cv2.inRange(hsv_img,lower_red_on, upper_red_on)
    #upper mask
    lower_red_on = np.array([170,50,150])
    upper_red_on = np.array([180,255,255])
    mask_red_on_1 = cv2.inRange(hsv_img, lower_red_on, upper_red_on)
    mask_red_on = mask_red_on_1 + mask_red_on_0
    mask_red_on = cv2.bitwise_and(img,img, mask= mask_red_on)
    mask_red_on = cv2.cvtColor(mask_red_on, cv2.COLOR_BGR2GRAY)
    return mask_red_on

def mask_yellow_on(img):
    #Use HSV to define the state of the traffic light
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # define range of yellow color in HSV
    lower_yellow_on = np.array([20,50,150])
    upper_yellow_on = np.array([40,255,255])
    mask_yellow_on = cv2.inRange(hsv_img, lower_yellow_on, upper_yellow_on)
    mask_yellow_on = cv2.bitwise_and(img,img, mask= mask_yellow_on)
    mask_yellow_on = cv2.cvtColor(mask_yellow_on, cv2.COLOR_BGR2GRAY)
    return mask_yellow_on

def mask_green_on(img):
    #Use HSV to define the state of the traffic light
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
    lower_green_on = np.array([50,50,150])
    upper_green_on = np.array([70,255,255])
    mask_green_on = cv2.inRange(hsv_img, lower_green_on, upper_green_on)
    mask_green_on = cv2.bitwise_and(img,img, mask= mask_green_on)
    mask_green_on = cv2.cvtColor(mask_green_on, cv2.COLOR_BGR2GRAY)
    return mask_green_on

def mask_red_all(img):
    #Use HSV to find red light location disregard of whether it is on or not
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_red_all = np.array([0,50,50])
    upper_red_all = np.array([10,255,255])
    mask_red_all_0 = cv2.inRange(hsv_img, lower_red_all, upper_red_all)
    #upper mask
    lower_red_all = np.array([170,50,50])
    upper_red_all = np.array([180,255,255])
    mask_red_all_1 = cv2.inRange(hsv_img, lower_red_all, upper_red_all)
    mask_red_all = mask_red_all_1 + mask_red_all_0
    mask_red_all = cv2.bitwise_and(img,img, mask= mask_red_all)
    mask_red_all = cv2.cvtColor(mask_red_all, cv2.COLOR_BGR2GRAY)   
    return mask_red_all 

def mask_yellow_all(img):
    #Use HSV to find yellow light location disregard of whether it is on or not
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    
    # define range of yellow color in HSV
    lower_yellow_all = np.array([20,50,50])
    upper_yellow_all = np.array([40,255,255])
    mask_yellow_all = cv2.inRange(hsv_img, lower_yellow_all, upper_yellow_all)
    mask_yellow_all = cv2.bitwise_and(img,img, mask= mask_yellow_all)
    mask_yellow_all = cv2.cvtColor(mask_yellow_all, cv2.COLOR_BGR2GRAY)
    return mask_yellow_all

def mask_green_all(img):
    #Use HSV to find green light location disregard of whether it is on or not
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)      
    # define range of green color in HSV
    lower_green_all = np.array([50,50,50])
    upper_green_all = np.array([70,255,255])
    mask_green_all = cv2.inRange(hsv_img, lower_green_all, upper_green_all)
    mask_green_all = cv2.bitwise_and(img,img, mask= mask_green_all)
    mask_green_all = cv2.cvtColor(mask_green_all, cv2.COLOR_BGR2GRAY) 
    return mask_green_all

def hough_circles(img):
    return cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 1, 50, param1=30, param2=10,minRadius=3, maxRadius=30)

def traffic_light_detection(img_in, radii_range):
    """Finds the coordinates of a traffic light image given a radii
    range.

    Use the radii range to find the circles in the traffic light and
    identify which of them represents the yellow light.

    Analyze the states of all three lights and determine whether the
    traffic light is red, yellow, or green. This will be referred to
    as the 'state'.

    It is recommended you use Hough tools to find these circles in
    the image.

    The input image may be just the traffic light with a white
    background or a larger image of a scene containing a traffic
    light.

    Args:
        img_in (numpy.array): image containing a traffic light.
        radii_range (list): range of radii values to search for.

    Returns:
        tuple: 2-element tuple containing:
        coordinates (tuple): traffic light center using the (x, y)
                             convention.
        state (str): traffic light state. A value in {'red', 'yellow',
                     'green'}
    """
    output = img_in.copy()
    image1 = img_in.copy()
    #Mask red light color when it is on
    _mask_red_on = mask_red_on(image1)
    #Mask yellow light color when it is on
    _mask_yellow_on = mask_yellow_on(image1)
    #Mask yellow light color when it is on
    _mask_green_on = mask_green_on(image1)

    #Detect if it is Traffic light and find its centroid
    #find mask red light regardless of its state
    _mask_red_all = mask_red_all(image1)
    #find mask yellow light regardless of its state
    _mask_yellow_all = mask_yellow_all(image1)
    #find mask green light regardless of its state
    _mask_green_all = mask_green_all(image1)  

    # find circles using HoughCicles
    # Find the traffic light that is on
    circles_red_on = hough_circles(_mask_red_on)
    circles_yellow_on = hough_circles(_mask_yellow_on)
    circles_green_on = hough_circles(_mask_green_on)
    #find the light regardless whether it is on
    circles_red_all = hough_circles(_mask_red_all)
    circles_yellow_all = hough_circles(_mask_yellow_all)
    circles_green_all = hough_circles(_mask_green_all)

    #determine if it is a traffic light
    center_x = None
    center_y = None
    center_r = None
    #loop over the circles and find the right cenroid
    if (circles_red_all is not None) & (circles_yellow_all is not None) & (circles_green_all is not None) :
        circles_red = np.round(circles_red_all[0, :]).astype("int")
        circles_yellow = np.round(circles_yellow_all[0, :]).astype("int")
        circles_green = np.round(circles_green_all[0, :]).astype("int")
        for (rd_x, rd_y, rd_r) in circles_red:
            for(yl_x, yl_y, yl_r) in circles_yellow:
                for(gr_x, gr_y, gr_r) in circles_green:
                    # Determine if this is traffic light, and find the centroid
                    if ((yl_x - 5 <= rd_x <= yl_x + 5) & (gr_x - 5 <= rd_x <= gr_x + 5) & (gr_y > yl_y > rd_y) & (gr_y - yl_y -10 <= yl_y - rd_y <= gr_y - yl_y + 10)) :
                        center_x = yl_x
                        center_y = yl_y
                        center_r = yl_r
    #determine tl state
    state = None
    #if nothing was found
    if (center_x is None) or (center_y is None):
        return (None, None),None
    #determine tl state
    if circles_red_on is not None:
        circles_red_on = np.round(circles_red_on[0, :]).astype("int")
        for(x, y, r) in circles_red_on:
            if (center_r - 2<= r <= center_r + 2):
                state = 'red'

    if circles_green_on is not None:
        circles_green_on = np.round(circles_green_on[0, :]).astype("int")
        for(x, y, r) in circles_green_on:
            if (center_r - 2<= r <= center_r + 2):
                state = 'green'
                
    if circles_yellow_on  is not None:
        circles_yellow_on = np.round(circles_yellow_on[0, :]).astype("int")
        for(x, y, r) in circles_yellow_on:
            if (center_r - 2 <= r <= center_r + 2):
                state = 'yellow'

    return (center_x, center_y), state

    raise NotImplementedError


def yield_sign_detection(img_in):
    """Finds the centroid coordinates of a yield sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of coordinates of the center of the yield sign.
    """
    output = img_in.copy()
    image1 = img_in.copy()  
    # Mask red color
    _mask_red_all = mask_red_all(image1)
    #Dilate the image
    gray = cv2.dilate(_mask_red_all, np.ones((5,5)))
    #Use Canny to find the outline
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    #Use HoughlinesP to find lines in yield sign
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 20,maxLineGap = 50)[0].tolist()
    #Remove duplicate lines
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                del lines[index2]

    N = np.shape(lines)[0]

    #find the upper horizontal bar and calculate the centroid of the equilateral triangle
    center_x = None
    center_y = None
    state = ''
    count = 0
    # determine state and find centroid
    if (N == 6):
        state = 'yield'
    for index1, (x1, y1, x2, y2) in enumerate(lines):
        for index2, (x3, y3, x4, y4) in enumerate(lines):
            if(y1==y2 and y3==y4 and y1 < y3):        
                center_x = np.int(abs(x2+x1)/2)
                center_y = np.int(y1 + (abs(x2-x1)*np.cos(np.pi/6) - (abs(x2-x1)/2)/np.cos(np.pi/6)))

    return center_x, center_y
    raise NotImplementedError


def stop_sign_detection(img_in):
    """Finds the centroid coordinates of a stop sign in the provided
    image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the stop sign.
    """
    output = img_in.copy()
    image1 = img_in.copy()
    #Mask red color
    _mask_red_all = mask_red_all(image1)
    #dilate image
    gray = cv2.dilate(_mask_red_all, np.ones((5,5)))
    #find outline with canny
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 20,maxLineGap = 50)[0].tolist()
    #remove duplicates
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                del lines[index2]

    N = np.shape(lines)[0]

    # find center
    center_x = None
    center_y = None
    state = ''
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000

    if (N == 8):
        state = 'stop'

    for i in range(N):    
        x1 = lines[i][0]
        y1 = lines[i][1]    
        x2 = lines[i][2]
        y2 = lines[i][3]    
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)       
        
    center_x = np.int((max_x + min_x)/2.0)
    center_y = np.int((max_y + min_y)/2.0)
            
    return center_x, center_y
    raise NotImplementedError


def warning_sign_detection(img_in):
    """Finds the centroid coordinates of a warning sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    output = img_in.copy()
    image1 = img_in.copy()
    #mask yellow color
    _mask_yellow_all = mask_yellow(image1)
    #dilate image
    gray = cv2.dilate(_mask_yellow_all, np.ones((5,5)))
    # Use Canny to find the outline
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    #Use HoughlinesP to find the lines on the sign
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 40,minLineLength = 20,maxLineGap = 50)[0].tolist()
    #Remove duplicates
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                del lines[index2]

    N = np.shape(lines)[0]    

    # find center
    center_x = None
    center_y = None
    state = ''
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000

    if (N == 4):
        state = 'warning'

    for i in range(N):    
        x1 = lines[i][0]
        y1 = lines[i][1]    
        x2 = lines[i][2]
        y2 = lines[i][3]    
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)       
        
    center_x = np.int((max_x + min_x)/2.0)
    center_y = np.int((max_y + min_y)/2.0)
            
    return center_x, center_y

    raise NotImplementedError


def construction_sign_detection(img_in):
    """Finds the centroid coordinates of a construction sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) tuple of the coordinates of the center of the sign.
    """
    output = img_in.copy()
    image1 = img_in.copy()
    #Use HSV to mask the sign
    _mask_orange_all = mask_orange(image1)
    #dilate
    gray = cv2.dilate(_mask_orange_all, np.ones((5,5)))
    #Use canny to find the outlines
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    #Use houghlinesP to find the circles
    lines = cv2.HoughLinesP(edges,0.2,np.pi/180, threshold = 30,minLineLength = 40,maxLineGap = 100)[0].tolist()
    #Remove duplicate lines
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                del lines[index2]

    N = np.shape(lines)[0]
    # find centers
    center_x = None
    center_y = None
    state = ''
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000

    if (N == 4):
        state = 'construction'

    for i in range(N):    
        x1 = lines[i][0]
        y1 = lines[i][1]    
        x2 = lines[i][2]
        y2 = lines[i][3]    
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)        
        
    center_x = np.int((max_x + min_x)/2.0)
    center_y = np.int((max_y + min_y)/2.0)
            
    return center_x, center_y

    raise NotImplementedError


def do_not_enter_sign_detection(img_in):
    """Find the centroid coordinates of a do not enter sign in the
    provided image.

    Args:
        img_in (numpy.array): image containing a traffic light.

    Returns:
        (x,y) typle of the coordinates of the center of the sign.
    """
    output = img_in.copy()
    image1 = img_in.copy()
    #Use HSV to mask the sign
    _mask_red_all = mask_red_all(image1)
    #Dilate image
    gray = cv2.dilate(_mask_red_all, np.ones((5,5)))
    #Canny
    edges = cv2.Canny(gray,50,250,apertureSize=5)   
    lines = cv2.HoughLinesP(edges,0.2,np.pi/180, threshold = 30,minLineLength = 40,maxLineGap = 100)[0].tolist()    
    #Remove duplicates
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                del lines[index2]

    N = np.shape(lines)[0]  
    # find center
    center_x = None
    center_y = None
    state = ''
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000

    if (N == 2):
        state = 'no_entry'
        
    center_x = np.int((lines[0][0] + lines[0][2])/2.0)
    center_y = np.int((lines[0][1] + lines[1][1])/2.0)
            
    return center_x, center_y
    raise NotImplementedError


def lineMagnitude (x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2)+ math.pow((y2 - y1), 2))
    return lineMagnitude

def DistancePointLine(px, py, x1, y1, x2, y2):
    ix = lineMagnitude(px, py, x1, y1)
    iy = lineMagnitude(px, py, x2, y2)
    return min(ix, iy)

def get_distance(line1, line2):
    dist1 = DistancePointLine(line1[0][0], line1[0][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist2 = DistancePointLine(line1[1][0], line1[1][1], 
                              line2[0][0], line2[0][1], line2[1][0], line2[1][1])
    dist3 = DistancePointLine(line2[0][0], line2[0][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])
    dist4 = DistancePointLine(line2[1][0], line2[1][1], 
                              line1[0][0], line1[0][1], line1[1][0], line1[1][1])


    return min(dist1,dist2,dist3,dist4)

def group_lines(lines):
    super_lines = []
    min_distance_to_merge = 20
    lines = sorted(lines, key= lambda _line:_line[0][0])
    
    for idx1, line in enumerate(lines):
        create_new_group = True
        group_updated = False

        for group in super_lines:
            for line3 in group:
                if (1 <= get_distance(line3, line) and get_distance(line3, line) <= min_distance_to_merge):
                    group.append(line)
                    create_new_group = False
                    group_updated = True
                    break
            if group_updated:
                break
        
        #create new group is lines are not close enough
        if (create_new_group):
            new_group = []
            new_group.append(line)
            for idx2, line2 in enumerate(lines):
                # check the distance between lines
                if (1 <= get_distance(line2, line) <= min_distance_to_merge):
                    new_group.append(line2)
            # append new group
            super_lines.append(new_group)

    return super_lines

def mask_red(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_red_on = np.array([0,50,50])
    upper_red_on = np.array([10,255,255])
    mask_red_on_0 = cv2.inRange(hsv_img, lower_red_on, upper_red_on)
    #upper mask
    lower_red_on = np.array([170,50,50])
    upper_red_on = np.array([180,255,255])
    mask_red_on_1 = cv2.inRange(hsv_img, lower_red_on, upper_red_on)
    mask_red_on = mask_red_on_1 + mask_red_on_0
    mask_red_on = cv2.bitwise_and(img,img, mask= mask_red_on)
    mask_red_on = cv2.cvtColor(mask_red_on, cv2.COLOR_BGR2GRAY)
    return mask_red_on

def mask_orange(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_orange_on = np.array([10,50,50])
    upper_orange_on = np.array([20,255,255])
    mask_orange_on = cv2.inRange(hsv_img, lower_orange_on, upper_orange_on)

    mask_orange_on  = cv2.bitwise_and(img,img, mask= mask_orange_on )
    mask_orange_on  = cv2.cvtColor(mask_orange_on , cv2.COLOR_BGR2GRAY)
    return mask_orange_on

def mask_yellow(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_yellow_on = np.array([30,50,50])
    upper_yellow_on = np.array([40,255,255])
    mask_yellow_on = cv2.inRange(hsv_img, lower_yellow_on, upper_yellow_on)
    mask_yellow_on  = cv2.bitwise_and(img,img, mask= mask_yellow_on )
    mask_yellow_on  = cv2.cvtColor(mask_yellow_on , cv2.COLOR_BGR2GRAY)  
    return mask_yellow_on

def traffic_sign_detection(img_in):
    """Finds all traffic signs in a synthetic image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    #return traffic_light_detection(img_in, (3,30))
    image1= img_in.copy()
    #Use red mask to find out red signs
    gray = mask_red(image1)
    #Filter, erode, dilate
    kernel = np.ones((7,7),np.uint8)
    gray = cv2.filter2D(gray,-1,kernel)
    gray = cv2.erode(gray,kernel,iterations=1)
    gray = cv2.dilate(gray, np.ones((5,5)))
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 20,maxLineGap = 20)[0].tolist()
    #remove duplicate lines
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-20<=y3<=y1+20 and y2-20<=y4<= y2+20 and x1-20<=x3<=x1+20 and x2-20<=x4<=x2+20: # Horizontal Lines
                del lines[index2]
    _lines = []
    for _line in lines:
        _lines.append([(_line[0], _line[1]),(_line[2], _line[3])])
    #group lines by calculating the distance between their ends
    gp_lines = group_lines(_lines)
    #remove duplicates within each group
    for i in range(np.shape(gp_lines)[0]):
        for index1, line1 in enumerate(gp_lines[i]):
            for index2, line2 in enumerate(gp_lines[i]):
                if index1!=index2 and line1[0][0]-5<=line2[0][0]<=line1[0][0]+5 and line1[0][1]-5<=line2[0][1]<= line1[0][1]+5: # Horizontal Lines
                    del gp_lines[i][index2]           

    # find center
    length = np.shape(gp_lines)[0]
    center_x = None
    center_y = None
    dictionary = {}
    state = ''
    count = 0
    for i in range(length):
        #yield sign
        if (len(gp_lines[i]) == 6):
            state = 'yield'
            for index1, [[x1, y1], [x2, y2]] in enumerate(gp_lines[i]):
                for index2, [[x3, y3], [x4, y4]] in enumerate(gp_lines[i]):
                    if(y1==y2 and y3==y4 and y1 < y3):        
                        center_x = np.int(math.fabs(x2+x1)/2)
                        center_y = np.int(y1 + (math.fabs(x2-x1)*math.cos(math.pi/6) - (math.fabs(x2-x1)/2)/math.cos(math.pi/6)))
            dictionary[state]=(center_x, center_y)
        #stop sign
        if(len(gp_lines[i]) == 8):
            state = 'stop'
            max_x = 0
            min_x = 1000
            max_y = 0
            min_y = 1000
            for [[x1,y1],[x2,y2]] in gp_lines[i]:       
                max_x = max(max_x, x1, x2)
                max_y = max(max_y, y1, y2)
                min_x = min(min_x, x1, x2)
                min_y = min(min_y, y1, y2)
            center_x = np.int((max_x + min_x)/2.0)
            center_y = np.int((max_y + min_y)/2.0)
            dictionary[state]=(center_x, center_y)
    #find no_entry sign
    circles = cv2.HoughCircles(edges, cv2.cv.CV_HOUGH_GRADIENT, 1, 100, param1=40, param2=20,minRadius=10, maxRadius=50)[0].tolist()
    circles = np.round(circles).astype("int")
    dictionary['no_entry'] = (circles[0][0],circles[0][1])

    (x,y), s = traffic_light_detection(image1, (3,30))
    if s is not None:
        dictionary[s] = (x,y)
    #Find construction sign, mask orange
    gray = mask_orange(image1)
    kernel = np.ones((7,7),np.uint8)
    gray = cv2.filter2D(gray,-1,kernel)
    gray = cv2.erode(gray,kernel,iterations=1)
    gray = cv2.dilate(gray, np.ones((5,5)))
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 20,maxLineGap = 20)[0].tolist()
    #remove duplicate
    for index1, (x1,y1,x2,y2) in enumerate(lines):
        for index2, (x3,y3,x4,y4) in enumerate(lines):
            if index1!=index2 and y1-20<=y3<=y1+20 and y2-20<=y4<= y2+20 and x1-20<=x3<=x1+20 and x2-20<=x4<=x2+20: # Horizontal Lines
                del lines[index2]
    N = np.shape(lines)[0]
    M = np.shape(lines)[1]
    # find construction sign center
    center_x = None
    center_y = None
    state = ''
    max_x = 0
    min_x = 1000
    max_y = 0
    min_y = 1000

    if (N == 4):
        state = 'construction'

    for i in range(N):    
        x1 = lines[i][0]
        y1 = lines[i][1]    
        x2 = lines[i][2]
        y2 = lines[i][3]    
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)        
        
    center_x = np.int((max_x + min_x)/2.0)
    center_y = np.int((max_y + min_y)/2.0)            
    dictionary[state] = (center_x,center_y)

    #find warning sign, mask yellow
    gray = mask_yellow(image1)
    kernel = np.ones((7,7),np.uint8)
    gray = cv2.filter2D(gray,-1,kernel)
    gray = cv2.erode(gray,kernel,iterations=1)
    gray = cv2.dilate(gray, np.ones((5,5)))
    edges = cv2.Canny(gray,50,250,apertureSize=5)  
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 20,minLineLength = 20,maxLineGap = 20)
    if lines is not None:
        lines = lines[0].tolist()
        #remove duplicate
        for index1, (x1,y1,x2,y2) in enumerate(lines):
            for index2, (x3,y3,x4,y4) in enumerate(lines):
                if index1!=index2 and y1-20<=y3<=y1+20 and y2-20<=y4<= y2+20 and x1-20<=x3<=x1+20 and x2-20<=x4<=x2+20: # Horizontal Lines
                    del lines[index2]
        N = np.shape(lines)[0]
        M = np.shape(lines)[1]
        # find warning sign center
        center_x = None
        center_y = None
        state = ''
        max_x = 0
        min_x = 1000
        max_y = 0
        min_y = 1000
        if (N == 4):
            state = 'warning'
        for i in range(N):    
            x1 = lines[i][0]
            y1 = lines[i][1]    
            x2 = lines[i][2]
            y2 = lines[i][3]    
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
            min_x = min(min_x, x1, x2)
            min_y = min(min_y, y1, y2)         
        center_x = np.int((max_x + min_x)/2.0)
        center_y = np.int((max_y + min_y)/2.0)            
        dictionary[state] = (center_x,center_y)
    return dictionary
    raise NotImplementedError

def traffic_sign_detection_noisy(img_in):
    """Finds all traffic signs in a synthetic noisy image.

    The image may contain at least one of the following:
    - traffic_light
    - no_entry
    - stop
    - warning
    - yield
    - construction

    Use these names for your output.

    See the instructions document for a visual definition of each
    sign.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    image1 = img_in.copy()
    #Denoise fileter
    dst = cv2.fastNlMeansDenoisingColored(image1,None,10,10,7,21)
    #Use traffice_sign_detection function to find the right traffic sign
    dictionary=traffic_sign_detection(dst)
    return dictionary
    raise NotImplementedError

def mask_yellow_real(img):
    #convert to HSV to mask the image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #lower mask
    lower_orange_on = np.array([20,150,150])
    upper_orange_on = np.array([30,255,255])
    mask_orange_on = cv2.inRange(hsv_img, lower_orange_on, upper_orange_on)

    mask_orange_on  = cv2.bitwise_and(img,img, mask= mask_orange_on )
    mask_orange_on  = cv2.cvtColor(mask_orange_on , cv2.COLOR_BGR2GRAY)   
    return mask_orange_on 

def traffic_sign_detection_challenge(img_in):
    """Finds traffic signs in an real image

    See point 5 in the instructions for details.

    Args:
        img_in (numpy.array): input image containing at least one
                              traffic sign.

    Returns:
        dict: dictionary containing only the signs present in the
              image along with their respective centroid coordinates
              as tuples.

              For example: {'stop': (1, 3), 'yield': (4, 11)}
              These are just example values and may not represent a
              valid scene.
    """
    image1 = img_in.copy()
    #denoising
    image1 = cv2.fastNlMeansDenoisingColored(image1,None,20,20,7,21)
    _mask_yellow_real = mask_yellow_real(image1)
    kernel = np.ones((5,5),np.uint8)
    gray = cv2.filter2D(_mask_yellow_real,-1,kernel)
    gray = cv2.erode(gray,kernel,iterations=1)
    gray = cv2.dilate(gray, np.ones((5,5)))
    edges = cv2.Canny(gray,50,250,apertureSize=5)
    lines = cv2.HoughLinesP(edges,0.5,np.pi/180, threshold = 40,minLineLength = 20,maxLineGap = 50)
    dictionary = {}
    if lines is not None:
        lines = lines[0].tolist()
        for index1, (x1,y1,x2,y2) in enumerate(lines):
            for index2, (x3,y3,x4,y4) in enumerate(lines):
                if index1!=index2 and y1-5<=y3<=y1+5 and y2-5<=y4<= y2+5 and x1-5<=x3<=x1+5 and x2-5<=x4<=x2+5: # Horizontal Lines
                    del lines[index2]    
        N = np.shape(lines)[0]
        # find center
        center_x = None
        center_y = None
        state = ''
        max_x = 0
        min_x = 1000
        max_y = 0
        min_y = 1000

        if (N == 4):
            state = 'warning'

        for i in range(N):    
            x1 = lines[i][0]
            y1 = lines[i][1]    
            x2 = lines[i][2]
            y2 = lines[i][3]    
            max_x = max(max_x, x1, x2)
            max_y = max(max_y, y1, y2)
            min_x = min(min_x, x1, x2)
            min_y = min(min_y, y1, y2)       
            
        center_x = np.int((max_x + min_x)/2.0)
        center_y = np.int((max_y + min_y)/2.0)
        dictionary[state] = (center_x, center_y)

    return dictionary
    raise NotImplementedError
