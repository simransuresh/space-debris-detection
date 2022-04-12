""" Import the master rockstar first of all """
import cv2 as cv
""" Import numpy lib for kernel operations """
import numpy as np

""" All the area except debris is black in space. So create a boundary from black (0) to white (255) """
lowerBoundary = np.array([0, 0, 0])
upperBoundary = np.array([0, 0, 255])

""" Capture the video. Video courtesy: Youtube, Movie courtesy: Gravity """
capture = cv.VideoCapture("./assets/debris.mp4")

""" Use two kernels opening and closing for filtering identity matrix of range row*column """
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))


""" Read the capture frame-by-frame """
while True:
    isTrue, frame = capture.read()
    """ Resizing for better visualization """
    frame = cv.resize(frame, (1366,768))

    """ Convert BGR to HSV
        BGR represents color luminance or intensity. It is hard to separate colors from it
        HSV Hue Saturation Value separates image luminance from color information
        HSV is a rearrangement of RGB in a cylindrical shape
    """
    hsv_image= cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    """ Create the Mask that uses kernel for filtering """
    mask = cv.inRange(hsv_image, lowerBoundary, upperBoundary)
    
    """ Pass the mask through the kernel for morphological manipulation """
    maskOpen = cv.morphologyEx(mask, cv.MORPH_OPEN, kernelOpen)
    maskClose = cv.morphologyEx(maskOpen, cv.MORPH_CLOSE, kernelClose)
    
    """ Using the mask obtained and chain-approx-none algorithm, find the contours in it 
        RETR_EXTERNAL - only eldest contour is given preference """
    contours, hierachy = cv.findContours(maskClose.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    
    """ Draw the contours found in the frame. Color is blue and thickness of 3 """
    cv.drawContours(frame, contours, -1, (255,0,0), 3)

    """ Draw red rectangles to distinguish the debris and add yellow text on it """
    for index in range(len(contours)):
        """ Plain x and y coordinate with its width and height """
        x, y, w, h = cv.boundingRect(contours[index])
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.putText(frame, str(index + 1), (x, y + h), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255))

    """ Detection of space debris shown. 
        Anything other than black in the boundary is detected as blue but only debris is marked as text"""
    cv.imshow("Space debris detection output", frame)

    """ Play till q key is pressed """
    if cv.waitKey(10) & 0xFF == ord('q'):
        break

""" Release the video capture and destroy window object """
capture.release()
cv.destroyAllWindows()