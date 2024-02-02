
import numpy as np
import cv2
from pynvml import *
import torch
'''img = cv2.imread('assets/student_db/student_db/train/trash/0004_0257.png', -1)
img = cv2.resize(img, (0, 0), fx=2, fy=2)  # resize image
img = cv2.rotate(img, cv2.cv2.ROTATE_180)
print(img[257])  # look at the pixel value at a certain row of  the image'''
#  change the color of a pixel
'''for i in range(100):  # loop through each rows of the image
    for j in range(img.shape[1]):  # remember the "shapes" give (row. columns, channels) of an image
        # this is to loop through each  column within each row, column represents the width of the image
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]  # pinpoint the xcat pixel 
        #by the
        # specific row and column number'''
#  copy and paste a part of the image
'''tag = img[100: 200, 300:400]  # numpy array slice. this is called
# from row 100 to 200 (within these 2 row number) copy column 300 to 400
img[50:150, 200:300] = tag'''
#  open cv with video capture device
'''cap = cv2.VideoCapture(0)  # use video capture device, o being the first camera in the system
cap2 = cv2.VideoCapture('Video file name ')  # load video file
while True:  # keep displaying the video with a while loop, this loop end when we click a key
    ret, frame = cap.read()  # get a frame from our video capture device
    # ret is an indicator of whether the reading of frames work, if the device is occupied by another process then it
    # will show no
    cv2.imshow('frame', frame)  # creat a window and start showing the frame
    if cv2.waitKey(1) == ord('q'):  # if q is pressed, cease the process
        #  this is accomplished  by computing the ordinal value of the keystroke with reference
        break
cap.release()
cv2.destroyAllWindows()'''
# mirroring videos multiple time
'''cap = cv2.VideoCapture(0)  # use video capture device, o being the first camera in the system
# cap2 = cv2.VideoCapture('Video file name ')  # load video file
while True:  # keep displaying the video with a while loop, this loop end when we click a key
    ret, frame = cap.read()  # get a frame from our video capture device
    # ret is an indicator of whether the reading of frames work, if the device is occupied by another process then it
    # will show no
    width = int(cap.get(3))  # cap.get(number), the number is there indicate what kind of information it can get from
    # from the image, in this case. 3 is get width and 4 is to get with
    height = int(cap.get(4))
    image = np.zeros(frame.shape, np.uint8)  # creat blank canvas, this canvas shares the same detention as the shape
    # of the frame
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # make smaller frame, quarter the size of the original
    # frame this is to make that the frame ( video footage) can be pasted 4 times on the in the black canvas
    image[:height // 2, :width // 2] = cv2.rotate(smaller_frame, cv2.cv2.ROTATE_180) # can also roate the image, but
    # do beware of the rotation angle
    image[height // 2:, :width // 2] = smaller_frame
    image[:height // 2, width // 2:] = smaller_frame
    image[height // 2:, width // 2:] = smaller_frame  # place the first smaller frame on the canvas at top left corner
    # this is 0-0.5width of the frame and 0-0.5 height of the frame.
    # (at this stage, the
    cv2.imshow('frame', image)
    if cv2.waitKey(1) == ord('q'):  # if q is pressed, cease the process
        #  this is accomplished  by computing the ordinal value of the keystroke with reference
        break

cap.release()
cv2.destroyAllWindows()'''

'''cv2.imshow('Image', img)  # the title of the displayed window#
cv2.waitKey(0)  # wait infinite amount of time before you close the window, 0 is infinite, 1 is 1 minute
cv2.destroyAllWindows()  # close the window upon request'''

# drawing on an image with open cv
'''cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    img = cv2.line(frame, (0, 0), (width, height), (0, 0, 255), 10)  # give property of the line (the place this going
    # to be drawn, starting coord (0,0) bing the top left corner, end coord, the color in BGR, thickness in pixels
    img = cv2.line(img, (width, 0), (0, height), (0, 0, 255), 10)
    image = np.zeros(frame.shape, np.uint8)
    smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    img = cv2.rectangle(img, (100, 100), (200,200), (255, 0, 0), 10)
    # property of drawing a rectangle on the image. the canvas it is drawing on, starting coord of top left corner,
    # end coord for bottom right, colour, thickness of the border (-1 = fill the area, other number indicates the
    # thickness
    img = cv2.circle(img, (300, 300), 60, (0, 0, 255), -1)
    # source image, center of the circle, radius, color, border thickness

    # text drawing, font
    font = cv2.FONT_ITALIC
    img = cv2.putText(img, 'Tony is tired', (200, height-80), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    # source image, text content, coord for bottom left of the text, font, font scale, font color, font thickness,
    # line format
    cv2.imshow('frame', img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''

# color identification and detection

'''cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    width = int(cap.get(3))
    height = int(cap.get(4))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # convert the image to a hsv color formate
    # this is essential for using color id
    lower_blue = np.array([110, 50, 50])  # defining a range of blue color
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)  # mask is portion of image which fit the input parameter provided
    result = cv2.bitwise_and(frame, frame, mask=mask)  # this is to apply the mask to the frame
    # simple explanation, comparing the mask to the frame, if it returns 1 then colour blue is found, and display the
    # pixel
    cv2.imshow('frame', result)

    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''

# corner detection, find all the corner of the image

'''img = cv2.imread('assets/student_db/student_db/train/notrash/0001_0155.png')
img = cv2.resize(img, (0, 0), fx=3, fy=3)

grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert any images to grey scale before sending off to process
corners = cv2.goodFeaturesToTrack(grey, 100, 0.01, 10)
# corners properties, source image, number of corner you want to return, corner quality (0= worst, 1 = best}, m
# minial distance between corners (Euclidean distance)
print(corners)  # at this stage corner are represented as floats
corners = np.int0(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 5, (255, 0, 0), 1)
cv2.imshow('Frame', img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''

# templet matching

'''img = cv2.imread('assets/student_db/student_db/train/trash/0002_0266.png', 0)
template = cv2.imread('assets/student_db/student_db/train/trash/0002_0457.png', 0)
# template = cv2.resize(template, (0, 0), fx=4, fy=4)
img = cv2.resize(img, (0, 0), fx=4, fy=4)
img2 = img.copy()
h, w = template.shape
methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCORR, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
for method in methods:
    result = cv2.matchTemplate(img2, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        location = min_loc
    else:
        location = max_loc
    bottom_right = (location[0] + w, location[1] + 1)
    cv2.rectangle(img2, location, bottom_right, 255, 2)
    cv2.imshow("match", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
# face detection
'''cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0, 0), 5)
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 5)
    cv2.imshow('MY FACE !', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()'''
def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()
torch.ones((1, 1)).to("cuda")
print_gpu_utilization()