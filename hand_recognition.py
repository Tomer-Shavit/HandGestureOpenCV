from cv2 import cv2
import numpy as np

video = cv2.VideoCapture(0)
mask_on = False
key = None
mask = None

# function for returning the 4 corners of the user's manual mask 
def make_mask():
    while True:
        _, im0 = video.read()
        im0_flipped = cv2.flip(im0, 1)
        showCrosshair = False
        fromCenter = False
        r = cv2.selectROI("Image", im0_flipped, fromCenter, showCrosshair)
        x, y, w, h = r
        break
    cv2.destroyWindow("Image")
    return (x, y, w, h)

while True:
    # start recording and flip the frame
    _, frame = video.read()
    flipped_frame = cv2.flip(frame, 1)

    # Track the face
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(flipped_frame, 1.05, 5)

    # Create the mask
    flipped_frame_blured = cv2.GaussianBlur(flipped_frame, (21, 21), 0)
    ycbcr_frame = cv2.cvtColor(flipped_frame_blured, cv2.COLOR_BGR2YCR_CB)
    lower_thresh = np.array([0, 133, 77], np.uint8)
    upper_thresh = np.array([235, 173, 127], np.uint8)
    skin_mask = cv2.inRange(ycbcr_frame, lower_thresh, upper_thresh)
    skin_mask = cv2.morphologyEx(
        skin_mask, cv2.MORPH_OPEN, np.ones((8, 8), np.uint8))

    # Apply rectangle mask to the face
    for x, y, w, h in faces:
        skin_mask = cv2.rectangle(
            skin_mask, (x-5, y-10), (x+w, y+h+70), (0, 0, 0), -1)

    # Checking if the user want to add an additional mask
    if mask_on == True or key == ord("m"):
        if key == ord("m"):
            mask = make_mask()
            mask_on = True
        skin_mask = cv2.rectangle(
            skin_mask, (mask[0], mask[1]), (mask[0]+mask[2], mask[1]+mask[3]), (0, 0, 0), -1)

    # Frame of only the skin
    skin_frame = cv2.bitwise_and(flipped_frame, flipped_frame, mask=skin_mask)

    # Finding the contours
    cnts, _ = cv2.findContours(
        skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Detecting the hand
    try:
        hand_contour = max(cnts, key= lambda x: cv2.contourArea(x))
    except ValueError:
        print("Can't detect hand")
    
    # Approxing the hand, creating a convex hull, and locating the defects
    epsilon = 0.0005*cv2.arcLength(hand_contour, True)
    approx = cv2.approxPolyDP(hand_contour,epsilon, True)
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull) 
    hull_contour = []

    # Creating the hull contour
    for i, h in enumerate([approx]):
        hull_contour.append([cv2.convexHull(h)])
    
    #In case there are no contours and the hull_area = 0
    try:
        hand_area = cv2.contourArea(approx)
        hull_area = cv2.contourArea(hull_contour[0][0])
        ratio = (hand_area/hull_area) * 100
        print(f"Ratio is: {ratio}")
    except ZeroDivisionError as e:
        print(e)    
    
    # Displaying the convex hull contour in purple
    cv2.drawContours(flipped_frame, hull_contour[0], -1, (255,0,255), 2)
    # Displaying contours in green
    cv2.drawContours(flipped_frame, [approx], -1, (0, 255, 0), 3)
        
    # Detect fingers and assigning red dots to defects
    if defects is not None:
        cnt = 0
        for i in range(defects.shape[0]):  # calculate the angle
            s, e, f, d = defects[i][0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            a = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = np.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = np.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  #cosine theorem
            if angle <= np.pi / 2:  # angle less than 90 degree, treat as fingers
                cnt += 1
                cv2.circle(flipped_frame, far, 4, [0, 0, 255], -1)

    # Detecting gestures and showing them on top left
    if cnt == 0:
        if int(ratio) > 86:
            cv2.putText(flipped_frame, "0", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
        elif int(ratio) > 70:
            cv2.putText(flipped_frame, "1", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
        elif int(ratio) > 50:
            cv2.putText(flipped_frame, "Shaka", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
    if cnt == 1:
        cv2.putText(flipped_frame, "2", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
    if cnt == 2:
        if int(ratio) > 70:
            cv2.putText(flipped_frame, "3", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
        elif int(ratio) > 50:
            cv2.putText(flipped_frame, "Spiderman", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
    if cnt == 3:
        if int(ratio) > 70:
            cv2.putText(flipped_frame, "4", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
        elif int(ratio) > 50:
            cv2.putText(flipped_frame, "OK", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)
    if cnt == 4:
        cv2.putText(flipped_frame, "5", (50,50), 2, 1, (0,0,255), 2, cv2.LINE_AA)

    # Displaying the frames
    cv2.imshow("Masked Frame", skin_mask)
   # cv2.imshow("Only Skin Frame", skin_frame)
    cv2.imshow("Main Frame", flipped_frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
