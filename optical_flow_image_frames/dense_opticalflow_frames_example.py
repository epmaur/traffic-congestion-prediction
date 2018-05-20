import cv2
import numpy as np
"""
Script to cut out optical flow images frames starting from 4 seconds before cycle ends until the end of cycle
Video frame rate is 25 frames per second
The first red light is fixed manually, further red lights are calculated using the cycle length
In this example video the first red light of the driving direction of interest is at 01:09
"""

cap = cv2.VideoCapture('../example_video/20180503080000-20180503100000.mp4')
ret, frame1 = cap.read()
frame1 = frame1[200:800, 1000:1900]
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

count = 0
frames = 0

# Length of cycle in frames
cycle = 2250

# First wanted frame (frame count of first red light minus four seconds)
get_frame = 1725 - 4 * 25

while 1:
    ret, frame2 = cap.read()
    frame2 = frame2[200:800, 1000:1900]
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv2.normalize(mag,None,0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', bgr)
    wait_key = cv2.waitKey(15) & 0xff

    # Only every 8th frame is saved from each second
    a = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]

    if ((count - get_frame) in a) or ((count - get_frame) > cycle and (count - get_frame) % cycle in a):
        # Name image after the video
        # Yes/ no shows if intersection was congested or not
        # At first all frames are labeled "yes" and manually changed afterwards
        # Both optical flow and regular frame is saved for inspection purposes
        cv2.imwrite('20180503080000-20180503100000_fb_%d.png' % frames, cv2.resize(frame2, (200, 200), cv2.INTER_CUBIC))
        cv2.imwrite('20180503080000-20180503100000_hsv_%d.png' % frames, cv2.resize(bgr, (64, 64), cv2.INTER_CUBIC))
        frames += 1

    count += 1
    print('count', count)
    prvs = next

cap.release()
cv2.destroyAllWindows()