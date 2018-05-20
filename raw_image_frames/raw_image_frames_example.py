import cv2
"""
Script to cut out raw images frames 4 seconds before cycle ends
Video frame rate is 25 frames per second
The first red light is fixed manually, further red lights are calculated using the cycle length
In this example video the first red light of the driving direction of interest is at 01:09
"""
vidcap = cv2.VideoCapture('../example_video/20180503080000-20180503100000_trimmed.mp4')
success, image = vidcap.read()
count = 0
frames = 0
success = True

# First wanted frame (frame count of first red light minus four seconds)
get_frame = 1725 - 4 * 25

# Cycle length in frames
cycle = 2250
while success:
  if (count == get_frame or (count - get_frame) % cycle == 0):
    # Name image after the video
    # Yes/ no shows if intersection was congested or not
    # At first all frames are labeled "yes" and manually changed afterwards
    cv2.imwrite("20180503080000-20180503100000_%d-yes.jpg" % frames, image[200:800, 1000:1900])
    print('Read a new frame: ', frames)
    frames += 1
  success, image = vidcap.read()
  count += 1
