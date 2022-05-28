import numpy as np
import cv2

video = cv2.VideoCapture('airbus.mp4')  # Чтение видеопотока
velocityX = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='f')
velocityY = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype='f')
diff_last = 0
save1 = np.array([])
save2 = np.array([])
indexDiff = 0
flag = False

# params for corner detection
feature_params = dict(maxCorners=50,
                      qualityLevel=0.1,
                      minDistance=5,
                      blockSize=5)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = video.read()
old_gray = cv2.cvtColor(old_frame,
                        cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None,
                             **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
safe_dot = []
abc = 0
speedX = 0
speedY = 0

while (1):

    ret, frame = video.read()
    frame_gray = cv2.cvtColor(frame,
                              cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray,
                                           frame_gray,
                                           p0, None,
                                           **lk_params)

    # Select good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    avg = 0
    count = 0
    for g in range(len(good_new)):
        if abs(good_new[g][0] - good_old[g][0]) > 2.8 or abs(good_new[g][1] - good_old[g][1]) > 1.05:
            good_new[g][0] = good_old[g][0] + speedX
            good_new[g][1] = good_old[g][1] + speedY
    if count < 10:
        velocityX[count] = (good_new[0][0] - good_old[0][0])
        velocityY[count] = (good_new[0][1] - good_old[0][1])
        count += 1
    else:
        if len(good_new) > 0:
            velocityX = np.roll(velocityX, -1)
            velocityX[9] = good_new[0][0] - good_old[0][0]
            velocityY = np.roll(velocityY, -1)
            velocityY[9] = good_new[0][1] - good_old[0][1]

    speedX = np.sum(velocityX / count)
    speedY = np.sum(velocityY / count)

    abc = abc + 1
    if abc == 458:
        abc = 458
    if len(good_new) < len(safe_dot):
        flag = True
        for indexDiff in range(len(safe_dot)):
            if indexDiff == len(good_old):
                save1 = np.append(good_old, safe_dot[indexDiff])
                save1 = save1.reshape((len(save1)) >> 1, 2)
                good_old = save1
                save1 = np.append(good_new, [safe_dot[indexDiff][0] + speedX, safe_dot[indexDiff][1] + speedY])
                save1 = save1.reshape((len(save1)) >> 1, 2)
                good_new = save1
                continue
            if safe_dot[indexDiff][0] != good_old[indexDiff][0] and safe_dot[indexDiff][1] != good_old[indexDiff][1]:
                save1 = good_old[:indexDiff]
                save2 = good_old[indexDiff:]
                save1 = np.append(save1, safe_dot[indexDiff])
                save1 = save1.reshape((len(save1)) >> 1, 2)
                save1 = np.append(save1, save2)
                save1 = save1.reshape((len(save1)) >> 1, 2)
                good_old = save1
                save1 = good_new[:indexDiff]
                save2 = good_new[indexDiff:]
                save1 = np.append(save1, [safe_dot[indexDiff][0] + speedX, safe_dot[indexDiff][1] + speedY])
                save1 = save1.reshape((len(save1)) >> 1, 2)
                save1 = np.append(save1, save2)
                save1 = save1.reshape((len(save1)) >> 1, 2)
                good_new = save1
    # draw the tracks
    if (len(good_new) == 10):
        abc = abc
    for i, (new, old) in enumerate(zip(good_new,
                                       good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        frame = cv2.circle(frame, (round(a), round(b)), 5, color[i].tolist(), -1)
        mask = cv2.line(mask, (round(a), round(b)), (round(c), round(d)), color[i].tolist(), 2)

    img = cv2.add(frame, mask)

    safe_dot = good_new
    cv2.imshow('frame', img)

    k = cv2.waitKey(25)
    if k == 27:
        break

    # Updating Previous frame and points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

video.release()
cv2.destroyAllWindows()
