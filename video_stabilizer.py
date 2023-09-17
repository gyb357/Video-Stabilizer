import video_manager as vm
import cv2
import numpy as np
import matplotlib.pyplot as plt


path        = 'C:/Users/USER/Desktop/project/video/'
input_path  = 'input.mp4'
output_path = 'output.mp4'

video  = vm.VideoManager(path + input_path, path + output_path)
cap    = video.load()
info   = video.frame_info(cap)
width  = video.info['width']
height = video.info['height']
count  = video.info['count']
out    = video.save('mp4v')

transforms = np.zeros((count - 1, 3), np.float32)


def tracking_point(cap):
    _, prev   = cap.read()
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners   = 500,
        qualityLevel = 0.01,
        minDistance  = 30,
        blockSize    = 3,
        k            = 0.04
    )

    i_arr  = np.array([])
    dx_arr = np.array([])
    dy_arr = np.array([])

    for i in range(count - 2):
        success, curr = cap.read()
        if success == False:
            break

        curr_gray               = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        idx      = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        transform = cv2.estimateAffine2D(prev_pts, curr_pts)[0]
        dx        = transform[0, 2]
        dy        = transform[1, 2]
        da        = np.arctan2(transform[1, 0], transform[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray     = curr_gray
        print("Frame: " + str(i) +  "/" + str(count) + " -  Tracked points : " + str(len(prev_pts)), dx, dy)

        prev_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners   = 500,
            qualityLevel = 0.01,
            minDistance  = 30,
            blockSize    = 3,
            k            = 0.04
        )

        i_arr  = np.append(i_arr, i)
        dx_arr = np.append(dx_arr, dx)
        dy_arr = np.append(dy_arr, dy)

        for pt in curr_pts:
            x, y = pt.ravel()
            cv2.circle(curr, (int(x), int(y)), 1, (0, 255, 255), -1)

        cv2.imshow('Tracked Points', curr)
        cv2.waitKey(1)

    return i_arr, dx_arr, dy_arr

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=50)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


i_arr, dx_arr, dy_arr = tracking_point(cap)
cv2.destroyAllWindows()
trajectory = np.cumsum(transforms, axis=0)
difference = smooth(trajectory) - trajectory
transforms_smooth = transforms + difference


cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


for i in range(count - 2):
    success, frame = cap.read()

    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    frame_stabilized = cv2.warpAffine(frame, m, (width, height))
    frame_stabilized = fixBorder(frame_stabilized)
    frame_out = cv2.hconcat([frame, frame_stabilized])

    #cv2.imshow("Before and After", frame_out)
    #cv2.waitKey(10)
    out.write(frame_stabilized)


cap.release()
out.release()
cv2.destroyAllWindows()


v2   = vm.VideoManager(path + output_path, path + output_path)
cap2 = v2.load()
info = v2.frame_info(cap2)
i_arr2, dx_arr2, dy_arr2 = tracking_point(cap2)


plt.plot(i_arr, dx_arr, '-r', 'dx')
plt.plot(i_arr, dy_arr, '-b', 'dy')
plt.plot(i_arr2, dx_arr2, '-y', 'stab_dx')
plt.plot(i_arr2, dy_arr2, '-g', 'stab_dy')
plt.show()

