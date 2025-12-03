import cv2
import numpy as np
import time

RECT_W_FRAC = 0.25
RECT_H_FRAC = 0.25

# Distance thresholds 
WARNING_DIST_FRAC = 0.15
DANGER_DIST_FRAC = 0.05

# Minimum contour area to be considered of hand
MIN_CONTOUR_AREA = 3000

# Morphology
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

# Default skin color ranges (in YCrCb and HSV) - reasonable starting point
DEFAULT_YCRCB_LOWER = np.array([0, 133, 77])
DEFAULT_YCRCB_UPPER = np.array([255, 173, 127])

DEFAULT_HSV_LOWER = np.array([0, 30, 60])
DEFAULT_HSV_UPPER = np.array([25, 180, 255])

def rect_from_frac(frame_shape, w_frac, h_frac):
    h, w = frame_shape[:2]
    rw, rh = int(w * w_frac), int(h * h_frac)
    cx, cy = w // 2, h // 2
    x1 = cx - rw // 2
    y1 = cy - rh // 2
    x2 = x1 + rw
    y2 = y1 + rh
    return (x1, y1, x2, y2)


def point_to_rect_distance(px, py, rect):
    x1, y1, x2, y2 = rect
    # If point is inside rect, distance is 0
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0
    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return (dx*dx + dy*dy) ** 0.5


def get_fingertip_from_contour(cnt):
    M = cv2.moments(cnt)
    if M['m00'] == 0:
        return None
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    center = np.array([cx, cy])
    pts = cnt.reshape(-1, 2)
    dists = np.linalg.norm(pts - center, axis=1)
    idx = np.argmax(dists)
    return tuple(pts[idx])


def build_mask(frame, ycrcb_range, hsv_range, bg_mask=None):
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(ycrcb, ycrcb_range[0], ycrcb_range[1])
    mask2 = cv2.inRange(hsv, hsv_range[0], hsv_range[1])
    mask = cv2.bitwise_or(mask1, mask2)
    if bg_mask is not None:
        mask = cv2.bitwise_and(mask, bg_mask)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
    mask = cv2.GaussianBlur(mask, (7,7), 0)
    _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
    return mask

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('ERROR: Could not open webcam')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ycrcb_range = (DEFAULT_YCRCB_LOWER.copy(), DEFAULT_YCRCB_UPPER.copy())
    hsv_range = (DEFAULT_HSV_LOWER.copy(), DEFAULT_HSV_UPPER.copy())

    bg_model = None
    use_bg = False

    last_time = time.time()
    fps = 0

    print("Controls: c=calibrate skin (place hand in small box), b=toggle bg-sub, r=reset calib, q=quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Draw the little calibration box top-left
        cal_box = (10, 10, 110, 110)
        cv2.rectangle(frame, (cal_box[0], cal_box[1]), (cal_box[2], cal_box[3]), (255,255,255), 1)
        cv2.putText(frame, 'Calibrate here', (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        fg_mask = None
        if use_bg and bg_model is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, bg_model)
            _, fg_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, KERNEL, iterations=1)

        mask = build_mask(frame, ycrcb_range, hsv_range, fg_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_cnt = None
        if contours:
            # Choose largest contour by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > MIN_CONTOUR_AREA:
                    hand_cnt = cnt
                    break

        fingertip = None
        state = 'SAFE'
        color = (0,255,0)
        dist = None

        # Virtual rectangle
        rect = rect_from_frac(frame.shape, RECT_W_FRAC, RECT_H_FRAC)
        x1,y1,x2,y2 = rect
        # Draw rectangle boundary thicker
        cv2.rectangle(frame, (x1,y1), (x2,y2), (200,200,200), 2)

        if hand_cnt is not None:
            # Draw contour and hull
            cv2.drawContours(frame, [hand_cnt], -1, (120,200,255), 2)
            hull = cv2.convexHull(hand_cnt)
            cv2.drawContours(frame, [hull], -1, (0,255,255), 1)

            # Fingertip
            fingertip = get_fingertip_from_contour(hand_cnt)
            if fingertip is not None:
                cv2.circle(frame, fingertip, 8, (255,0,0), -1)

                # Distance to rectangle
                dist = point_to_rect_distance(fingertip[0], fingertip[1], rect)

                frame_diag = (w*w + h*h) ** 0.5
                if dist <= DANGER_DIST_FRAC * frame_diag:
                    state = 'DANGER'
                    color = (0,0,255)
                elif dist <= WARNING_DIST_FRAC * frame_diag:
                    state = 'WARNING'
                    color = (0,220,220)
                else:
                    state = 'SAFE'
                    color = (0,255,0)

        overlay_h = 40
        cv2.rectangle(frame, (0,0), (250, overlay_h), (50,50,50), -1)
        cv2.putText(frame, f'STATE: {state}', (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # If danger, big warning
        if state == 'DANGER':
            cv2.putText(frame, 'DANGER DANGER', (w//6, h//2), cv2.FONT_HERSHEY_DUPLEX, 2.0, (0,0,255), 4)

        # Show distance numeric for debugging
        if dist is not None:
            cv2.putText(frame, f'Dist: {int(dist)} px', (10, overlay_h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)

        now = time.time()
        dt = now - last_time
        if dt > 0:
            fps = 1.0 / dt
        last_time = now
        cv2.putText(frame, f'FPS: {int(fps)}', (w - 110, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)

        mask_small = cv2.resize(mask, (160,120))
        mh, mw = mask_small.shape[:2]
        frame[10:10+mh, w-10-mw:w-10] = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)

        cv2.imshow('Hand-Boundary POC', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            x1c,y1c,x2c,y2c = cal_box
            sample = frame[y1c:y2c, x1c:x2c]
            if sample.size != 0:
                sample_ycrcb = cv2.cvtColor(sample, cv2.COLOR_BGR2YCrCb)
                sample_hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
                for ch in range(3):
                    mn = np.percentile(sample_ycrcb[:,:,ch], 5)
                    mx = np.percentile(sample_ycrcb[:,:,ch], 95)
                    ycrcb_range[0][ch] = max(0, mn - 10)
                    ycrcb_range[1][ch] = min(255, mx + 10)
                for ch in range(3):
                    mn = np.percentile(sample_hsv[:,:,ch], 5)
                    mx = np.percentile(sample_hsv[:,:,ch], 95)
                    hsv_range[0][ch] = max(0, mn - 10)
                    hsv_range[1][ch] = min(255, mx + 10)
                print('Calibrated skin color ranges (YCrCb and HSV)')
        elif key == ord('b'):
            use_bg = not use_bg
            if use_bg:
                # capture background
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                bg_model = gray
                print('Background model captured, background subtraction ON')
            else:
                bg_model = None
                print('Background subtraction OFF')
        elif key == ord('r'):
            ycrcb_range = (DEFAULT_YCRCB_LOWER.copy(), DEFAULT_YCRCB_UPPER.copy())
            hsv_range = (DEFAULT_HSV_LOWER.copy(), DEFAULT_HSV_UPPER.copy())
            bg_model = None
            use_bg = False
            print('Calibration reset')

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
