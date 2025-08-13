import numpy as np
import cv2
from sklearn.metrics import pairwise


class FingerCount:
    def __init__(self):
        self.background = None

        self.accumulated_weight = 0.5

        self.roi_top = 20
        self.roi_bottom = 300
        self.roi_right = 300
        self.roi_left = 600


    def calc_accum_avg(self, frame, accumulated_weight):
        if self.background is None:
            self.background = frame.copy().astype('float')
            return None

        cv2.accumulateWeighted(frame, self.background, accumulated_weight)


    def segment(self, frame, threshold_min=25):
        diff = cv2.absdiff(self.background.astype('uint8'), frame)

        ret, thresholded = cv2.threshold(diff, threshold_min, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        else:
            # ASSUMING THE LARGEST EXTERNAL CONTOUR IN ROI, IS THE HAND
            hand_segment = max(contours, key=cv2.contourArea)

            return (thresholded, hand_segment)

    def count_finger(self, thresholded, hand_segment):

        conv_hull = cv2.convexHull(hand_segment)

        # TOP
        top = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
        bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
        left = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
        right = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])

        cX = (left[0] + right[0]) // 2
        cY = (top[1] + bottom[1]) // 2

        distance = pairwise.euclidean_distances([[cX, cY]], Y=[left, right, top, bottom])[0]

        max_distance = distance.max()

        radius = int(0.9 * max_distance)
        circumfrence = (2 * np.pi * radius)

        circular_roi = np.zeros(thresholded.shape, dtype='uint8')

        cv2.circle(circular_roi, (cX, cY), radius, 255, 10)

        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

        contours, hierarchy = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0

        for cnt in contours:

            (x, y, w, h) = cv2.boundingRect(cnt)

            out_of_wrist = (cY + (cY * 0.25)) > (y + h)

            limit_points = ((circumfrence * 0.25) > cnt.shape[0])

            if out_of_wrist and limit_points:
                count += 1

        return count

if __name__ == '__main__':

    finger_count = FingerCount()

    cam = cv2.VideoCapture(0)

    num_frames = 0



    while True:

        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)
        frame_copy = frame.copy()

        roi = frame[finger_count.roi_top:finger_count.roi_bottom, finger_count.roi_right:finger_count.roi_left]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if num_frames < 60:
            finger_count.calc_accum_avg(gray, finger_count.accumulated_weight)

            if num_frames <= 59:
                cv2.putText(frame_copy, 'WAIT. GETTING BACKGROUND', (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.imshow('Finger Count', frame_copy)
        else:
            hand = finger_count.segment(gray)

            if hand is not None:
                thresholded, hand_segment = hand

                # DRAWS CONTOURS AROUND REAL HAND IN LIVE STREAM
                cv2.drawContours(frame_copy, [hand_segment + (finger_count.roi_right, finger_count.roi_top)], -1, (255, 0, 0), 5)

                fingers = finger_count.count_finger(thresholded, hand_segment)

                cv2.putText(frame_copy, str(fingers), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                cv2.imshow('Thresholded', thresholded)

        cv2.rectangle(frame_copy, (finger_count.roi_left, finger_count.roi_top), (finger_count.roi_right, finger_count.roi_bottom), (0, 0, 255), 5)

        num_frames += 1

        cv2.imshow('Finger Count', frame_copy)

        k = cv2.waitKey(1)

        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

