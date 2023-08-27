import cv2
import numpy as np

width = 1280
height = 720


class MediapipeHands:
    import mediapipe as mp
    def __init__(self, static_image_mode=False, max_num_hands=2, model_complexity=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.hands = self.mp.solutions.hands.Hands(static_image_mode, max_num_hands, model_complexity,
                                                   min_detection_confidence, min_tracking_confidence)
        self.mpdraw = self.mp.solutions.drawing_utils

    def handsdata(self, frame, auto_draw=False):
        if auto_draw:
            return self.hands.process(frame)
        else:
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            allhands = []
            handstype = []
            results = self.hands.process(frame)
            if results.multi_hand_landmarks != None:
                for hand in results.multi_handedness:
                    for handtype in hand.classification:
                        handstype.append(handtype.label)
                for hand in results.multi_hand_landmarks:
                    singlehand = []
                    for landmark in hand.landmark:
                        singlehand.append((int(landmark.x * width), int(landmark.y * height)))
                    allhands.append(singlehand)
            return allhands, handstype

    def drawLandmarks(self, frame, data, auto_draw=False):
        if auto_draw:
            if data.multi_hand_landmarks is not None:
                for hand in data.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(frame, hand, self.mp.solutions.hands.HAND_CONNECTIONS,
                                               landmark_drawing_spec=self.mpdraw.DrawingSpec(color=(0, 0, 255),
                                                                                             thickness=2,
                                                                                             circle_radius=1))
        else:
            allhands = data
            for myHand in allhands:
                # Draw skeleton lines
                connections = [(0, 1), (1, 2), (2, 3), (3, 4),
                               (0, 5), (5, 6), (6, 7), (7, 8),
                               (0, 17), (17, 18), (18, 19), (19, 20),
                               (5, 9), (9, 13), (13, 17),
                               (9, 10), (10, 11), (11, 12),
                               (13, 14), (14, 15), (15, 16)]

                for connection in connections:
                    x1, y1 = myHand[connection[0]]
                    x2, y2 = myHand[connection[1]]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Blue

                # Draw landmarks
                for i in myHand:
                    cv2.circle(frame, (i[0], i[1]), 4, (255, 255, 255), 1)  # White
                    cv2.circle(frame, (i[0], i[1]), 3, (0, 0, 255), -1)  # Blue

        return frame

def findDistances(handData):
    distMatrix = np.zeros([len(handData), len(handData)], dtype=np.float32)
    palmSize = ((handData[0][0] - handData[9][0]) ** 2 + (handData[0][1] - handData[9][1]) ** 2) ** .5
    for rows in range(0, len(handData)):
        for columns in range(0, len(handData)):
            distMatrix[rows][columns] = (((handData[rows][0] - handData[columns][0]) ** 2 + (
                        handData[rows][1] - handData[columns][1]) ** 2) ** .5) / palmSize
    return distMatrix


def findError(knowngestures, unknownMatrix, keypoints):
    error = 9999999
    idx = -1
    for i in range(len(knowngestures)):
        currenterror = 0
        for rows in keypoints:
            for columns in keypoints:
                currenterror += abs(knowngestures[i][rows][columns] - unknownMatrix[rows][columns])
        if currenterror < error:
            error = currenterror
            idx = i
    return error, idx