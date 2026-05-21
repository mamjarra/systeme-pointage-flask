import mediapipe as mp
import cv2

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands    = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def hand_up(self, frame):
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        raised  = False

        if results.multi_hand_landmarks:
            for lm in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, lm, self.mp_hands.HAND_CONNECTIONS)
                wrist_y = lm.landmark[0].y
                tip_y   = min(lm.landmark[i].y for i in [8,12,16,20])
                y_all   = [l.y for l in lm.landmark]
                if tip_y < wrist_y and min(y_all) < 0.35:
                    raised = True
                    h, w, _ = frame.shape
                    cv2.putText(frame, "MAIN LEVEE !", (w//2-120, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        return raised, frame