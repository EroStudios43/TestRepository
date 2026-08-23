import cv2
import mediapipe as mp
import numpy as np
import math


# ============================================================
# MediaPipe setup
# ============================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


# ============================================================
# MediaPipe landmark indices
# ============================================================

LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

LEFT_EYE_CORNERS = [33, 133]
RIGHT_EYE_CORNERS = [362, 263]

LEFT_EYE_LIDS = [159, 145]
RIGHT_EYE_LIDS = [386, 374]

LEFT_EYE_OUTLINE = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246
]

RIGHT_EYE_OUTLINE = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398
]


# ============================================================
# Gaze angle thresholds
# ============================================================

CENTER_ANGLE = 30

LEFT_ANGLE_MIN = 135
LEFT_ANGLE_MAX = 180

RIGHT_ANGLE_MIN = -180
RIGHT_ANGLE_MAX = -135

UP_ANGLE_MIN = -135
UP_ANGLE_MAX = -45

DOWN_ANGLE_MIN = 45
DOWN_ANGLE_MAX = 135


# ============================================================
# Look-away detection settings
# ============================================================

# Number of consecutive frames before the warning appears.
ALERT_FRAMES = 15

off_center_count = 0


# ============================================================
# Eye processing function
# ============================================================

def draw_eye_angular(
    face_landmarks,
    iris_idxs,
    corner_idxs,
    lid_idxs,
    outline_idxs,
    frame,
    w,
    h,
    label
):
    # --------------------------------------------------------
    # Eye outline
    # --------------------------------------------------------

    pts = np.array([
        (
            int(face_landmarks.landmark[i].x * w),
            int(face_landmarks.landmark[i].y * h)
        )
        for i in outline_idxs
    ])

    cv2.polylines(
        frame,
        [pts],
        True,
        (0, 255, 255),
        1
    )


    # --------------------------------------------------------
    # Eye corners
    # --------------------------------------------------------

    corners = []

    for i in corner_idxs:
        x = int(face_landmarks.landmark[i].x * w)
        y = int(face_landmarks.landmark[i].y * h)

        corners.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            2,
            (50, 255, 50),
            -1
        )


    # --------------------------------------------------------
    # Eyelids
    # --------------------------------------------------------

    y_top = face_landmarks.landmark[lid_idxs[0]].y * h
    y_bottom = face_landmarks.landmark[lid_idxs[1]].y * h

    for i in lid_idxs:
        x = int(face_landmarks.landmark[i].x * w)
        y = int(face_landmarks.landmark[i].y * h)

        cv2.circle(
            frame,
            (x, y),
            2,
            (0, 165, 255),
            -1
        )


    # --------------------------------------------------------
    # Iris
    # --------------------------------------------------------

    iris_pts = []

    for i in iris_idxs:
        x = int(face_landmarks.landmark[i].x * w)
        y = int(face_landmarks.landmark[i].y * h)

        iris_pts.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            2,
            (255, 105, 180),
            -1
        )


    # Calculate iris center
    (cx, cy), _ = cv2.minEnclosingCircle(
        np.array(iris_pts, dtype=np.int32)
    )

    cx = int(cx)
    cy = int(cy)

    cv2.circle(
        frame,
        (cx, cy),
        2,
        (0, 0, 255),
        -1
    )


    # --------------------------------------------------------
    # Calculate normalized iris position
    # --------------------------------------------------------

    eye_left = min(corners[0][0], corners[1][0])
    eye_right = max(corners[0][0], corners[1][0])

    eye_width = eye_right - eye_left

    if eye_width > 0:
        iris_x_ratio = (cx - eye_left) / eye_width
    else:
        iris_x_ratio = 0.5

    eye_top = min(y_top, y_bottom)
    eye_bottom = max(y_top, y_bottom)

    eye_height = eye_bottom - eye_top

    if eye_height > 0:
        iris_y_ratio = (cy - eye_top) / eye_height
    else:
        iris_y_ratio = 0.5

    cv2.putText(
        frame,
        f"{label}: iris X = {iris_x_ratio:.2f}",
        (20, 150 if label == "L" else 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"{label}: iris Y = {iris_y_ratio:.2f}",
        (20, 290 if label == "L" else 325),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.line(
        frame,
        (eye_left, cy),
        (eye_right, cy),
        (255, 255, 255),
        1
    )

    

    # θ is in [-180°, +180°]
    #
    # 0°       = right
    # ±90°     = up/down
    # ±180°    = left


    # --------------------------------------------------------
    # Classify horizontal gaze
    # --------------------------------------------------------

    if iris_x_ratio < 0.40:
        direction = "Left"

    elif iris_x_ratio > 0.60:
        direction = "Right"

    else:
        direction = "Center"

    cv2.putText(
        frame,
        f"{label}: {direction}",
        (20, 80 if label == "L" else 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    if iris_y_ratio > 0.40:
        vertical = "Up"

    elif iris_y_ratio < 0.15:
        vertical = "Down"

    else:
        vertical = "Center"

    return direction, vertical


# ============================================================
# Camera
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
    exit()


# ============================================================
# Main loop
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not read frame.")
        break


    # Mirror image
    frame = cv2.flip(frame, 1)


    # Convert BGR → RGB for MediaPipe
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # Frame dimensions
    h, w, _ = frame.shape


    # Process frame
    results = face_mesh.process(rgb_frame)

    # ========================================================
    # Face detected
    # ========================================================

    if results.multi_face_landmarks:

        for face_landmarks in results.multi_face_landmarks:

            # ------------------------------------------------
            # Left eye
            # ------------------------------------------------

           left_direction, left_vertical = draw_eye_angular(
            face_landmarks,
            LEFT_IRIS,
            LEFT_EYE_CORNERS,
            LEFT_EYE_LIDS,
            LEFT_EYE_OUTLINE,
            frame,
            w,
            h,
            "L"
        )

        right_direction, right_vertical = draw_eye_angular(
            face_landmarks,
            RIGHT_IRIS,
            RIGHT_EYE_CORNERS,
            RIGHT_EYE_LIDS,
            RIGHT_EYE_OUTLINE,
            frame,
            w,
            h,
            "R"
        )

        # Horizontal gaze
        if left_direction == right_direction:
            horizontal_gaze = left_direction
        else:
            horizontal_gaze = "Center"


        # Vertical gaze
        if left_vertical == right_vertical:
            vertical_gaze = left_vertical
        else:
            vertical_gaze = "Center"

        cv2.putText(
        frame,
            f"Horizontal: {horizontal_gaze}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Vertical: {vertical_gaze}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        looking_away = (
            horizontal_gaze != "Center"
            or
            vertical_gaze != "Center"
        )

        if looking_away:
            off_center_count += 1
        else:
            off_center_count = 0


        # ------------------------------------------------
        # Warning
        # ------------------------------------------------

        if off_center_count > ALERT_FRAMES:

            overlay = frame.copy()

            cv2.rectangle(
                overlay,
                (0, 0),
                (w, h),
                (0, 0, 255),
                -1
            )

            cv2.addWeighted(
                overlay,
                0.3,
                frame,
                0.7,
                0,
                frame
            )

            cv2.putText(
                frame,
                "FOCUS!",
                (w // 3, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                4
            )


    # ========================================================
    # Display
    # ========================================================

    cv2.imshow(
        "Eye Tracking",
        frame
    )


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# Cleanup
# ============================================================

cap.release()
cv2.destroyAllWindows()
face_mesh.close()