import cv2
import mediapipe as mp
import numpy as np
import time
import sys


# CONFIG (DEFAULT = EASY MODE)

SQUAT_GOAL = 20
CURL_GOAL = 20
ARM_GOAL_TIME = 40

MIN_DIST = 0.10
MAX_DIST = 0.22

FONT = cv2.FONT_HERSHEY_SIMPLEX


# STATES

MENU = 0
DIFFICULTY = 6
CUSTOM_SETUP = 7

SQUAT = 1
CURL = 2
ARM = 3
FINISH = 4
EXIT = 5

exercise_names = {
    SQUAT: "Squat",
    CURL: "Bicep Curl",
    ARM: "Straighten Arm"
}

state = MENU
selected_exercise = None
selected_mode = "easy"
finish_message = ""


# CUSTOM VALUES

custom_squat = 20
custom_curl = 20
custom_arm = 40


# INIT

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils
draw_style = mp_draw.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3)
cap = cv2.VideoCapture(0)


# SESSION DATA

squat_reps = 0
curl_reps = 0
arm_time = 0
stage = None
last_time = time.time()

def calculate_angle(a, b, c):
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    ang = abs(rad * 180 / np.pi)
    return 360-ang if ang > 180 else ang

def draw_center_text(frame, text, y_ratio, scale=1.2, thickness=3, color=(0,0,0)):
    h, w, _ = frame.shape
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = (w - tw) // 2
    y = int(h * y_ratio)
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness)

def draw_button(frame, xr, yr, wr, hr, text):
    h, w, _ = frame.shape
    x = int(xr * w)
    y = int(yr * h)
    bw = int(wr * w)
    bh = int(hr * h)

    scale = bh / 45
    thickness = max(1, int(scale * 2))

    cv2.rectangle(frame, (x,y), (x+bw,y+bh), (50,50,50), -1)
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    cv2.putText(frame, text,
                (x + (bw-tw)//2, y + (bh+th)//2),
                FONT, scale, (255,255,255), thickness)
    return (x,y,x+bw,y+bh)

def inside(x, y, box):
    return box[0] < x < box[2] and box[1] < y < box[3]

def check_distance(lm):
    l = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    r = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    d = abs(l.x - r.x)
    if d < MIN_DIST: return "too far"
    if d > MAX_DIST: return "too close"
    return "ok"

# MOUSE HANDLER
def mouse(event, x, y, flags, param):
    global state, selected_exercise, selected_mode
    global SQUAT_GOAL, CURL_GOAL, ARM_GOAL_TIME
    global custom_squat, custom_curl, custom_arm

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # ---------------- MENU ----------------
    if state == MENU:
        if inside(x,y,btn_squat):
            selected_exercise = SQUAT
            state = DIFFICULTY
        if inside(x,y,btn_curl):
            selected_exercise = CURL
            state = DIFFICULTY
        if inside(x,y,btn_arm):
            selected_exercise = ARM
            state = DIFFICULTY
        if inside(x,y,btn_quit):
            state = EXIT

    # ---------------- DIFFICULTY ----------------
    elif state == DIFFICULTY:
        if inside(x,y,btn_easy):
            selected_mode = "easy"
            SQUAT_GOAL = 20
            CURL_GOAL = 20
            ARM_GOAL_TIME = 40
            state = selected_exercise

        if inside(x,y,btn_normal):
            selected_mode = "normal"
            SQUAT_GOAL = 40
            CURL_GOAL = 40
            ARM_GOAL_TIME = 80
            state = selected_exercise

        if inside(x,y,btn_hard):
            selected_mode = "hard"
            SQUAT_GOAL = 60
            CURL_GOAL = 60
            ARM_GOAL_TIME = 120
            state = selected_exercise

        if inside(x,y,btn_custom):
            state = CUSTOM_SETUP

        if inside(x,y,btn_back):
            state = MENU

    # ---------------- CUSTOM SETUP ----------------
    elif state == CUSTOM_SETUP:
        if inside(x,y,btn_sq_plus): custom_squat += 5
        if inside(x,y,btn_sq_minus): custom_squat = max(5, custom_squat-5)

        if inside(x,y,btn_cu_plus): custom_curl += 5
        if inside(x,y,btn_cu_minus): custom_curl = max(5, custom_curl-5)

        if inside(x,y,btn_ar_plus): custom_arm += 5
        if inside(x,y,btn_ar_minus): custom_arm = max(5, custom_arm-5)

        if inside(x,y,btn_start):
            SQUAT_GOAL = custom_squat
            CURL_GOAL = custom_curl
            ARM_GOAL_TIME = custom_arm
            state = selected_exercise

        if inside(x,y,btn_back):
            state = DIFFICULTY

    # ---------------- EXERCISE ----------------
    elif state in (SQUAT, CURL, ARM):
        if inside(x,y,btn_back):
            state = MENU
        if inside(x,y,btn_quit):
            state = EXIT

    # ---------------- FINISH ----------------
    elif state == FINISH:
        if inside(x,y,btn_menu):
            state = MENU
        if inside(x,y,btn_quit):
            state = EXIT


cv2.namedWindow("Rehab System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Rehab System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("Rehab System", mouse)

# MAIN LOOP

while cap.isOpened():

    if state == EXIT:
        break

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    h,w,_ = frame.shape

    now = time.time()
    dt = now - last_time
    last_time = now


    if state == MENU:
        cv2.rectangle(frame,(0,0),(w,h),(230,230,230),-1)
        draw_center_text(frame, "Rehabilitation System", 0.12, 1.4)
        draw_center_text(frame, "Select Exercise", 0.20, 1.0)

        btn_squat = draw_button(frame, 0.35,0.32,0.3,0.08,"Squat")
        btn_curl  = draw_button(frame, 0.35,0.44,0.3,0.08,"Bicep Curl")
        btn_arm   = draw_button(frame, 0.35,0.56,0.3,0.08,"Straighten Arm")
        btn_quit  = draw_button(frame, 0.35,0.70,0.3,0.08,"Quit App")

    
    # DIFFICULTY MENU
    
    elif state == DIFFICULTY:
        cv2.rectangle(frame,(0,0),(w,h),(220,220,220),-1)

        draw_center_text(frame, f"Exercise: {exercise_names[selected_exercise]}", 0.15, 1.3)
        draw_center_text(frame, "Choose Difficulty", 0.25, 1.0)

        btn_easy   = draw_button(frame, 0.35,0.35,0.3,0.08,"Easy")
        btn_normal = draw_button(frame, 0.35,0.47,0.3,0.08,"Normal")
        btn_hard   = draw_button(frame, 0.35,0.59,0.3,0.08,"Hard")
        btn_custom = draw_button(frame, 0.35,0.71,0.3,0.08,"Custom")

        btn_back   = draw_button(frame, 0.05,0.85,0.2,0.08,"Back")


    elif state == CUSTOM_SETUP:
        cv2.rectangle(frame,(0,0),(w,h),(210,210,210),-1)

        draw_center_text(frame, f"Custom Mode - {exercise_names[selected_exercise]}", 0.15, 1.2)

        # -------- Squat
        draw_center_text(frame, f"Squat Reps: {custom_squat}", 0.30, 1.0)
        btn_sq_minus = draw_button(frame, 0.30,0.35,0.12,0.07,"-")
        btn_sq_plus  = draw_button(frame, 0.58,0.35,0.12,0.07,"+")

        # -------- Curl
        draw_center_text(frame, f"Curl Reps: {custom_curl}", 0.48, 1.0)
        btn_cu_minus = draw_button(frame, 0.30,0.53,0.12,0.07,"-")
        btn_cu_plus  = draw_button(frame, 0.58,0.53,0.12,0.07,"+")

        # -------- Arm
        draw_center_text(frame, f"Arm Hold (s): {custom_arm}", 0.66, 1.0)
        btn_ar_minus = draw_button(frame, 0.30,0.71,0.12,0.07,"-")
        btn_ar_plus  = draw_button(frame, 0.58,0.71,0.12,0.07,"+")

        btn_start = draw_button(frame, 0.35,0.83,0.3,0.08,"Start Exercise")
        btn_back  = draw_button(frame, 0.05,0.85,0.2,0.08,"Back")


    elif state in (SQUAT, CURL, ARM) and results.pose_landmarks:

        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS,
                               draw_style, draw_style)

        lm = results.pose_landmarks.landmark
        dist_status = check_distance(lm)
        coaching_text = ""

        cv2.putText(frame, f"Exercise: {exercise_names[state]}",
                    (40,40), FONT, 0.9, (0,0,0), 3)

        btn_back = draw_button(frame, 0.72,0.05,0.23,0.07,"Back")
        btn_quit = draw_button(frame, 0.72,0.14,0.23,0.07,"Quit")

        if dist_status != "ok":
            msg = "Move closer" if dist_status == "too far" else "Move further"
            cv2.putText(frame, msg, (40,120), FONT, 0.9, (0,0,255), 3)

        # ---------- SQUAT ----------
        if state == SQUAT:
            hip = lm[mp_pose.PoseLandmark.LEFT_HIP]
            knee = lm[mp_pose.PoseLandmark.LEFT_KNEE]
            ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            angle = calculate_angle(hip,knee,ankle)

            if angle < 90:
                coaching_text = "Good depth"
                stage = "down"
            elif angle > 160:
                coaching_text = "Stand tall"
            else:
                coaching_text = "Go lower"

            if dist_status == "ok" and angle > 160 and stage == "down":
                squat_reps += 1
                stage = None

            if squat_reps >= SQUAT_GOAL:
                finish_message = "Squat Completed!"
                state = FINISH

        # ---------- CURL ----------
        if state == CURL:
            # LEFT
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            le = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]

            # RIGHT
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

            left_angle = calculate_angle(ls, le, lw)
            right_angle = calculate_angle(rs, re, rw)

            left_flex = left_angle < 60
            right_flex = right_angle < 60

            left_extend = left_angle > 160
            right_extend = right_angle > 160

            if left_flex and right_flex:
                coaching_text = "Good curl"
                stage = "up"
            elif left_extend and right_extend:
                coaching_text = "Extend fully"
            else:
                coaching_text = "Curl both arms together"

            if (dist_status == "ok"
                and left_extend and right_extend
                and stage == "up"):
                curl_reps += 1
                stage = None


            if curl_reps >= CURL_GOAL:
                finish_message = "Curl Completed!"
                state = FINISH

        # ---------- STRAIGHT ARM ----------
        if state == ARM:
            # LEFT arm
            ls = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
            le = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
            lw = lm[mp_pose.PoseLandmark.LEFT_WRIST]
            lh = lm[mp_pose.PoseLandmark.LEFT_HIP]

            # RIGHT arm
            rs = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            re = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
            rw = lm[mp_pose.PoseLandmark.RIGHT_WRIST]
            rh = lm[mp_pose.PoseLandmark.RIGHT_HIP]

            # Angles
            left_elbow = calculate_angle(ls, le, lw)
            right_elbow = calculate_angle(rs, re, rw)

            left_abd = calculate_angle(lh, ls, le)
            right_abd = calculate_angle(rh, rs, re)

            left_height_ok = abs(le.y - ls.y) < 0.05
            right_height_ok = abs(re.y - rs.y) < 0.05

            left_ok = left_elbow > 160 and 70 < left_abd < 110 and left_height_ok
            right_ok = right_elbow > 160 and 70 < right_abd < 110 and right_height_ok

            if not left_ok and not right_ok:
                coaching_text = "Raise both arms"
            elif not left_ok:
                coaching_text = "Fix left arm position"
            elif not right_ok:
                coaching_text = "Fix right arm position"
            else:
                coaching_text = "Good hold"
                if dist_status == "ok":
                    arm_time += dt
            if arm_time >= ARM_GOAL_TIME:
                finish_message = "Straighten Arm Completed!"
                state = FINISH


        # -------- Display counter --------
        if state == ARM:
            text = f"Time: {int(arm_time)}/{ARM_GOAL_TIME}s"
        else:
            goal = SQUAT_GOAL if state == SQUAT else CURL_GOAL
            count = squat_reps if state == SQUAT else curl_reps
            text = f"Reps: {count}/{goal}"

        cv2.putText(frame, text, (40,80), FONT, 1.0, (0,0,0), 3)
        cv2.putText(frame, f"Coaching: {coaching_text}",
                    (40, h - 40), FONT, 0.9, (0,0,255), 2)

    
    # FINISH SCREEN
    
    elif state == FINISH:
        cv2.rectangle(frame,(0,0),(w,h),(210,210,210),-1)
        draw_center_text(frame, finish_message, 0.35, 1.4)

        btn_menu = draw_button(frame, 0.35,0.50,0.3,0.08,"Menu")
        btn_quit = draw_button(frame, 0.35,0.62,0.3,0.08,"Quit")

        squat_reps = curl_reps = arm_time = 0
        stage = None

    cv2.imshow("Rehab System", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()








