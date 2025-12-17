import cv2
import numpy as np
from ultralytics import YOLO
import math

import utils.json_config

CONFIG = utils.json_config.load("config/detect_obb.json")

coords_diff = (CONFIG['WORKAREA_POSE']['x'], CONFIG['WORKAREA_POSE']['y'])
orientation_diff = CONFIG['WORKAREA_POSE']['orientation_degrees']
pick_height = CONFIG['PICK_HEIGHT']

MODEL_PATH = CONFIG['MODEL_PATH']

model = YOLO(MODEL_PATH)

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

# ArUco markers and correspondig points for plane definition
CORNER_MARKERS = CONFIG['CORNER_MARKERS']
PLANE_POINTS = np.array(CONFIG['CORNER_COORDINATES'], dtype=np.float32)

BOTTLE_STAND_POSE = CONFIG['BOTTLE_STAND_POSE']

NORMAL_BOTTLE = (0, 255, 0)
NORMAL_CAP = (0, 0, 255)
SELECTED_COLOR = (255, 0, 255)

H = None


def normalize_angle(theta):
    return (theta + math.pi) % (2 * math.pi) - math.pi


def local_to_global_pose(x_l, y_l, theta_l, x0, y0, theta0):
    x_g = x0 + x_l * math.cos(theta0) - y_l * math.sin(theta0)
    y_g = y0 + x_l * math.sin(theta0) + y_l * math.cos(theta0)

    theta_g = normalize_angle(theta0 + theta_l)

    return x_g, y_g, theta_g


def detect_all_markers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    return corners, ids


def get_plane_markers(corners, ids):
    if ids is None:
        return None

    ids = ids.flatten()
    pts = {}

    for corner, id_ in zip(corners, ids):
        if id_ in CORNER_MARKERS:
            c = corner[0]
            center = np.mean(c, axis=0)
            pts[id_] = center

    if len(pts) != 4:
        return None

    ordered = np.array([pts[i] for i in CORNER_MARKERS], dtype=np.float32)
    return ordered


def process_frame(frame):
    selected_obj = None

    results = model(frame, conf=0.8, agnostic_nms=True, iou=0.3)

    annotated = results[0].plot()

    corners, ids = detect_all_markers(frame)

    # Show the ArUco markers
    if ids is not None:
        for corner, id_ in zip(corners, ids.flatten()):
            pts = corner[0].astype(int)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)
            center = tuple(np.mean(pts, axis=0).astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{id_}", (pts[0][0], pts[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 150, 255), 2)
    # compute homography
    plane_markers = get_plane_markers(corners, ids)
    if plane_markers is not None:
        H, _ = cv2.findHomography(plane_markers, PLANE_POINTS)

        pts = np.float32(PLANE_POINTS).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, np.linalg.inv(H))
        dst = np.int32(dst)

        cv2.polylines(frame, [dst], True, (0, 255, 0), 2)

    r = results[0]

    bottles = []
    caps = []

    img_h, img_w = r.orig_shape

    bottles_result = []

    # Separate objects by classes
    for i in range(len(r.obb)):
        box_obj = r.obb[i]
        cls_name = results[0].names[int(box_obj.cls[0])]

        x, y, w, h, angle = box_obj.xywhr[0].cpu().numpy()  # normalized
        pts = box_obj.xyxyxyxy[0].cpu().numpy().astype(np.int32)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        try:
            pix = np.array([[x, y, 1]], dtype=np.float32).T
            pt = H @ pix
            pt /= pt[2]
            coords = (pt[0][0], pt[1][0])
        except:
            coords = (1, 1)

        obj = {
            "center": (int(x), int(y)),
            "angle": angle,
            "pts": pts,
            "mask": mask,
            "conf": r.obb[i].conf[0].item(),
            "coords": coords
        }

        if cls_name == 'bottle':
            bottles.append(obj)
        elif cls_name == 'cap':
            caps.append(obj)

    # Vizualisation
    for i in range(len(bottles)):
        b = bottles[i]
        b['cap'] = None
        best_coverage = 0.5
        for c in caps:
            count = cv2.countNonZero(cv2.bitwise_and(b['mask'], c['mask']))
            coverage = count / cv2.countNonZero(c['mask'])

            if coverage > best_coverage:
                b['cap'] = c['center']
                b['cap_coords'] = c['coords']

        b['orientation'] = 0
        if b['cap'] is not None:
            bottle_center = np.array(b['coords'])
            cap_center = np.array(b['cap_coords'])
            vec = cap_center - bottle_center
            angle = np.degrees(np.arctan2(vec[1], vec[0])) + 90
            orientation = angle % 360 - 180
            b['orientation'] = orientation

            if selected_obj is None:
                selected_obj = b

    for b in bottles:
        if b['cap'] is None:
            continue

        color = NORMAL_BOTTLE

        if selected_obj is b:
            color = SELECTED_COLOR

        cv2.drawContours(frame, [b["pts"]], -1, color, 3)
        cv2.line(frame, b["center"], b["cap"], color=(0, 255, 0), thickness=2)
        cv2.circle(frame, b["center"], 5, (0, 255, 0), -1)
        coords = b['coords']
        orientation = b['orientation']

        x0, y0 = coords_diff[0], coords_diff[1]
        theta0 = math.radians(orientation_diff)
        x_l, y_l = coords[0], coords[1]
        theta_l = math.radians(orientation)
        x, y, theta = local_to_global_pose(
            x_l, y_l, theta_l, x0, y0, theta0)
        theta = math.degrees(theta) + 90

        cv2.putText(frame, f"({x:.1f}, {y:.1f})",
                    (b["center"][0] + 5, b["center"][1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"{int(theta)} deg",
                    (b["center"][0] + 5, b["center"][1] + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Bottle {b['conf']:.2f}",
                    (b["center"][0] + 5, b["center"][1] + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        bottles_result.append({
            "x": float(x),
            "y": float(y),
            "yaw": float(theta),
            "conf": float(b["conf"]),
            "selected": selected_obj is b
        })

    for c in caps:
        cv2.drawContours(frame, [c["pts"]], -1, (0, 0, 255), 2)
        cv2.circle(frame, c["center"], 5, (0, 0, 255), -1)

    pick_place = {}
    if selected_obj is not None:

        # Write pick and place command if the robot is available
        x0, y0 = coords_diff[0], coords_diff[1]
        theta0 = math.radians(orientation_diff)
        x_l, y_l = float(selected_obj['coords'][0]), float(
            selected_obj['coords'][1])
        theta_l = math.radians(float(selected_obj.get('orientation', 0)))
        x, y, theta = local_to_global_pose(x_l, y_l, theta_l, x0, y0, theta0)
        theta = math.degrees(theta)
        pick_place = {
            "pick_pose": {
                "x": x,
                "y": y,
                "z": pick_height,
                "roll_degrees": 0,
                "pitch_degrees": 180,
                "yaw_degrees": theta + 90
            },
            "place_pose": {
                "x": BOTTLE_STAND_POSE['x'],
                "y": BOTTLE_STAND_POSE['y'],
                "z": BOTTLE_STAND_POSE['z'],
                "roll_degrees": 57,
                "pitch_degrees": -90,
                "yaw_degrees": 32,
            }
        }

    return {
        "yolo_processed_image": annotated,
        "processed_image": frame,
        "bottles": bottles_result,
        "pick_and_place": pick_place
    }
