"""Reading KITTI files and the geometry that connects LiDAR and camera.

Coordinate systems (all in metres):
  * LiDAR (velodyne): x forward, y left, z up.
  * Camera (rectified): x right, y DOWN, z forward. KITTI 3D boxes are given
    here, with `location` at the BOTTOM centre of the box.
  * Image: u (column, pixels from the left), v (row, pixels from the top).

The calibration file gives three matrices:
  Tr_velo_to_cam (3x4): LiDAR -> camera (unrectified)
  R0_rect (3x3):       rotation that rectifies the camera frame
  P2 (3x4):            rectified camera -> image of the left colour camera (image_2)
So a LiDAR point goes to the image as:  P2 @ R0_rect @ Tr_velo_to_cam @ [x y z 1].
"""

import numpy as np


def read_calib(path):
    """Return P2 (3x4), R0 (4x4) and Tr_velo_to_cam (4x4) from a calib .txt file."""
    values = {}
    with open(path) as f:
        for line in f:
            if ":" in line:
                key, rest = line.split(":", 1)
                values[key] = np.array([float(x) for x in rest.split()])
    P2 = values["P2"].reshape(3, 4)
    R0 = np.eye(4)
    R0[:3, :3] = values["R0_rect"].reshape(3, 3)
    Tr = np.eye(4)
    Tr[:3, :4] = values["Tr_velo_to_cam"].reshape(3, 4)
    return {"P2": P2, "R0": R0, "Tr": Tr}


def read_velodyne(path):
    """Each point is 4 float32 numbers: x, y, z, reflectance."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_labels(path):
    """Ground-truth objects from a label_2 .txt file.
    Raw columns: type truncated occluded alpha x1 y1 x2 y2 h w l x y z rotation_y.
    We return dimensions as (l, h, w), the order OpenPCDet uses."""
    objects = []
    with open(path) as f:
        for line in f:
            v = line.split()
            h, w, l = map(float, v[8:11])
            objects.append({
                "name": v[0],
                "bbox": np.array(list(map(float, v[4:8]))),
                "dimensions": np.array([l, h, w]),
                "location": np.array(list(map(float, v[11:14]))),
                "rotation_y": float(v[14]),
            })
    return objects


def velo_to_cam(points_xyz, calib):
    """LiDAR points (N, 3) -> rectified camera coordinates (N, 3)."""
    homo = np.hstack([points_xyz, np.ones((len(points_xyz), 1))])  # (N, 4)
    return (homo @ (calib["R0"] @ calib["Tr"]).T)[:, :3]


def cam_to_image(points_cam, P2):
    """Rectified camera points (N, 3) -> pixel coordinates (N, 2) and depth (N,)."""
    homo = np.hstack([points_cam, np.ones((len(points_cam), 1))])
    uvw = homo @ P2.T
    depth = uvw[:, 2]
    return uvw[:, :2] / depth[:, None], depth


def box_corners_cam(location, dims_lhw, rotation_y):
    """The 8 corners (8, 3) of a 3D box in camera coordinates.
    The box is built around the origin, rotated about the camera's y axis
    (the vertical axis, pointing down), then moved to `location`."""
    l, h, w = dims_lhw
    x = np.array([l, l, -l, -l, l, l, -l, -l]) / 2
    y = np.array([0, 0, 0, 0, -h, -h, -h, -h])  # bottom face at y=0, top at y=-h (y points down)
    z = np.array([w, -w, -w, w, w, -w, -w, w]) / 2
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (R @ np.vstack([x, y, z])).T + location


def box_to_image_rect(location, dims_lhw, rotation_y, P2, image_wh):
    """Project a 3D box into the image and return the tightest 2D rectangle
    [x1, y1, x2, y2] around it, clipped to the image. Returns None if the box
    is behind the camera or entirely outside the image."""
    corners = box_corners_cam(location, dims_lhw, rotation_y)
    if np.any(corners[:, 2] < 0.1):  # a corner behind or at the camera: projection breaks
        return None
    uv, _ = cam_to_image(corners, P2)
    w, h = image_wh
    x1, y1 = np.clip(uv.min(axis=0), 0, [w - 1, h - 1])
    x2, y2 = np.clip(uv.max(axis=0), 0, [w - 1, h - 1])
    if x2 - x1 < 1 or y2 - y1 < 1:
        return None
    return np.array([x1, y1, x2, y2])


def iou_2d(box, boxes):
    """IoU between one box [x1,y1,x2,y2] and an (N, 4) array of boxes."""
    if len(boxes) == 0:
        return np.zeros(0)
    boxes = np.asarray(boxes)
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)
