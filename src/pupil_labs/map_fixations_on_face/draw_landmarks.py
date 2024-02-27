import warnings
import cv2
import numpy as np


warnings.filterwarnings("ignore")

## Function to draw bounding box and facial landmarks on a frame
def draw_face_landmarks(frame, face_info, aoi_size, ellipse_size):
    transparency = 60  # Adjust this value as needed
    color = (255, 0, 0)
    # Draw bounding box
    p1 = (int(face_info["p1 x [px]"]), int(face_info["p1 y [px]"]))
    p2 = (int(face_info["p2 x [px]"]), int(face_info["p2 y [px]"]))
    cv2.rectangle(frame, p1, p2, color, 2)

    # Draw facial landmarks as dots
    landmarks = ["eye left", "eye right", "nose", "mouth left", "mouth right"]
    for landmark in landmarks:
        x_key = f"{landmark} x [px]"
        y_key = f"{landmark} y [px]"
        if not np.isnan(face_info[x_key]) and not np.isnan(face_info[y_key]):
            landmark_point = (int(face_info[x_key]), int(face_info[y_key]))

            # Use the same radius as the ellipse around the mouth
            if "mouth" in landmark:
                continue
            else:
                # For eyes and nose, use the same radius as the ellipse
                overlay = frame.copy()
                # Draw the circle on the overlay
                cv2.circle(overlay, landmark_point, int(aoi_size), color, -1)
                # Perform alpha blending to combine the original frame and the overlay
                cv2.addWeighted(overlay, transparency / 255.0, frame, 1 - transparency / 255.0, 0, frame)


    # Draw unified "mouth" AOI as an ellipse
    mouth_left = (int(face_info["mouth left x [px]"]), int(face_info["mouth left y [px]"]))
    mouth_right = (int(face_info["mouth right x [px]"]), int(face_info["mouth right y [px]"]))

    if not np.isnan(mouth_left).any() and not np.isnan(mouth_right).any():
        # Calculate center and axes length for the ellipse
        center_x = int((mouth_left[0] + mouth_right[0]) / 2)
        center_y = int((mouth_left[1] + mouth_right[1]) / 2)
        major_axis = int(abs(mouth_right[0] - mouth_left[0]) / 2)
        minor_axis = int(abs(mouth_right[1] - mouth_left[1]) / 2)

        major_axis += ellipse_size
        minor_axis += ellipse_size
        # Draw the ellipse
        overlay = frame.copy()
        # Draw the ellipse on the overlay
        cv2.ellipse(overlay, (int(center_x), int(center_y)), (int(major_axis), int(minor_axis)), 0, 0, 360, color, thickness=-1)

        # Perform alpha blending to combine the original frame and the overlay
        cv2.addWeighted(overlay, transparency / 255.0, frame, 1 - transparency / 255.0, 0, frame)
