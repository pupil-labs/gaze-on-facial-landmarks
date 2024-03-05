import warnings
import cv2
import logging
from pupil_labs.map_fixations_on_face.getpoints import circles_overlap, circle_ellipse_overlap
#import math

warnings.filterwarnings("ignore")

# def circles_overlap(center1, radius1, center2, radius2):
#     distance_between_centers = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
#     sum_of_radii = radius1 + radius2
#     return distance_between_centers <= sum_of_radii

# def circle_ellipse_overlap(circle_center, circle_radius, ellipse_center, ellipse_major_axis, ellipse_minor_axis):
#     # Convert the ellipse to a bounding box
#     half_major = ellipse_major_axis / 2
#     half_minor = ellipse_minor_axis / 2

#     # Calculate the distance between the centers
#     distance_x = abs(circle_center[0] - ellipse_center[0])
#     distance_y = abs(circle_center[1] - ellipse_center[1])

#     # Calculate the angle between the major axis and the line connecting the circle center to the ellipse center
#     angle = math.atan2(distance_y, distance_x)

#     # Rotate the distance vector back to the ellipse's coordinate system
#     rotated_x = distance_x * math.cos(angle) + distance_y * math.sin(angle)
#     rotated_y = -distance_x * math.sin(angle) + distance_y * math.cos(angle)

#     # Calculate the distance between the circle center and the closest point on the ellipse
#     closest_x = min(half_major, max(-half_major, rotated_x))
#     closest_y = min(half_minor, max(-half_minor, rotated_y))

#     distance = math.sqrt((rotated_x - closest_x)**2 + (rotated_y - closest_y)**2)

#     # Check if the distance is less than or equal to the circle radius
#     return distance <= circle_radius


## Function to draw bounding box and facial landmarks on a frame
def map_and_draw(frame, face_info, aoi_size, ellipse_size, gaze_crcl_size):
    transparency = 60  # Adjust this value as needed
    # Define colors for landmarks
    default_color = (255, 0, 0)  # Default color for landmarks
    gaze_color = (0, 255, 0)  # Color for the gazed landmark

    # Draw bounding box
    p1 = (int(face_info["p1 x [px]"]), int(face_info["p1 y [px]"]))
    p2 = (int(face_info["p2 x [px]"]), int(face_info["p2 y [px]"]))
    cv2.rectangle(frame, p1, p2, default_color, 2)

    mapped_gaze_on_face = []
    landmark_list = []

    new_landmarks = ["eye left", "eye right", "nose", "mouth"]
    mouth_x = int(
        (face_info["mouth left x [px]"] + face_info["mouth right x [px]"]) / 2
    )
    mouth_y = int(
        (face_info["mouth left y [px]"] + face_info["mouth right y [px]"]) / 2
    )
    major_axis = int(
        abs(face_info["mouth right x [px]"] - face_info["mouth left x [px]"]) / 2
    )
    minor_axis = int(
        abs(face_info["mouth right y [px]"] - face_info["mouth left y [px]"]) / 2
    )
    major_axis += ellipse_size
    minor_axis += ellipse_size

    # Take gaze points
    gaze_timestamp = face_info["timestamp [ns]"]
    gaze_coordinates = (face_info["gaze x [px]"], face_info["gaze y [px]"])
    #logging.info(f"Gaze coordinates {gaze_coordinates}")
    overlay = frame.copy()
    # If gaze has not been mapped on the face based on the Face Mapper enrichment, move on
    if not face_info["gaze on face"]:
        mapped_gaze_on_face.append(
            {
                "gaze_timestamp": gaze_timestamp,
                "landmark": "Not on landmark",
                "gaze coordinates": gaze_coordinates,
                "landmark coordinates": None,
            }
        )
        landmark_list.append("Not on landmark")

    # If the gaze is on face:
    elif face_info["gaze on face"]:
        for landmark in new_landmarks:
            if "mouth" in landmark:
                landmark_coordinates = (mouth_x, mouth_y)
                if circle_ellipse_overlap(
                    gaze_coordinates,
                    gaze_crcl_size,
                    landmark_coordinates,
                    major_axis,
                    minor_axis,
                ):
                    
                    landmark_list.append(landmark)
                    color = gaze_color
    
                else:
                   
                    landmark_list.append("Not on landmark")
                    color= default_color
            
                # Draw the ellipse on the overlay
                cv2.ellipse(overlay, landmark_coordinates, (int(major_axis), int(minor_axis)), 0, 0, 360, color, thickness=-1)
                cv2.addWeighted(overlay, transparency / 255.0, frame, 1 - transparency / 255.0, 0, frame)
            else:
                x_key = f"{landmark} x [px]"
                y_key = f"{landmark} y [px]"
                landmark_coordinates = (int(face_info[x_key]), int(face_info[y_key]))
                if circles_overlap(
                    landmark_coordinates,
                    aoi_size,
                    gaze_coordinates,
                    gaze_crcl_size,
                ):
                    
                    landmark_list.append(landmark)
                    color = gaze_color
                elif not circles_overlap(
                    gaze_coordinates,
                    gaze_crcl_size,
                    landmark_coordinates,
                    aoi_size,
                ):
                    
                    landmark_list.append("Not on landmark")
                    color = default_color
                 # Draw the circle on the overlay
                cv2.circle(overlay, landmark_coordinates, int(aoi_size), color, -1)
                cv2.addWeighted(overlay, transparency / 255.0, frame, 1 - transparency / 255.0, 0, frame)
        mapped_gaze_on_face.append(
            {
                "gaze_timestamp": gaze_timestamp,
                "landmarks": landmark_list,
                "gaze coordinates": gaze_coordinates,
            }
        )
    unique_elements = set(landmark_list)
    unique_list = list(unique_elements)

    logging.info(f"Gaze point mapping outcome: {unique_list}. "
                    )
    return unique_list, mapped_gaze_on_face, gaze_coordinates
