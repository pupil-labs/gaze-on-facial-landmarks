# # Helpers to get overlap between gaze and AOI points
import math


def circles_overlap(center1, radius1, center2, radius2):
    distance_between_centers = math.sqrt(
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    )
    sum_of_radii = radius1 + radius2
    return distance_between_centers <= sum_of_radii


def circle_ellipse_overlap(
    circle_center, circle_radius, ellipse_center, ellipse_major_axis, ellipse_minor_axis
):
    # Convert the ellipse to a bounding box
    half_major = ellipse_major_axis / 2
    half_minor = ellipse_minor_axis / 2

    # Calculate the distance between the centers
    distance_x = abs(circle_center[0] - ellipse_center[0])
    distance_y = abs(circle_center[1] - ellipse_center[1])

    # Calculate the angle between the major axis and the line connecting the circle center to the ellipse center
    angle = math.atan2(distance_y, distance_x)

    # Rotate the distance vector back to the ellipse's coordinate system
    rotated_x = distance_x * math.cos(angle) + distance_y * math.sin(angle)
    rotated_y = -distance_x * math.sin(angle) + distance_y * math.cos(angle)

    # Calculate the distance between the circle center and the closest point on the ellipse
    closest_x = min(half_major, max(-half_major, rotated_x))
    closest_y = min(half_minor, max(-half_minor, rotated_y))

    distance = math.sqrt((rotated_x - closest_x) ** 2 + (rotated_y - closest_y) ** 2)

    # Check if the distance is less than or equal to the circle radius
    return distance <= circle_radius
