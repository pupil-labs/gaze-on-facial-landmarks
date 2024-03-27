import warnings
import cv2
import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from pupil_labs.gaze_on_facial_landmarks.getpoints import (
    circles_overlap,
    circle_ellipse_overlap,
)
warnings.filterwarnings("ignore")


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
    # logging.info(f"Gaze coordinates {gaze_coordinates}")
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
                    color = default_color

                # Draw the ellipse on the overlay
                cv2.ellipse(
                    overlay,
                    landmark_coordinates,
                    (int(major_axis), int(minor_axis)),
                    0,
                    0,
                    360,
                    color,
                    thickness=-1,
                )
                cv2.addWeighted(
                    overlay,
                    transparency / 255.0,
                    frame,
                    1 - transparency / 255.0,
                    0,
                    frame,
                )
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
                cv2.addWeighted(
                    overlay,
                    transparency / 255.0,
                    frame,
                    1 - transparency / 255.0,
                    0,
                    frame,
                )
        mapped_gaze_on_face.append(
            {
                "gaze_timestamp": gaze_timestamp,
                "landmarks": landmark_list,
                "gaze coordinates": gaze_coordinates,
            }
        )
    unique_elements = set(landmark_list)
    unique_list = list(unique_elements)
    logging.info(f"Gaze point mapping outcome: {unique_list}. ")
    return unique_list, mapped_gaze_on_face, gaze_coordinates

def get_percentages(df):
    # Filter rows where 'gaze on face' is True
    gaze_true_data = df[df['gaze on face'] == True]

    # Count occurrences of each category in 'landmark' column
    landmark_counts = gaze_true_data['landmark'].explode().value_counts()
    # Initialize an empty dictionary to store individual landmarks and their counts
    individual_landmarks_counts = {}

    # Iterate over the landmark counts
    for index, count in landmark_counts.items():
        # Convert the string representation of a list to a list
        landmarks = index.strip('[]').split(', ')
        for landmark in landmarks:
            landmark = landmark.strip("'")  # Remove single quotes from the landmark string
            individual_landmarks_counts[landmark] = individual_landmarks_counts.get(landmark, 0) + count

    # Convert the dictionary to a DataFrame
    individual_landmarks_df = pd.DataFrame(individual_landmarks_counts.items(), columns=['landmark', 'count'])

    # Calculate percentage of rows for each individual landmark
    individual_landmarks_df['percentage'] = (individual_landmarks_df['count'] / individual_landmarks_df['count'].sum()) * 100
    return individual_landmarks_df

def plot_percentages(df_perc, out_path):
    # Plotting
    # Define custom colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Plotting
    plt.figure(figsize=(12, 10))
    plt.bar(df_perc['landmark'], df_perc['percentage'], color=colors) 
    
    # Adjusting font size and style
    plt.title('Percentage of Gaze Mapped on Different Landmarks', fontsize=30, fontweight='bold')
    plt.xlabel('Landmark', fontsize=28)
    plt.ylabel('%-Gaze Mapped \n(out of all data detected on face)', fontsize=28)
    plt.xticks(rotation=45, fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    figure_path = os.path.join(out_path, "_barplot.png")
    plt.savefig(figure_path, bbox_inches='tight', pad_inches=1.0)
    logging.info(f"Barplot saved at: {figure_path}")

    # Plotting Pie Chart
    plt.figure(figsize=(12, 10))
    plt.pie(df_perc['percentage'], labels=df_perc['landmark'], colors=colors, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 25}) # Adjust fontsize here
    plt.title('Percentage of Gaze Mapped on Different Landmarks', fontsize=30, fontweight='bold')
    pie_path = os.path.join(out_path, "_pie.png")
    # Save the plot to out_path
    plt.savefig(pie_path,bbox_inches='tight', pad_inches=1.0)
    logging.info(f"Pie chart saved at: {pie_path}")