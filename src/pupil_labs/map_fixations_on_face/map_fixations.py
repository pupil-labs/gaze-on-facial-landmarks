import warnings
import logging
import os
import math
import glob

import av
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import pupil_labs.map_fixations_on_face.map_fixations as map_fixations
# from fractions import Fraction
# from pathlib import Path
# from pupil_labs.dynamic_content_on_rim.uitools.ui_tools import get_savedir
# from pupil_labs.dynamic_content_on_rim.video.read import get_frame, read_video_ts

# from rich.logging import RichHandler
# from rich.progress import Progress
warnings.filterwarnings("ignore")

## Map fixations on facial landmarks
def map_fixations_on_face(fixations_merged_csv, face_csv, aoi_radius, ellipse_size):
    mapped_fixations_on_face = []
    landmarks = ['eye left', 'eye right', 'nose', 'mouth left', 'mouth right']

    for _, fixation in fixations_merged_csv.iterrows():
        fixation_timestamp = fixation['start timestamp [ns]']
        closest_timestamp = face_csv['timestamp [ns]'].sub(fixation_timestamp).abs().idxmin()

        if not fixation['fixation on face']:
            mapped_fixations_on_face.append({
                'fixation_id': fixation['fixation id'],
                'fixation_timestamp': fixation_timestamp,
                'landmark': 'Not on face',
                'fixation coordinates': (fixation['fixation x [px]'], fixation['fixation y [px]']),
                'landmark coordinates': None
            })
            continue

        elif fixation['fixation on face']:
            mapped = False
            for landmark in landmarks:
                if landmark in ['eye left', 'eye right', 'nose']:
                    landmark_x = face_csv.loc[closest_timestamp, f'{landmark} x [px]']
                    landmark_y = face_csv.loc[closest_timestamp, f'{landmark} y [px]']

                    landmark_aoi = {
                        'x_min': landmark_x - aoi_radius,
                        'x_max': landmark_x + aoi_radius,
                        'y_min': landmark_y - aoi_radius,
                        'y_max': landmark_y + aoi_radius
                    }

                elif landmark in ['mouth left', 'mouth right']:
                    # Create ellipse around "mouth left" and "mouth right"
                    mouth_left = (face_csv.loc[closest_timestamp, 'mouth left x [px]'],
                                  face_csv.loc[closest_timestamp, 'mouth left y [px]'])
                    mouth_right = (face_csv.loc[closest_timestamp, 'mouth right x [px]'],
                                   face_csv.loc[closest_timestamp, 'mouth right y [px]'])

                    # Calculate ellipse parameters
                    center_x = int((mouth_left[0] + mouth_right[0]) / 2)
                    center_y = int((mouth_left[1] + mouth_right[1]) / 2)
                    major_axis = int(abs(mouth_right[0] - mouth_left[0]) / 2)
                    minor_axis = int(abs(mouth_right[1] - mouth_left[1]) / 2)

                    # Adjust the major and minor axes based on the ellipse_size
                    major_axis += ellipse_size
                    minor_axis += ellipse_size

                    landmark_aoi = {
                        'x_min': center_x - major_axis,
                        'x_max': center_x + major_axis,
                        'y_min': center_y - minor_axis,
                        'y_max': center_y + minor_axis
                    }

                x_fit = landmark_aoi['x_min'] <= fixation['fixation x [px]'] <= landmark_aoi['x_max']
                y_fit = landmark_aoi['y_min'] <= fixation['fixation y [px]'] <= landmark_aoi['y_max']

                if x_fit and y_fit:
                    mapped_fixations_on_face.append({
                        'fixation_id': fixation['fixation id'],
                        'fixation_timestamp': fixation_timestamp,
                        'landmark': 'mouth' if 'mouth' in landmark else landmark,
                        'fixation coordinates': (fixation['fixation x [px]'], fixation['fixation y [px]']),
                        'landmark coordinates': (landmark_aoi['x_min'], landmark_aoi['y_min'])
                    })
                    mapped = True
                    break

            if not mapped:
                mapped_fixations_on_face.append({
                    'fixation_id': fixation['fixation id'],
                    'fixation_timestamp': fixation_timestamp,
                    'landmark': 'Not mapped',
                    'fixation coordinates': (fixation['fixation x [px]'], fixation['fixation y [px]']),
                    'landmark coordinates': None
                })
    logging.info("Fixations mapped!")
    mapped_fixations_on_face = pd.DataFrame(mapped_fixations_on_face)
    return mapped_fixations_on_face