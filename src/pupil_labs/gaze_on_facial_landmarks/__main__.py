import warnings
import logging
import os
import glob

import av
import cv2
import numpy as np
import pandas as pd
import tkinter as tk
import pupil_labs.gaze_on_facial_landmarks.map_on_landmarks as map_on_landmarks

from rich.logging import RichHandler
from tkinter import filedialog
from fractions import Fraction
from pathlib import Path
from pupil_labs.dynamic_content_on_rim.uitools.ui_tools import get_savedir
from pupil_labs.dynamic_content_on_rim.video.read import get_frame, read_video_ts
from rich.progress import Progress

warnings.filterwarnings("ignore")


def run_all(args_input):
    face_folder = Path(args_input.get("face_mapper_output_folder"))
    raw_data_folder = Path(args_input.get("raw_data_output_folder"))
    start = args_input.get("start", "")
    end = args_input.get("end", "")
    aoi_circle = args_input.get("aoi_radius")
    ellipse = args_input.get("ellipse_size")
    gaze_circle_size = args_input.get("gaze_circle_size")

    logging.getLogger(__name__)
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    logging.getLogger("libav.swscaler").setLevel(logging.ERROR)

    # Output folder should be the face mapper enrichment folder
    output_path = face_folder
    logging.info(f"Gaze circle size: {gaze_circle_size}")
    # Get the subfolder within the first level
    subfolders = [folder for folder in raw_data_folder.iterdir() if folder.is_dir()]
    if subfolders:
        selected_second_level = subfolders[0].name
        raw_data_folder = raw_data_folder / selected_second_level
    else:
        logging.info("No subfolders found")

    logging.info(
        "[white bold on #0d122a]◎ Mapping gaze on facial landmarks by Pupil Labs[/]",
        extra={"markup": True},
    )

    # Format to read timestamps
    oftype = {"timestamp [ns]": np.uint64}

    # Read the timestamps
    world_timestamps_df = pd.read_csv(
        Path(raw_data_folder, "world_timestamps.csv"), dtype=oftype
    )
    face_df = pd.read_csv(Path(face_folder, "face_detections.csv"), dtype=oftype)

    events_df = pd.read_csv(Path(raw_data_folder, "events.csv"), dtype=oftype)

    gaze_df = pd.read_csv(Path(raw_data_folder, "gaze.csv"), dtype=oftype)
    gaze_on_face = pd.read_csv(Path(face_folder, "gaze_on_face.csv"), dtype=oftype)
    selected_col = ["timestamp [ns]", "gaze on face"]
    gaze_face = gaze_on_face[selected_col]
    gaze_all = pd.merge_asof(
        gaze_df,
        gaze_face,
        on="timestamp [ns]",
        direction="nearest",
    )

    files = glob.glob(str(Path(raw_data_folder, "*.mp4")))
    if len(files) != 1:
        error = "There should be only one video in the raw folder!"
        raise Exception(error)
    video_path = files[0]
    print(video_path)

    # Read the video
    logging.info(
        "[white bold on #0d122a]Reading video...[/]",
        extra={"markup": True},
    )
    # Read the video
    logging.info(
        "[white bold on #0d122a]Reading video...[/]",
        extra={"markup": True},
    )
    _, frames, pts, ts = read_video_ts(video_path)
    logging.info(
        "[white bold on #0d122a]Reading audio...[/]",
        extra={"markup": True},
    )

    ts = world_timestamps_df["timestamp [ns]"]

    video_df = pd.DataFrame(
        {
            "frames": np.arange(frames),
            "pts": [int(pt) for pt in pts],
            "timestamp [ns]": ts,
        }
    )

    logging.info("Merging dataframes")
    face_df = face_df.sort_values(by="timestamp [ns]")
    gaze_all = gaze_all.sort_values(by="timestamp [ns]")

    video_face = pd.merge_asof(
        video_df,
        face_df,
        on="timestamp [ns]",
        direction="nearest",
    )

    merged_video = pd.merge_asof(
        video_face,
        gaze_all,
        on="timestamp [ns]",
        direction="nearest",
    )

    if start != "recording.begin":
        logging.info(f"Looking for start event: {start}")
        if not events_df["name"].isin([start]).any():
            raise Exception("Start event not found!")
        else:
            start = events_df[events_df["name"] == start]["timestamp [ns]"].values[0]
            merged_video = merged_video[merged_video["timestamp [ns]"] >= start]

    if end != "recording.end":
        logging.info(f"Looking for end event: {end}")
        if not events_df["name"].isin([end]).any():
            raise Exception("End event not found!")
        else:
            end = events_df[events_df["name"] == end]["timestamp [ns]"].values[0]
            merged_video = merged_video[merged_video["timestamp [ns]"] <= end]
            logging.info(f"Ending at {end}")

    # Read first frame
    with av.open(video_path) as vid_container:
        logging.info("Reading first frame")
        vid_frame = next(vid_container.decode(video=0))

    num_processed_frames = 0

    # Get the output path
    if output_path is None:
        output_file = get_savedir(None, type="video")
        out_csv = output_file.replace(os.path.split(output_file)[1], "merged_data.csv")
        output_path = os.path.split(output_file)[0]
    else:
        output_file = os.path.join(output_path, "gaze-on-face.mp4")
        out_csv = os.path.join(output_path, "merged_data.csv")
    logging.info(f"Output path: {output_file}")

    # Here we go!
    with av.open(video_path) as video:
        logging.info("Ready to process video")
        # Prepare the output video
        with av.open(output_file, "w") as out_container:
            out_video = out_container.add_stream(
                "libx264", rate=30, options={"crf": "18"}
            )
            out_video.width = video.streams.video[0].width
            out_video.height = video.streams.video[0].height
            out_video.pix_fmt = "yuv420p"
            out_video.codec_context.time_base = Fraction(1, 30)

            lpts = -1
            merged_video["landmark"] = None
            # For every frame in the video
            with Progress() as progress_bar:
                video_task = progress_bar.add_task(
                    "📹 Processing video", total=merged_video.shape[0]
                )
                while num_processed_frames < merged_video.shape[0]:
                    row = merged_video.iloc[num_processed_frames]

                    # Get the frame
                    vid_frame, lpts = get_frame(video, int(row["pts"]), lpts, vid_frame)
                    if vid_frame is None:
                        break

                    # Convert to ndarray
                    img_original = vid_frame.to_ndarray(format="rgb24")
                    frame = cv2.cvtColor(img_original, cv2.COLOR_RGB2BGR)
                    frame = np.asarray(frame, dtype=np.float32)
                    frame = frame[:, :, :]

                    # Map gaze on facial landmark and draw AOIs
                    (
                        landmark_list,
                        mapped_gaze,
                        gaze_coor,
                    ) = map_on_landmarks.map_and_draw(
                        frame, row, aoi_circle, ellipse, gaze_circle_size
                    )

                    merged_video.at[num_processed_frames, "landmark"] = landmark_list
                    xy = np.array(gaze_coor, dtype=np.int32)

                    # Add the closest annotation
                    current_ts = row["timestamp [ns]"]

                    # Find the closest event name using the timestamp
                    closest_event = events_df.iloc[
                        (events_df["timestamp [ns]"] - current_ts).abs().argsort()[:1]
                    ]
                    merged_video.loc[
                        num_processed_frames, "closest_annotation"
                    ] = closest_event["name"].values[0]

                    # make a aoi_circle on the gaze
                    if not np.isnan(xy).any():
                        cv2.circle(frame, xy, gaze_circle_size, (255, 0, 0), 10)

                        text_landmark = f"{landmark_list}"
                        text_location = (frame.shape[1] - 500, 50)
                        text_box_size = (300, 50)

                        # Create a black box, put text inside
                        cv2.rectangle(
                            frame,
                            text_location,
                            (
                                text_location[0] + text_box_size[0],
                                text_location[1] + text_box_size[1],
                            ),
                            (0, 0, 0),
                            -1,
                        )
                        cv2.putText(
                            frame,
                            text_landmark,
                            (text_location[0] + 10, text_location[1] + 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    # Finally get the frame ready.
                    out_ = cv2.normalize(
                        frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )

                    cv2.cvtColor(out_, cv2.COLOR_BGR2RGB, out_)
                    np.expand_dims(out_, axis=2)
                    out_frame = av.VideoFrame.from_ndarray(out_, format="rgb24")
                    for packet in out_video.encode(out_frame):
                        out_container.mux(packet)
                    if num_processed_frames % 100 == 0:
                        logging.info(
                            f"Processed {num_processed_frames} frames out of {merged_video.shape[0]}"
                        )
                    progress_bar.advance(video_task)
                    progress_bar.refresh()
                    num_processed_frames += 1

                out_container.close()

            # save the csvs
            selected_columns = ["frames", "pts","timestamp [ns]", "p1 x [px]", "p1 y [px]","p2 x [px]",	"p2 y [px]","eye left x [px]", "eye left y [px]","eye right x [px]",	"eye right y [px]",	"nose x [px]",
                                "nose y [px]",	"mouth left x [px]",	"mouth left y [px]",	"mouth right x [px]",	"mouth right y [px]",	"section id_y",	"recording id_y", "gaze x [px]", "gaze y [px]",	"gaze on face", "landmark"]
            merged_selected = merged_video[selected_columns]
            merged_selected.to_csv(out_csv, index=False)
            logging.info(f"CSV files saved at: {out_csv}")

            logging.info(
                "[white bold on #0d122a]◎ Mapping and rendering completed! ⚡️[/]",
                extra={"markup": True},
            )


def browse_directory(entry_var):
    directory = filedialog.askdirectory()
    entry_var.set(directory)


def run_main():
    root = tk.Tk()
    root.title("Map gaze on facial landmarks")

    # Entry for Face Mapper Directory
    tk.Label(root, text="Face Mapper directory").pack()
    face_mapper_output_folder = tk.StringVar()
    tk.Entry(root, textvariable=face_mapper_output_folder).pack()
    tk.Button(
        root, text="Browse", command=lambda: browse_directory(face_mapper_output_folder)
    ).pack()

    # Entry for Raw Data Directory
    tk.Label(root, text="Raw data directory").pack()
    raw_data_output_folder = tk.StringVar()
    tk.Entry(root, textvariable=raw_data_output_folder).pack()
    tk.Button(
        root, text="Browse", command=lambda: browse_directory(raw_data_output_folder)
    ).pack()

    # Entry for AOI Radius
    tk.Label(root, text="AOI radius").pack()
    aoi_radius_var = tk.IntVar()
    aoi_radius_var.set(30)
    tk.Entry(root, textvariable=aoi_radius_var).pack()

    # Entry for Ellipse
    tk.Label(root, text="Ellipse size").pack()
    ellipse_size_var = tk.IntVar()
    ellipse_size_var.set(30)
    tk.Entry(root, textvariable=ellipse_size_var).pack()

    # Entry for gaze circle size
    tk.Label(root, text="Gaze circle size").pack()
    gaze_size_var = tk.IntVar()
    gaze_size_var.set(20)
    tk.Entry(root, textvariable=gaze_size_var).pack()

    # Entry for start event
    tk.Label(root, text="Start event").pack()
    start_var = tk.StringVar()
    start_var.set("recording.begin")
    tk.Entry(root, textvariable=start_var).pack()

    # Entry for end event
    tk.Label(root, text="End event").pack()
    end_var = tk.StringVar()
    end_var.set("recording.end")
    tk.Entry(root, textvariable=end_var).pack()

    # Button to run the pipeline
    tk.Button(
        root,
        text="Let's start",
        command=lambda: run_analysis_callback(
            root,
            {
                "face_mapper_output_folder": face_mapper_output_folder.get(),
                "raw_data_output_folder": raw_data_output_folder.get(),
                "aoi_radius": aoi_radius_var.get(),
                "ellipse_size": ellipse_size_var.get(),
                "gaze_circle_size": gaze_size_var.get(),
                "start": start_var.get(),
                "end": end_var.get(),
            },
        ),
    ).pack()

    root.mainloop()


def run_analysis_callback(root, input_data):
    run_all(input_data)
    root.destroy()  # Close the main window


if __name__ == "__main__":
    run_main()
