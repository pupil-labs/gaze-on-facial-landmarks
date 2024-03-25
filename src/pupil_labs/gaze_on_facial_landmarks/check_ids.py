""""
Adapted from the check_ids function from Dynamic RIM module
"""
import logging
import numpy as np
import pandas as pd
from rich.table import Table
from typing import Optional
from rich import box
from rich.console import Console

def rich_df(
    pandas_dataframe: pd.DataFrame,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Based on https://gist.github.com/neelabalan/33ab34cf65b43e305c3f12ec6db05938"""
    rich_table = Table(show_header=True, header_style="bold magenta")

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD
    console = Console(width=150)
    console.print(rich_table)
    return rich_table

def check_ids(gaze_df, world_timestamps_df, gaze_on_face_df):
    """
    Checks if the recording IDs of the gaze data and the world timestamps are the same.
    :param gaze_df: The gaze data.
    :param world_timestamps_df: The world timestamps.
    :param gaze_on_face_df: The gaze data from the face mapper.
    returns gaze_on_face_df with only the matching ID.
    """
    g_ids = gaze_df["recording id"].unique()
    w_ids = world_timestamps_df["recording id"].unique()
    rim_ids = gaze_on_face_df["recording id"].unique()
    if g_ids.shape[0] != 1 or w_ids.shape[0] != 1:
        error_base = "None or more than one recording ID found "
        if g_ids.shape[0] != 1:
            error_end = "in gaze data: {g_ids}"
        elif w_ids.shape[0] != 1:
            error_end = "in world timestamps: {w_ids}"
        logging.error(error_base + error_end)
        raise SystemExit(error_base + error_end)
    if not np.isin(rim_ids, g_ids).any():
        error = (
            "Recording ID of Face Mapper gaze data does not match recording ID"
            " of the Raw data, please check if you selected the"
            " right folder."
        )
        logging.error(error)
        raise SystemExit(error)
    else:
        ID = g_ids[0]
        logging.info(
            f"""Recording ID of Face Mapper gaze data matches recording ID of the RAW data
            id: {ID} """
        )
        isID = gaze_on_face_df["recording id"] == ID
        gaze_on_face_df.drop(gaze_on_face_df.loc[np.invert(isID)].index, inplace=True)

        if gaze_on_face_df.empty:
            error = f"No valid gaze data in Face Mapper gaze data for recording ID {ID}"
            logging.error(error)
            raise SystemExit(error)
        rich_df(gaze_on_face_df.head())
    return gaze_on_face_df