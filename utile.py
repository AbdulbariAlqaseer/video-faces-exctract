from typing import Tuple, Union
from config import MARGIN, ROW_SIZE, TEXT_COLOR, FONT_SIZE, FONT_THICKNESS
import math
import cv2

import numpy as np
import pandas as pd

def detection_info(detection_result, width=None, height=None) -> tuple:
    """extract info from detection result.
    Args:
        detection_result: The list of all "Detection" entities to be visualize.
        width: width of image.
        height: height of image.
    Returns:
        tuple.
    """
    for detection in detection_result.detections:
        bbox = detection.bounding_box   
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height

        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name

        probability = round(category.score, 2)
        
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)

        if width and height:
            keypoints_px = [_normalized_to_pixel_coordinates(keypoint.x, keypoint.y, width, height) for keypoint in detection.keypoints]
            yield (start_point, end_point, category_name, probability, result_text, text_location, keypoints_px)
        else:
            yield (start_point, end_point, category_name, probability, result_text, text_location)

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def visualize(
    image,
    detection_result
) -> np.ndarray:
  """Draws bounding boxes and keypoints on the input image and return it.
  Args:
    image: The input RGB image.
    detection_result: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  annotated_image = image.copy()
  height, width, _ = image.shape

  for (start_point, end_point, _, _, result_text, text_location, keypoints_px) in\
    detection_info(detection_result, width, height):
    # print(f"{start_point = }, {end_point = }", end="\t")
    # Draw bounding_box
    cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)
    
    # Draw keypoints
    for keypoint_px in keypoints_px:
      color, thickness, radius = (0, 255, 0), 2, 2
      cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

    # Draw label and score
    cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

  return annotated_image

def save_data(save_path, data, columns):
    # save data about every face in dataframe
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(save_path, index=False)

def index_to_time(index, fps):
   return index / fps

