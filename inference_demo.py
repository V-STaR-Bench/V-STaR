import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
import os
import requests
from tqdm import tqdm
import json
import re
import torch
import math

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "Qwen/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

def inference(video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "video", "video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels, "fps": 1.0},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    # print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0], resized_height, resized_width

def read_anno(anno_file):
    with open(anno_file, 'r') as f:
        data = json.load(f)
    return data

def find_video(video_folder, vid):
    """
    Finds the vid.mp4 file in the video_folder and its subfolders.

    Args.
        video_folder (str): path of the folder to search.
        vid (str): the filename of the video (without extension).

    Returns.
        str: absolute path to the vid.mp4 file, or None if not found.
    """
    target_filename = f"{vid}.mp4"
    for root, _, files in os.walk(video_folder):
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None

def get_answer_vqa(data, video_path):
    prompt = f"Answer the question about the video: {data['question']} \n (If the answer is a person, you don't need to identify the person.)"
    answer_vqa, _, _ = inference(video_path, prompt)
    return answer_vqa

def get_answer_temporal(data, video_path):
    video_length = round(data['frame_count']/data['fps'], 2)
    temporal_question = data['temporal_question']
    prompt = f"This video is {video_length} seconds long. Answer the question about the video: {temporal_question} \n Output the start and end moment timestamps.",
    answer_temporal, _, _ = inference(video_path, prompt)
    return answer_temporal

def get_answer_temporal_2(data, video_path, bboxes):
    video_length = round(data['frame_count']/data['fps'], 2)
    temporal_question = data['temporal_question']
    w, h = data['width'], data['height']
    prompt = f"This video is {video_length} seconds long with a resolution of {w}x{h} (width x height). Answer the question about the video: {temporal_question} \n There are {len(bboxes)} bounding boxes of the key object related to the question in the video without knowing the time, which are:{bboxes}. Output the start and end moment timestamps.",
    answer_temporal, _, _ = inference(video_path, prompt)
    return answer_temporal

def get_answer_spatial(data, video_path):
    video_length = round(data['frame_count']/data['fps'], 2)
    st, et = math.ceil(data['timestamps'][0]), math.floor(data['timestamps'][1])
    time_range = list(range(st, et + 1))
    w, h = data['width'], data['height']
    spatial_question = data['spatial_question']
    prompt = f"""Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format. \n
                    For each whole second within the time range {time_range} provided (inclusive of the boundaries), output a series of bounding boxes of the object in JSON format. The keys should be the whole seconds (as strings), and the values should be the box in [x1, y1, x2, y2] format.
                    Example output: {{"{time_range[0]}": [x1, y1, x2, y2],...}}
                    """
    answer_spatial, input_height, input_width = inference(video_path, prompt)
    return answer_spatial, input_height, input_width

def get_answer_spatial_2(data, video_path, bboxes):
    video_length = round(data['frame_count']/data['fps'], 2)
    st, et = math.ceil(data['timestamps'][0]), math.floor(data['timestamps'][1])
    time_range = list(range(st, et + 1))
    w, h = data['width'], data['height']
    spatial_question = data['spatial_question_2']
    prompt = f"""Please answer the question about the video: {spatial_question} with a series of bounding boxes in [x1, y1, x2, y2] format. \n
                    For each whole second that may related to the question, output a series of bounding boxes of the object in JSON format. You only need to output {len(bboxes)} bbox(es). You need to determine which frame is related to the question, and you don't need to output the bbox for the frames not related to the question.
                    The keys should be the whole seconds (as strings), and the values should be the bounding box in [x0,y0,x1,y1] format. 
                    \n Example output:
                    {{"0": [x0,y0,x1,y1], "1":..., ..., "{len(bboxes)}":...}} (if the frames at 0~{len(bboxes)} second are related to the questions)
                    """
    answer_spatial, input_height, input_width = inference(video_path, prompt)
    return answer_spatial, input_height, input_width

def extract_timestamps(result):
    """extract timestamps from the answer"""
    match = re.findall(r"\b\d+(?:\.\d+)?\b", result)
    return [float(match[0]), float(match[1])] if len(match) == 2 else []


def fix_incomplete_json(json_str):
    """
    fix the incomplete brackets of the json
    """
    # Counting left and right brackets
    open_square = json_str.count('[')
    close_square = json_str.count(']')
    open_curly = json_str.count('{')
    close_curly = json_str.count('}')

    # Complete the square brackets
    if open_square > close_square:
        json_str += ']' * (open_square - close_square)
    elif close_square > open_square:
        json_str = '[' * (close_square - open_square) + json_str

    # Complete the curly brackets
    if open_curly > close_curly:
        json_str += '}' * (open_curly - close_curly)
    elif close_curly > open_curly:
        json_str = '{' * (close_curly - open_curly) + json_str

    return json_str


def extract_bounding_boxes(answer_spatial, data, input_height, input_width):
    """
    Extract bounding boxes from the input answer_spatial and denormalize the coordinates using the width and height from the data.
    """
    w, h = data['width'], data['height']  # 提取宽度和高度

    def denormalize_bbox(bbox):
        """
        denormalize the coordinates of bbox
        """
        try:
            if len(bbox) == 1:
                bbox = bbox[0]
            if len(bbox) == 2:
                bbox = bbox[1]
            x_min = int(bbox[0] / input_width * w)
            y_min = int(bbox[1] / input_height * h)
            x_max = int(bbox[2] / input_width * w)
            y_max = int(bbox[3] / input_height * h)
            return [x_min, y_min, x_max, y_max]
        except Exception as e:
            print(f"Processing {bbox} occurs Error {e}")
            return bbox

    # match markdown json
    markdown_pattern = r'```json\s*\n(\[.*?\]|\{.*?\})\s*\n```'
    match = re.search(markdown_pattern, answer_spatial, re.DOTALL)
    if not match:
        # If there is no Markdown wrapper, then try to match the JSON format directly
        json_pattern = r'(\[[\s\S]*\]|\{[\s\S]*\})'
        match = re.search(json_pattern, answer_spatial, re.DOTALL)
    if match:
        # match bbox in JSON
        bounding_boxes_str = match.group(1).strip()
        # Replace single quotes with double quotes to conform to the JSON specification
        bounding_boxes_str = bounding_boxes_str.replace("'", '"')
        try:
            # Convert strings to dictionary or list format
            bounding_boxes = json.loads(bounding_boxes_str)
            # If it's a list and contains a dictionary inside, expand it to a single dictionary
            if isinstance(bounding_boxes, list) and all(isinstance(item, dict) for item in bounding_boxes):
                combined_dict = {}
                for item in bounding_boxes:
                    combined_dict.update(item)
                bounding_boxes = combined_dict
                # Determine if the extracted JSON is a dictionary or a list.
            if isinstance(bounding_boxes, list):
                # bounding boxes in list
                return {str(box[0]): box[1] for box in bounding_boxes}
            elif isinstance(bounding_boxes, dict):
                # bounding boxes in dictionary
                return {key: value for key, value in bounding_boxes.items()}
        except Exception as e:
            # if failed, try to fix it.
            fixed_bounding_boxes_str = fix_incomplete_json(bounding_boxes_str)
            try:
                bounding_boxes = json.loads(fixed_bounding_boxes_str)
                if isinstance(bounding_boxes, list):
                    return [box for box in bounding_boxes]
                elif isinstance(bounding_boxes, dict):
                    return {key: value for key, value in bounding_boxes.items()}
            except Exception as e:
                print(f"Failed after fixing: {e}\nExtracted JSON: {fixed_bounding_boxes_str}")
                return None
    else:
        print("No match found for the bounding box JSON.")
        return None

def test_qwen2_5vl(video_folder, anno_file, result_file):
    anno = read_anno(anno_file)

    for idx, data in enumerate(tqdm(anno, desc="Processing videos", unit="video")):
        try:
            vid = data['vid']
            timestamps = data['timestamps']
            video_length = round(data['frame_count']/data['fps'], 1)
            boxes = [[box_data["xmin"], box_data["ymin"], box_data["xmax"], box_data["ymax"]] \
                        for box in data["bboxes"] for box_data in box.values()]
            video_path = find_video(video_folder, vid)
            answer_vqa = get_answer_vqa(data, video_path)
            # chain one
            answer_temporal = get_answer_temporal(data, video_path)
            answer_temporal_post = extract_timestamps(answer_temporal)

            answer_spatial, input_height, input_width = get_answer_spatial(data, video_path)
            answer_spatial_post = extract_bounding_boxes(answer_spatial, data, input_height, input_width)
            
            # chain two
            answer_spatial_2, input_height, input_width = get_answer_spatial_2(data, video_path, boxes)
            answer_spatial_post_2 = extract_bounding_boxes(answer_spatial_2, data, input_height, input_width)

            answer_temporal_2 = get_answer_temporal_2(data, video_path, boxes)
            answer_temporal_post_2 = extract_timestamps(answer_temporal_2)
            
            # update data
            data['answer_vqa'] = answer_vqa
            data['answer_temporal_pre'] = answer_temporal
            data['answer_temporal'] = answer_temporal_post
            data['answer_spatial_pre'] = answer_spatial
            data['answer_spatial'] = answer_spatial_post

            data['answer_spatial_pre_2'] = answer_spatial_2
            data['answer_spatial_2'] = answer_spatial_post_2
            data['answer_temporal_pre_2'] = answer_temporal_2
            data['answer_temporal_2'] = answer_temporal_post_2
            # update result_file
            with open(result_file, 'w') as f:
                json.dump(anno, f, indent=4)
        except Exception as e:
            print("")
            print(f"ERROR in data {idx}: {e}.")
            continue


if __name__ == "__main__":
    video_folder = "/Path/to/video/folder"
    anno_file = "/path/to/anno/file.json"
    result_file = "/path/to/result/file.json"
    test_qwen2_5vl(video_folder, anno_file, result_file)