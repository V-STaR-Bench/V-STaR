import json
import numpy as np
from tqdm import tqdm
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
import ast

model_name = "Qwen/Qwen2.5-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


system_prompt = """
As an AI assistant, your task is to evaluate a candidate answer in comparison to a given correct answer.
The question itself, the correct 'groundtruth' answer, and the candidate answer will be provided to you.
Your assessment should range from 0 to 3, \
based solely on the semantic similarity between the groundtruth and the candidate answer, \
disregarding any grammatical differences.
A rating of 0 suggests no similarity, implying the candidate answer is entirely incorrect.
A rating of 1 suggests low similarity, meaning the candidate answer is largely incorrect.
A rating of 2 suggests high similarity, meaning the candidate answer is largely correct.
Lastly, a rating of 3 indicates complete similarity, which means the candidate answer is entirely correct.
Your response should be a single integer from 0, 1, 2, or 3.
"""

tmpl = 'Question: {}\nGroundtruth answer: {}\nCandidate answer: {}\nYour response: '

def qwen2_5_evaluation(question, gt, candidate):
    user_prompt=tmpl.format(question, gt, candidate)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    score = response
    try:
        score = int(score)
    except (ValueError, TypeError):
        score = -1
    return score


def calculate_temporal_iou(gt_range, pred_range):
    """compute Temporal IoU"""
    if not pred_range:  # If pred_range is None or empty
        return 0.0  # Return default 0.0
    
    # If pred_range is a string, then try to convert to a list
    if isinstance(pred_range, str):
        try:
            pred_range = ast.literal_eval(pred_range)
        except (ValueError, SyntaxError):
            return 0.0  #  The conversion fails and returns the default value of 0.0
    
    # Ensure that pred_range is a list or tuple of two values
    if not isinstance(pred_range, (list, tuple)) or len(pred_range) != 2 or \
    not all(isinstance(x, (int, float)) for x in pred_range):
        return 0.0  # is not a valid values, returns the default value of 0.0

    gt_start, gt_end = gt_range
    pred_start, pred_end = pred_range
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = max(gt_end, pred_end) - min(gt_start, pred_start)
    return intersection / union if union > 0 else 0.0


def compute_iou(gt_bbox, pred_bbox):
        """Calculate the IoU for two boxes"""
        if not isinstance(pred_bbox, (list, tuple)) or len(pred_bbox) != 4:
            return 0.0
        
        # get GT bbox coordinates
        gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_bbox['xmin'], gt_bbox['ymin'], gt_bbox['xmax'], gt_bbox['ymax']
        pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_bbox

        # calculate intersection
        x1 = max(gt_xmin, pred_xmin)
        y1 = max(gt_ymin, pred_ymin)
        x2 = min(gt_xmax, pred_xmax)
        y2 = min(gt_ymax, pred_ymax)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # calculate union
        gt_area = (gt_xmax - gt_xmin) * (gt_ymax - gt_ymin)
        pred_area = (pred_xmax - pred_xmin) * (pred_ymax - pred_ymin)
        union = gt_area + pred_area - intersection

        return intersection / union if union > 0 else 0.0

def calculate_bbox_iou(gt_bbox, pred_bboxes):
    """Calculate single BBox IoU, support multiple prediction boxes to take max IoU"""
    try:
        if not pred_bboxes:
            return 0.0
        
        # Handling of individual boxes
        if isinstance(pred_bboxes[0], (int, float)) and len(pred_bboxes) == 4:
            pred_bboxes = [pred_bboxes]
        
        # Calculate the IoU for all prediction frames and return the maximum value
        return max([compute_iou(gt_bbox, pred_bbox) for pred_bbox in pred_bboxes])
    except:
        return 0.0

def calculate_spatial_metrics(gt_bboxes, pred_bboxes):
    """Compute vIoU and AP"""
    if not pred_bboxes:  # Checks if pred_bboxes are None or empty.
        return [0.0] * 5, 0.0  # Return default: 0 for all APs, 0 for m_vIoU

    iou_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    ious = []
    aps = []
    for gt_bbox_entry in gt_bboxes:
        for frame_id, gt_bbox in gt_bbox_entry.items():
            frame_id = frame_id.split("_")[0]
            if frame_id in pred_bboxes:
                pred_bbox = pred_bboxes[frame_id]
                iou = calculate_bbox_iou(gt_bbox, pred_bbox)
                ious.append(iou)
            else:
                ious.append(0.0)
    mIoU = np.mean(ious) if ious else 0.0

    for threshold in iou_thresholds:
        scores = [1 if iou >= threshold else 0 for iou in ious]
        if len(ious) > 0:
            aps.append(np.mean(scores))
        else:
            aps.append(0.0)
    return aps, mIoU

def evaluate_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    model_name = file_path.split("/")[-1].split("_")[0]
    domains = {}
    durations = {}
    overall_stats = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}


    for idx, item in enumerate(tqdm(data, desc=f"Evaluating {model_name} results", unit="item")):
        video_length = round(item['frame_count']/item['fps'], 2)
        w, h = item['width'], item['height']
        domain = item.get("domain", "unknown")
        if domain not in domains:
            domains[domain] = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}
        
        if video_length <= 60:
            duration = "Short"
        elif 60 < video_length <= 180:
            duration = "Medium"
        else:
            duration = "Long"
        if duration not in durations:
            durations[duration] = {"all_rating":[], "valid_rating": [], "correct_num":0, "temporal_ious": [], "temporal_ious_2": [], "spatial_aps": [[] for _ in range(5)],
                    "spatial_aps_2": [[] for _ in range(5)], "spatial_mious": [], "spatial_mious_2": [],
                    "vqa_temporal_idx":[], "vqa_spatial_idx":[], "temporal_spatial_idx":[],"vqa_temp_spatial_idx":[],
                    "vqa_temporal_idx_2":[], "vqa_spatial_idx_2":[], "temporal_spatial_idx_2":[],"vqa_temp_spatial_idx_2":[]}

        if 'answer_vqa' in item and item['answer_vqa']:
            score = qwen2_5_evaluation(item['question'], item['refine_answer'], item['answer_vqa'])
            # score1 = qwen2_5_evaluation(item['question'], item['refine_answer'], item['answer_vqa'])
            # score2 = qwen2_5_evaluation(item['question'], item['object'], item['answer_vqa'])
            # score = max(score1,score2)
        else:
            continue
        overall_stats["all_rating"].append(score if score != -1 else 0)
        domains[domain]["all_rating"].append(score if score != -1 else 0)
        durations[duration]["all_rating"].append(score if score != -1 else 0)
        if score != -1:
            overall_stats["valid_rating"].append(score)
            domains[domain]["valid_rating"].append(score)
            durations[duration]["valid_rating"].append(score)
        if score >= 2:
            overall_stats["correct_num"] += 1
            domains[domain]["correct_num"] += 1
            durations[duration]["correct_num"] += 1
        data[idx]["VQA_score"] = score
        # compute answer_temporal
        if 'answer_temporal' in item and item['answer_temporal']:
            temporal_iou = calculate_temporal_iou(item['temporal_gt_sec'], item['answer_temporal'])
        else:
            temporal_iou = 0.0

        overall_stats["temporal_ious"].append(temporal_iou)
        domains[domain]["temporal_ious"].append(temporal_iou)
        durations[duration]["temporal_ious"].append(temporal_iou)
        data[idx]["temporal_IoU"] = temporal_iou

        # compute answer_temporal_2
        if 'answer_temporal_2' in item and item['answer_temporal_2']:
            temporal_iou_2 = calculate_temporal_iou(item['temporal_gt_sec'], item['answer_temporal_2'])
        else:
            temporal_iou_2 = 0.0
        
        overall_stats["temporal_ious_2"].append(temporal_iou_2)        
        domains[domain]["temporal_ious_2"].append(temporal_iou_2)
        durations[duration]["temporal_ious_2"].append(temporal_iou_2)
        data[idx]["temporal_IoU_2"] = temporal_iou_2

        # compute answer_spatial
        if 'answer_spatial' in item and item['answer_spatial']:
            aps, mIoU = calculate_spatial_metrics(item['bboxes'], item['answer_spatial'])
        else:
            aps, mIoU = [0.0] * 5, 0.0
        for i, ap in enumerate(aps):
            domains[domain]["spatial_aps"][i].append(ap)
            durations[duration]["spatial_aps"][i].append(ap)
            overall_stats["spatial_aps"][i].append(ap)
        domains[domain]["spatial_mious"].append(mIoU)
        durations[duration]["spatial_mious"].append(mIoU)
        overall_stats["spatial_mious"].append(mIoU)
        data[idx]["AP1@0.1:0.9"] = aps
        data[idx]["spatial_mIoU"] = mIoU

        # compute answer_spatial_2
        if 'answer_spatial_2' in item and item['answer_spatial_2']:
            aps_2, mIoU_2 = calculate_spatial_metrics(item['bboxes'], item['answer_spatial_2'])
        else:
            aps_2, mIoU_2 = [0.0] * 5, 0.0
        for i, ap in enumerate(aps_2):
            domains[domain]["spatial_aps_2"][i].append(ap)
            durations[duration]["spatial_aps_2"][i].append(ap)
            overall_stats["spatial_aps_2"][i].append(ap)
        domains[domain]["spatial_mious_2"].append(mIoU_2)
        durations[duration]["spatial_mious_2"].append(mIoU_2)
        overall_stats["spatial_mious_2"].append(mIoU_2)
        data[idx]["AP2@0.1:0.9"] = aps_2
        data[idx]["spatial_mIoU_2"] = mIoU_2


        with open(f'metrics/{model_name}_merged_v2_metrics.json', 'w') as f:
            json.dump(data, f, indent=4)
        
        if score >= 2 and temporal_iou >= 0.3:
            domains[domain]["vqa_temporal_idx"].append(idx)
            durations[duration]["vqa_temporal_idx"].append(idx)
            overall_stats["vqa_temporal_idx"].append(idx)
        if score >= 2 and temporal_iou_2 >= 0.3:
            domains[domain]["vqa_temporal_idx_2"].append(idx)
            durations[duration]["vqa_temporal_idx_2"].append(idx)
            overall_stats["vqa_temporal_idx_2"].append(idx)
        if score >= 2 and mIoU >= 0.1:
            domains[domain]["vqa_spatial_idx"].append(idx)
            durations[duration]["vqa_spatial_idx"].append(idx)
            overall_stats["vqa_spatial_idx"].append(idx)
        if score >= 2 and mIoU_2 >= 0.1:
            domains[domain]["vqa_spatial_idx_2"].append(idx)
            durations[duration]["vqa_spatial_idx_2"].append(idx)
            overall_stats["vqa_spatial_idx_2"].append(idx)
        if temporal_iou >= 0.3 and mIoU >= 0.1:
            domains[domain]["temporal_spatial_idx"].append(idx)
            durations[duration]["temporal_spatial_idx"].append(idx)
            overall_stats["temporal_spatial_idx"].append(idx)
        if temporal_iou_2 >= 0.3 and mIoU_2 >= 0.1:
            domains[domain]["temporal_spatial_idx_2"].append(idx)
            durations[duration]["temporal_spatial_idx_2"].append(idx)
            overall_stats["temporal_spatial_idx_2"].append(idx)
        if score >= 2 and temporal_iou >= 0.3 and mIoU >= 0.1:
            domains[domain]["vqa_temp_spatial_idx"].append(idx)
            durations[duration]["vqa_temp_spatial_idx"].append(idx)
            overall_stats["vqa_temp_spatial_idx"].append(idx)
        if score >= 2 and temporal_iou_2 >= 0.3 and mIoU_2 >= 0.1:
            domains[domain]["vqa_temp_spatial_idx_2"].append(idx)
            durations[duration]["vqa_temp_spatial_idx_2"].append(idx)
            overall_stats["vqa_temp_spatial_idx_2"].append(idx)

    with open(f'metrics/{model_name}_merged_metrics.json', 'w') as f:
            json.dump(data, f, indent=4)

    def print_stats(label, stats, total_samples):
        avg_all_score = np.mean(stats["all_rating"])
        avg_valid_score = np.mean(stats["valid_rating"]) if stats["valid_rating"] else 0
        acc_vqa = stats["correct_num"] / total_samples

        r1_iou30 = np.mean([1 if iou >= 0.3 else 0 for iou in stats["temporal_ious"]])
        r1_iou50 = np.mean([1 if iou >= 0.5 else 0 for iou in stats["temporal_ious"]])
        r1_iou70 = np.mean([1 if iou >= 0.7 else 0 for iou in stats["temporal_ious"]])
        mean_temporal_iou = np.mean(stats["temporal_ious"])

        r1_iou30_2 = np.mean([1 if iou >= 0.3 else 0 for iou in stats["temporal_ious_2"]])
        r1_iou50_2 = np.mean([1 if iou >= 0.5 else 0 for iou in stats["temporal_ious_2"]])
        r1_iou70_2 = np.mean([1 if iou >= 0.7 else 0 for iou in stats["temporal_ious_2"]])
        mean_temporal_iou_2 = np.mean(stats["temporal_ious_2"])

        mean_aps = [np.mean(ar_list) for ar_list in stats["spatial_aps"]]
        mean_miou = np.mean(stats["spatial_mious"])

        mean_aps_2 = [np.mean(ar_list) for ar_list in stats["spatial_aps_2"]]
        mean_miou_2 = np.mean(stats["spatial_mious_2"])


        vqa_temp = len(stats["vqa_temporal_idx"]) / total_samples
        vqa_temp_2 = len(stats["vqa_temporal_idx_2"]) / total_samples
        vqa_spat = len(stats["vqa_spatial_idx"]) / total_samples
        vqa_spat_2 = len(stats["vqa_spatial_idx_2"]) / total_samples
        temp_spat = len(stats["temporal_spatial_idx"]) / total_samples
        temp_spat_2 = len(stats["temporal_spatial_idx_2"]) / total_samples
        vqa_temp_spat = len(stats["vqa_temp_spatial_idx"]) / total_samples
        vqa_temp_spat_2 = len(stats["vqa_temp_spatial_idx_2"]) / total_samples

        print(f"{label}:")
        print(f"VQA: Avg All Score: {avg_all_score:.4f}, Avg Valid Score: {avg_valid_score:.4f}, Accuracy: {acc_vqa:.4f}")
        print("Chain 1:")
        print(f"Temporal Answer: R1@IoU=0.3: {r1_iou30:.4f}, R1@IoU=0.5: {r1_iou50:.4f}, R1@IoU=0.7: {r1_iou70:.4f}, Mean IoU: {mean_temporal_iou:.4f}")
        print(f"Spatial Answer: mAP@0.1: {mean_aps[0]:.4f}, mAP@0.3: {mean_aps[1]:.4f}, mAP@0.5: {mean_aps[2]:.4f}, mAP@0.7: {mean_aps[3]:.4f}, mAP@0.9: {mean_aps[4]:.4f}, Mean mIoU: {mean_miou:.4f}")
        print("\n")  
        print("Chain 2:")
        print(f"Temporal Answer: R1@IoU=0.3: {r1_iou30_2:.4f}, R1@IoU=0.5: {r1_iou50_2:.4f}, R1@IoU=0.7: {r1_iou70_2:.4f}, Mean IoU: {mean_temporal_iou_2:.4f}")
        print(f"Spatial Answer: mAP@0.1: {mean_aps_2[0]:.4f}, mAP@0.3: {mean_aps_2[1]:.4f}, mAP@0.5: {mean_aps_2[2]:.4f}, mAP@0.7: {mean_aps_2[3]:.4f}, mAP@0.9: {mean_aps_2[4]:.4f}, Mean mIoU: {mean_miou_2:.4f}")
        print("\n")

        AM = (acc_vqa + mean_temporal_iou + mean_miou)/3
        AM2 = (acc_vqa + mean_temporal_iou_2 + mean_miou_2)/3
        mAM = (AM + AM2) / 2

        LGM = -(math.log(1 - acc_vqa) + math.log(1 - mean_temporal_iou) + math.log(1 - mean_miou)) / 3     
        LGM2 = -(math.log(1 - acc_vqa) + math.log(1 - mean_temporal_iou_2) + math.log(1 - mean_miou_2)) / 3
        mLGM = (LGM + LGM2) / 2

        print(f"AM1:{AM:.4f}, AM2:{AM2:.4f}, mAM:{mAM:.4f}")
        print(f"LGM1:{LGM:.4f}, LGM2:{LGM2:.4f}, mLGM:{mLGM:.4f}\n")

        print("Joint Performance:")
        print(f"VQA & Temp:  Chain 1: {vqa_temp:.4f}, Chain 2: {vqa_temp_2:.4f}")
        print(f"VQA & Spat: Chain 1: {vqa_spat:.4f} Chain 2: {vqa_spat_2:.4f}")
        print(f"Temp & Spat:  Chain 1: {temp_spat:.4f} Chain 2: {temp_spat_2:.4f}")
        print(f"VQA & Temp & Spat:  Chain 1:{vqa_temp_spat:.4f} Chain 2: {vqa_temp_spat_2:.4f}")
        print(f"VQA & Temp list: \n Chain 1:{stats['vqa_temporal_idx']} \nChain 2:{stats['vqa_temporal_idx_2']}")
        print(f"VQA & Spat list: \n Chain 1:{stats['vqa_spatial_idx']} \n Chain 2: {stats['vqa_spatial_idx_2']}")
        print(f"Temp & Spat list:  \n Chain 1:{stats['temporal_spatial_idx']} \n Chain 2: {stats['temporal_spatial_idx_2']}")
        print(f"VQA & Temp & Spat list: \n Chain 1:{stats['vqa_temp_spatial_idx']} \n Chain 2:{stats['vqa_temp_spatial_idx_2']}\n")
       
    print_stats("Overall Statistics", overall_stats, len(data))
    for duration, stats in durations.items():
        print_stats(f"Video Length: {duration}", stats, len(stats["all_rating"]))
    for domain, stats in domains.items():
        print_stats(f"Domain: {domain}", stats, len(stats["all_rating"]))


# print("\nEvaluating GPT-4o:\n")
# evaluate_json('results/gpt4o/gpt4o_answer_update_merged.json')

# print("\nEvaluating Gemini-2-Flash:\n")
# evaluate_json('results/gemini2/gemini2_answer_merged.json')

# print("\nEvaluating Video-Llama3:\n")
# evaluate_json('results/videollama3/videollama3_answer_merged.json')

print("\nEvaluating Qwen2.5-VL:")
evaluate_json('results/qwen2_5/qwen2_5vl_answer_merged.json')

# print("\nEvaluating InternVL-2.5:")
# evaluate_json('results/internvl2_5/internvl2_5_answer_merged.json')

# print("\nEvaluating Llava-Video:")
# evaluate_json('results/llava-video/llavavideo_answer_merged.json')

# print("\nEvaluating Qwen2-VL:")
# evaluate_json('results/qwen2/qwen2vl_answer_merged.json')

# print("\nEvaluating VideoChat2:")
# evaluate_json('results/videochat2/videochat2_answer_merged.json')

# print("\nEvaluating Oryx-1.5:")
# evaluate_json('results/oryx-1.5/oryx15_answer_merged.json')

# print("\nEvaluating VideoCCAM:")
# evaluate_json('results/videoccam/videoccam12_answer_merged.json')

# print("\nEvaluating TimeChat:")
# evaluate_json('results/timechat/timechat_answer_merged.json')

# print("\nEvaluating VTimeLLM:")
# evaluate_json('results/vtimellm/vtimellm_answer_merged.json')

# print("\nEvaluating Trace:")
# evaluate_json('results/trace/trace_answer_merged.json')

# print("\nEvaluating Sa2VA:")
# evaluate_json('results/sa2va/sa2va_answer_merged.json')


