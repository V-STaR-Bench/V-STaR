# **Download the Dataset**

Video: Please down the video data from [HuggingFace](https://huggingface.co/datasets/V-STaR-Bench/V-STaR).

Annotation: You can download the annotation from here or from [HuggingFace](https://huggingface.co/datasets/V-STaR-Bench/V-STaR).

## Annotation Structure

Each annotation is organized by

```
{
  "vid": ...,								# Video ID
  "domain": ..., 
  "fps": ..., 
  "width": ..., 
  "height": ..., 
  "frame_count": ..., 							# total frame number of the video
  "question": ..., 							# VQA question
  "chain": "<think>...<think>", # spatio-temporal thinking chain
  "object": ..., 							# object of the boxes
  "answer": ..., 
  "temporal_question": ..., 						# temporal grounding question
  "timestamps": [..., ...], 
  "spatial_question": ..., 						# Chain 1 Spatial grounding question
  "spatial_question_2": ..., 						# Chain 2 Spatial grounding question	
  "bboxes": [
	{"{timestamp}_{frame_index}":{"xmin": ...,"ymin": ..., "xmax": ..., "ymax": ...}},...
  ]
}
```

