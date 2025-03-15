![vbench_logo](https://raw.githubusercontent.com/Vchitect/VBench/master/asset/vbench_logo_github_20240605.jpg)

<!-- [![arXiv](https://img.shields.io/badge/arXiv-2311.99999-b31b1b.svg)](https://arxiv.org/abs/2311.99999) -->
[![V-STaR Paper](https://img.shields.io/badge/ArXiV%202025-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2311.17982)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/V-STaR-Bench/V-STaR-LeaderBoard)
[![Dataset Download](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset%20Download-blue)](https://huggingface.co/spaces/Vchitect/VBench_Video_Arena)
[![Project Page](https://img.shields.io/badge/VSTaR-Website-green?logo=googlechrome&logoColor=green)](https://v-star-bench.github.io/)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FV-STaR-Bench%2FV-STaR&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)


This repository contains the implementation of the following paper and its related serial works in progress. We evaluate Video LLMs models!
> **V-STaR: Benchmarking Video-LLMs on Video Spatio-Temporal Reasoning**<br>
> <p align="center">
    <a href="https://lwpyh.github.io/">Jian Hu</a>,&nbsp;&nbsp;
    <a href="https://zxccade.github.io/">Zixu Cheng</a>,&nbsp;&nbsp;
    <a href="https://chenyangsi.top/">Chenyang Si</a>,&nbsp;&nbsp;
    <a href="https://weivision.github.io/">Wei Li</a>,&nbsp;&nbsp;
    <a href="http://www.eecs.qmul.ac.uk/~sgg/">Shaogang Gong</a>
</p>



### Table of Contents
- [Updates](#updates)
- [Overview](#overview)
- [Evaluation Results](#evaluation_results)
- [Usage](#usage)
- [Citation and Acknowledgement](#citation_and_acknowledgement)

<a name="overview"></a>
## :mega: Overview
![overall_structure](./asset/fig_extention_teaser.jpg)
We propose **VBench**, a comprehensive benchmark suite for video generative models. We design a comprehensive and hierarchical <b>Evaluation Dimension Suite</b> to decompose "video generation quality" into multiple well-defined dimensions to facilitate fine-grained and objective evaluation. For each dimension and each content category, we carefully design a <b>Prompt Suite</b> as test cases, and sample <b>Generated Videos</b> from a set of video generation models. For each evaluation dimension, we specifically design an <b>Evaluation Method Suite</b>, which uses carefully crafted method or designated pipeline for automatic objective evaluation. We also conduct <b>Human Preference Annotation</b> for the generated videos for each dimension, and show that VBench evaluation results are <b>well aligned with human perceptions</b>. VBench can provide valuable insights from multiple perspectives. <b>VBench++</b> supports a wide range of video generation tasks, including text-to-video and image-to-video, with an adaptive Image Suite for fair evaluation across different settings. It evaluates not only technical quality but also the trustworthiness of generative models, offering a comprehensive view of model performance. We continually incorporate more video generative models into VBench to inform the community about the evolving landscape of video generation.

<a name="evaluation_results"></a>
## :mortar_board: Evaluation Results

***See our leaderboard for the most updated ranking and numerical results (with models like GPT-4o, Gemini-2-flash and Qwen2.5-VL)***. [![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-blue)](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)

<p align="center">
  <img src="./asset/all-dim.jpg" width="65%"/>
</p>
We visualize the evaluation results of the 6 most recent top-performing Video-LLMs across 9 V-STaR domains.

<p align="center">
  <img src="./asset/radar-open-new.jpg" width="48%" style="margin-right: 4%;" />
  <img src="./asset/radar-close-new.jpg" width="48%" />
</p>

Additionally, we present radar charts separately for the evaluation results of open-source and closed-source models. The results are normalized per dimension for clearer comparisons.

#### :trophy: Leaderboard

See numeric values at our [Leaderboard](https://huggingface.co/spaces/V-STaR-Bench/V-STaR-LeaderBoard) :1st_place_medal::2nd_place_medal::3rd_place_medal:

**How to join VBench Leaderboard?**

please contact us via email to update your results.

#### Evaluation Criterion

To evaluate the open-ended *"what"* question, we use Qwen2.5-72B-Instruct to score answers from 0 to 4, denoting entirely incorrect, largely incorrect, largely correct, and entirely correct. Answers scoring above 2 are considered correct, allowing us to compute accuracy. 

For the *"when"* question, we follow the commonly used temporal grounding metrics, <R@n, tIoU=m>, which refers to the percentage of top-n prediction with temporal IoU score larger than m, and mean temporal IoU score (m\_tIoU). 

For the *"where"* question, we use the Average Precision score (AP@vIoU=m) and mean visual Intersection over the Union (m\_vIoU) of every annotated frame. We follow the proposed LGM and AM to measure a model's spatial-temporal reasoning ability. A higher LGM indicates a better overall spatio-temporal reasoning ability of the model, and a higher AM indicates a more average performance of the model on the three metrics.

<a name="usage"></a>
## Usage
Use V-STaR to evaluate Video-LLMs:

We provide our inference_demo.py script to test Qwen2.5-VL-7B with:

```
python inference_demo.py 
```
You can try your Video-LLMs to infer on V-STaR based on the provided scripts to test the model's spatio-temporal reasoning ability.

To evaluate the results, update your result file path in the eval.py script and run:

```
python eval.py
```
Noted: You need at least 2 NVIDIA A100 80G GPUs to run Qwen-2.5-72B for evaluation.

### Submit to Leaderboard

please contact us via email to update your results.

<a name="citation_and_acknowledgement"></a>

## :black_nib: Citation

   If you find our repo useful for your research, please consider citing our paper:

   ```bibtex
    @InProceedings{
    }
   ```
