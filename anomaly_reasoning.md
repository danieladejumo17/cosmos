Write a script 'vllm_cosmos_reasoning.py' to perform anomlay reasoning on videos in the 'generated_vids' folder.

- Use the notebook 'cookbooks/cosmos3/reasoner/run_with_vllm.ipynb' for reference
- The anomaly reasoning script reasons on a video and a text prompt and gives a final output of whether the scenerio is indicates an anomaly requiring intervention or not.
- Add functions that parse the VLM output and classifies it as anomaly or not.

## Cosmos Model:
- The script uses Cosmos3 Nano or Cosmos3 Super. Add a command line arg for this with default Cosmos3 Nano
- When using Cosmos3 Nano, vllm should use a single GPU
- When using Cosmos3 Super, vllm should use 4 GPUs with tensor parrallelism of 4

## Base Prompt:
1. Action Grounding Mode:
"""
You are an autonomous driving safety expert analyzing this ego vehicle's video for semantic or contextual anomalies, that may impact safe AV operation. The ego vehicle's action (state sequence) given afterwards is in the format [[velocity_in_mph, heading_in_degrees], ...]. Think about the video and the ego vehicle's action, is there any semantic misunderstanding of the autopilot that requires intervention? Reply with exactly one word of the following:
Classification: Anomaly — if there is a semantic anomaly
Classification: Normal — if there is no semantic anomaly.
"""

2. No Action Grounding Mode:
"""
You are an autonomous driving safety expert analyzing this ego vehicle's video for semantic or contextual anomalies, that may impact safe AV operation. Think about the video, is there any semantic misunderstanding of the autopilot that requires intervention? Reply with exactly one word of the following:
Classification: Anomaly — if there is a semantic anomaly
Classification: Normal — if there is no semantic anomaly.
"""

## Operating Modes:
The user should be able to slelect the inference mode which changes whak kind of prompt is sent to the VLM
1. Action_Grounding
- The text prompt sent to the VLM is combination of the base prompt and the ego vehicle action sequence. <base_prompt> <"Ego Vehicle State Sequence (5Hz): "> <ego_vehicle_action_sequence>
- The ego vehicle action sequence is a .txt file, located in the same folder as the video we are performing inference on. The action sequence txt file has the same stem name as the filename of the video

2. No Action_Grounding
- The text prompt sent to the VLM is only the base prompt without the ego vehicle action sequence. <base_prompt>

## Metrics:
- Compute the accuracy, TP, FP, TN, FN, Precision, Recall, F1-Score, and inference time for the inference on the entire dataset
- Create a Metrics object in a utils.py for tracking the performance during inference and computing the final metrics at the end of inference on the folder
- Videos in the negative_scenerios folder have a ground truth label of 'anomaly'
- Videos in the positive_scenerios folder have a ground truth label of 'normal'

## Inference Experiment Results:
- For each experiment run, log a detailed JSON report of the experiment setup, environment, time, machine, cli args, inference parameters, and per video inference result and metrics. Add the average inference metrics as well.
- Save this to a log folder in the root of the repo - logs/exp_name_[action]_[timestamp]/. If action grounding mode is used, folder name should include 'action', otherwise it should not.

## Optimizations
- Inference FPS is 4fps
- Max Output token is 100


Add the dataset folder and the experiment name to the CLI args of the script. An example implementation for Cosmos Reason1 that may guide your implementation is in 'fp16_inference_cosmos_reason1.py'. Videos in the provided dataset folder already have the right duration.
