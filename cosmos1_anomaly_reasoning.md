Write a transformers_cosmos1_reasoning.py similar to 'vllm_cosmos_reasoning.py', but which uses cosmos reason1 and the hugging face transformers library instead. 

-  Append reason1 to the logs folder naming i.e. ..._reason1_[timestamp]
- Use this prompt for the action grounding mode:
"""
You are an autonomous driving safety expert analyzing this ego vehicle's video for semantic or contextual anomalies, that may impact safe AV operation. The ego vehicle's action (state sequence) given afterwards is in the format [[velocity_in_mph, heading_in_degrees], ...].

<think>
Think about the video scenario and the ego vehicle's action.
</think>

<answer>
Is there any semantic misunderstanding of the autopilot that requires intervention? Reply with exactly one word of the following:
Classification: Anomaly — if there is a semantic anomaly
Classification: Normal — if there is no semantic anomaly.
</answer>
"""

- Use this prompt for the no action grounding mode:
"""
You are an autonomous driving safety expert analyzing this ego vehicle's video for semantic or contextual anomalies, that may impact safe AV operation.

<think>
Think about the video scenario.
</think>

<answer>
Is there any semantic misunderstanding of the autopilot that requires intervention? Reply with exactly one word of the following:
Classification: Anomaly — if there is a semantic anomaly
Classification: Normal — if there is no semantic anomaly.
</answer>
"""

- Create a setup_cosmos1.sh script to install the necessary libraries