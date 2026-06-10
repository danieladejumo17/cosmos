Write script(s) to perfrom inverse dynamics for all the videos in 'generated_vids'.

- The script should use the cosmos3 inverse dynamics mode as discussed in the README.md
- Reference the script used for performing inverse dynamics that is relevant to autonomous driving.

- The final action sequence required is [[velocity, heading_angle_from_center], ...]
- The final action sequence should be processed from the action returned from the cosmos3 model. If an initial speed is required for the computation, assume 30 mph.
- Downsample the final action sequence to 5Hz
- Ask for any other info required for processing the speed and vehicle heading (steering) from the action generated from cosmos3

- For each video, check the corresponding prompt.txt ('prompts/negative_scenarios/prompt.txt' for negative prompt). prompt_0_[].json is the first line in the prompt text. Validate that the tail of the generated action sequence matches the last sentence of the prompt text. E.g. if the prompt says the vehicle decelerates, the generated action sequence should reflect that.
- Flag action sequence that do not match the last sentence of the prompt.

- Save the generated action sequence next to each video as a .txt file with the same name as the video
- Upload the action sequence file to huggingface

TODO:
//// Change to a mode where the velocity is computed from the first few frames instead of anchoring to 30 at start
Which inverse-dynamics backend should the script use?

Cosmos Framework native PyTorch (python -m cosmos_framework.scripts.inference) - clones NVIDIA/cosmos-framework + uv venv, matches the README AV reference, no Docker needed (Recommended)
How should velocity be computed, and how is the assumed 30 mph initial speed used? (model outputs ego-pose deltas; translation_scale=1.35 yields approximate meters)

Anchor the first-frame speed to 30 mph and scale the model's per-step displacement profile to it (model gives the speed-profile shape; 30 mph sets absolute scale) (Recommended)
How should 'heading_angle_from_center' (steering) be defined, in degrees?

Per-step heading yaw relative to the first-frame heading: atan2(fwd_x, fwd_z) minus initial, in degrees (0