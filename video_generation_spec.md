Create a 'generate_videos.py' script to perform text to video generation as done in 'cookbooks/cosmos3/generator/audiovisual/run_with_diffusers.ipynb'.

- Videos are to be generated with the Cosmos Nano model. (This should be controlled by a variable I can change)
- Videos are to be generated without audio
- The script should be able to run standalone in a python env with diffusers, dependencies, and env variables defined.

- The script has to iterate through a folder with JSON prompts and generate videos for each JSON prompt (e.g. ./prompts)
- The generated videos should be saved in './generated_vids' (same folder as the python script)
- The generated videos should have the same folder structure and naming convention as the JSON prompts
- Generated videos should be 5s long, 720p,

- After each video is generated, as async process should upload the video to huggingface dataset repo
- The videos should be upload to huggingface with the same folder structure and naming convention as done locally
- The huggingface dataset repo is 'danieladejumo/av_semantic_anomalies'

- Make the script as simple as possible. (e.g. no need for creating unnecessary payloads and other overheads)


// TODOs:
- The last segment/action should be at least 2s long
- Starting speed is ~30mph
- Decelerating to a full stop means speed has to drop to zero by the end of the last segment