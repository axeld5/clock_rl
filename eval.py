import time
import torch
from qwen_vl_utils import process_vision_info

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

trained_model_id = "axel-darmouni/Qwen2.5-VL-3B-Instruct-Clock"

trained_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    trained_model_id,
    torch_dtype="auto",
    device_map="auto",
)
trained_processor = AutoProcessor.from_pretrained(trained_model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def generate_with_reasoning(problem, image):
    # Conversation setting for sending to the model
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": problem},
            ],
        },
    ]
    prompt = trained_processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)

    # Process images using the process_vision_info from qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = trained_processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(trained_model.device)

    # Generate text without gradients
    start_time = time.time()
    with torch.no_grad():
        output_ids = trained_model.generate(**inputs, max_new_tokens=500)
    end_time = time.time()

    # Decode and extract model response
    generated_text = trained_processor.decode(output_ids[0], skip_special_tokens=True)

    # Get inference time
    inference_duration = end_time - start_time

    # Get number of generated tokens
    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens