from datasets import load_dataset
from transformers import AutoProcessor

dataset_id = "rohitsaxena/DateTimeQA"
dataset = load_dataset(dataset_id)
dataset = dataset["clock"]
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(model_id, use_fast=True, padding_side="left")

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>."
)


def make_conversation(example):
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example["question"]},
            ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return {
        "prompt": prompt,
        "image": example["image"],
    }


train_dataset = dataset.map(make_conversation)
train_dataset = train_dataset.remove_columns(["type", "question"])

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    pretrained_model_name_or_path=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

import re


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def format_hour_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\d{1,2}:\d{2}:\d{2}.*?\n</answer>$"
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    return rewards

def accuracy_reward(completions: list[str], solution: list[str], **kwargs) -> list[float]:
    """Reward function that checks if the completion matches the ground truth.
    Ground truth is expected to be in the format matching format_hour_reward pattern.
    """
    rewards = []
    
    # Pattern to extract time from answer section
    time_pattern = r"<answer>\n.*?(\d{1,2}:\d{2}:\d{2}).*?\n</answer>"
    
    for completion, sol in zip(completions, solution):
        # Extract time from completion
        completion_match = re.search(time_pattern, completion, re.DOTALL | re.MULTILINE)
        completion_time = completion_match.group(1) if completion_match else None
        
        # Extract time from solution
        solution_match = re.search(time_pattern, sol, re.DOTALL | re.MULTILINE)
        solution_time = solution_match.group(1) if solution_match else None
        
        # Compare extracted times
        if completion_time and solution_time:
            reward = 1.0 if completion_time == solution_time else 0.0
        else:
            # Fallback to exact text match if time extraction fails
            reward = 1.0 if completion.strip() == sol.strip() else 0.0
        
        rewards.append(reward)
    
    return rewards


from trl import GRPOConfig

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2.5-VL-3B-Instruct-Clock",
    learning_rate=1e-5,
    remove_unused_columns=False,  # to access the solution column in accuracy_reward
    num_train_epochs=1,
    bf16=True,
    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024,  # default: 256
    num_generations=2,  # default: 8
    max_prompt_length=2048,
    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=True,
    save_strategy="steps",
    save_steps=10,
)

from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    processing_class=processor,
    reward_funcs=[format_reward, format_hour_reward, accuracy_reward],
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
trainer.save_model(training_args.output_dir)
trainer.push_to_hub(dataset_name=dataset_id)