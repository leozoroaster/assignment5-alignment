import torch
import grpo
from vllm import LLM, SamplingParams
import SFT_helpers
import random

def transform_model(vllm_model):
    return

def sample_dataset(dataset, train_batch_size):
    n = len(dataset["questions"])
    idx = random.sample(range(n), k=min(train_batch_size, n))
    return {
        "questions": [dataset["questions"][i] for i in idx],
        "answers": [dataset["answers"][i] for i in idx],
    }

def sample_batch(input_ids, labels, rollout_batch_size):
    n = len(input_ids)
    idx = random.sample(range(n), k=min(rollout_batch_size, n))
    q_batch = [input_ids[i] for i in idx]
    a_batch = [labels[i] for i in idx]
    return q_batch, a_batch

def sample_response(q_batch, policy):
    return

def flatten_answer(a_batch,group_size):
    return [a for a in a_batch for _ in range(group_size)]

def grpo_train(
    dataset,
    reward_fn,
    tokenizer,
    model="Qwen/Qwen2.5-Math-1.5B",
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 1,
    train_batch_size: int = 256,
    gradient_accumulation_steps: int = 128,
    gpu_memory_utilization: float = 0.85,
    loss_type = "reinforce_with_baseline",
    use_std_normalization: bool = True,
):

    dataset=sample_dataset(dataset, train_batch_size)
    prompt_strs=dataset["questions"]
    output_strs=dataset["answers"]
    output_dict=SFT_helpers.tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)

    input_ids = output_dict["input_ids"]
    labels = output_dict["labels"]
    response_mask= output_dict["response_mask"]

    sampling_params = SamplingParams(
        temperature=sampling_temperature, min_tokens=sampling_min_tokens, max_tokens=sampling_max_tokens, stop=["</answer>"], include_stop_str_in_output=True
    )

    vllm_model=LLM(model=model, gpu_memory_utilization=gpu_memory_utilization)
    policy=transform_model(model)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    for step in range(n_grpo_steps):
        q_batch, a_batch=sample_batch(input_ids, labels, rollout_batch_size)

        rollout_responses = sample_response(q_batch, policy)

        repeated_ground_truths=flatten_answer(a_batch,group_size)

        advantages, raw_rewards, _=grpo.compute_group_normalized_rewards(reward_fn, rollout_responses, repeated_ground_truths, group_size, advantage_eps, use_std_normalization)

        old_log_probs=SFT_helpers.get_response_log_probs(policy, q_batch, a_batch)["log_probs"]

        for t in range(epochs_per_rollout_batch):
            mini_batch_size=train_batch_size//gradient_accumulation_steps
            for inner_step in range(gradient_accumulation_steps):
                start, end = inner_step*mini_batch_size, (inner_step+1)*mini_batch_size

                policy_log_probs=SFT_helpers.get_response_log_probs(policy, q_batch[start:end], a_batch[start:end])["log_probs"]

                curr_policy_log_probs, curr_response_mask, curr_raw_rewards, curr_advantages, curr_old_log_probs=policy_log_probs, response_mask[start:end], raw_rewards[start:end], advantages[start:end], old_log_probs[start:end]

                grpo.grpo_microbatch_train_step(curr_policy_log_probs, curr_response_mask, gradient_accumulation_steps, loss_type, curr_raw_rewards, curr_advantages, curr_old_log_probs, cliprange=0.1)

                if (inner_step+1)%gradient_accumulation_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()

    return policy
