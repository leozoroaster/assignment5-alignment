import torch

def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
):

    roll_out_batch_size=len(rollout_responses)
    n_groups=roll_out_batch_size//group_size
    rewards=[]

    for i in range(roll_out_batch_size):
        reward=reward_fn(rollout_responses[i],repeated_ground_truths[i])["reward"]
        rewards.append(reward)

    raw_rewards = torch.tensor(rewards).view(n_groups, group_size)

    reward_mean=torch.mean(raw_rewards, dim=-1, keepdim=True)
    reward_std=torch.std(raw_rewards, dim=-1, keepdim=True)

    metadata=dict()
    metadata["mean"]=reward_mean
    metadata["std"]=reward_std

    advantages=raw_rewards-reward_mean

    if normalize_by_std:
        advantages/=(reward_std+advantage_eps)

    advantages/=group_size

    advantages=advantages.reshape(-1)
    raw_rewards=raw_rewards.reshape(-1)

    return advantages, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs:torch.Tensor,
)-> torch.Tensor:

    return -policy_log_probs*raw_rewards_or_advantages

def compute_grpo_clip_loss(
    advantages:torch.Tensor,
    policy_log_probs:torch.Tensor,
    old_log_probs:torch.Tensor,
    cliprange: float,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    adjusted_probs=torch.exp(policy_log_probs-old_log_probs)
    clipped_adjusted_probs=torch.clamp(adjusted_probs, 1- cliprange, 1+ cliprange)

    metadata=dict()

    raw_estimation=adjusted_probs*advantages
    clipped_estimation=clipped_adjusted_probs*advantages

    is_clipped = clipped_adjusted_probs != adjusted_probs
    metadata["is_clipped"]=is_clipped

    loss=-torch.minimum(raw_estimation, clipped_estimation)

    return loss, metadata

def compute_policy_gradient_loss(
    policy_log_probs:torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor|None =None,
    advantages: torch.Tensor| None=None,
    old_log_probs:torch.Tensor|None= None,
    cliprange: float |None=None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    assert loss_type in ["no_baseline", "reinforce_with_baseline", "grpo_clip"], print("wrong loss type")

    if loss_type=="no_baseline":
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), dict()

    if loss_type=="reinforce_with_baseline":
        adjusted_probs = torch.exp(policy_log_probs - old_log_probs)
        loss = -adjusted_probs * advantages
        return loss, dict()

    loss, metadata=compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    return loss, metadata

def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim:int |None=None,
)-> torch.Tensor:

    tensor=tensor*mask
    tensor_sum=torch.sum(tensor,dim=dim, keepdim=False)

    mask_sum=torch.sum(mask,dim=dim, keepdim=False)

    return tensor_sum/mask_sum.clamp(min=1)

def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: str,
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss, metadata= compute_policy_gradient_loss(policy_log_probs, loss_type, raw_rewards, advantages, old_log_probs, cliprange)
    scalar_loss=masked_mean(loss, response_mask, dim=-1)
    mean_loss=torch.mean(scalar_loss)/gradient_accumulation_steps

    mean_loss.backward()
    metadata["mean_loss"]=mean_loss

    return mean_loss, metadata
