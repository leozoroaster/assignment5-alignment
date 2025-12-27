import torch

def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    response_tokens=[]
    prompt_lens=[]
    output_lens=[]
    batch_size=len(prompt_strs)
    max_len=0

    for i in range(batch_size):
        curr_q=prompt_strs[i]
        curr_o=output_strs[i]
        tokenized_q=tokenizer(curr_q)["input_ids"]
        prompt_lens.append(len(tokenized_q))
        tokenized_o=tokenizer(curr_o)["input_ids"]
        output_lens.append(len(tokenized_o))
        combined=tokenized_q+tokenized_o
        response_tokens.append(combined)
        max_len=max(max_len, len(combined))

    output_dict=dict()
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    raw_tensor=torch.full((batch_size, max_len), pad_id, dtype=torch.long)

    for i in range(batch_size):
        new_row=response_tokens[i]
        raw_tensor[i, :len(new_row)]=torch.tensor(new_row, dtype=torch.long)

    input_ids=raw_tensor[:,:-1]
    labels=raw_tensor[1:,:]
    response_mask=torch.zeros(batch_size, max_len-1)
    for i in range(batch_size):
        response_mask[i,prompt_lens[i]-1:prompt_lens[i]-1+output_lens[i]]=torch.ones(output_lens[i])

    output_dict["input_ids"]=input_ids
    output_dict["labels"] = labels
    output_dict["response_mask"] = response_mask

    return output_dict

def compute_entropy(logits:torch.Tensor)-> torch.Tensor:
    max_value=torch.max(logits, dim=-1, keepdim=True).values
    logits-=max_value
    exp_logits=torch.exp(logits)
    sum_exp=torch.sum(exp_logits,dim=-1,keepdim=True)
    probs=exp_logits/sum_exp
    return -probs*(logits-torch.log(sum_exp)).sum(dim=-1)

def get_response_log_probs(
    model,
    input_ids,
    labels,
    return_token_entropy=False,
    )-> dict[str, torch.Tensor]:

    out = model(input_ids)
    logits = out.logits
    B, T, V = logits.shape

    log_probs_vocab = torch.nn.functional.log_softmax(logits, dim=-1)

    target = labels[:, 1:]

    log_probs_next = log_probs_vocab[:, :-1, :].gather(
        dim=-1, index=target.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    log_probs = torch.zeros((B, T), device=logits.device, dtype=log_probs_next.dtype)
    log_probs[:, 1:] = log_probs_next

    result = {"log_probs": log_probs}

    if return_token_entropy:
        result["token_entropy"] = compute_entropy(logits)  # (B, T)

    return result

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant:float,
    dim:int|None= None,
)-> torch.Tensor:

    tensor=tensor*mask

    tensor=torch.sum(tensor, dim=dim)

    tensor/=normalize_constant

    return tensor

def sft_microbatch_train_step(
    policy_log_probs:torch.Tensor,
    response_mask:torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant:float =1.0,
)-> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    normalized_log_probs=masked_normalize(policy_log_probs, response_mask, normalize_constant)

    neg_log_probs=-normalized_log_probs

    loss=neg_log_probs/gradient_accumulation_steps

    loss.backward()

    metadata = {
        "loss_scaled": loss.detach(),
    }
    return loss.detach(), metadata
