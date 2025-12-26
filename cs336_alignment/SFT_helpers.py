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