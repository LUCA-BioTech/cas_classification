import torch

def get_masked_attention(attention_weights, attention_mask):
    # attention_weights is a PyTorch tensor with shape [batch_size, heads, sequences, sequences]

    num_heads, _, _ = attention_weights.shape
    head_masked_attention = []

    for i in range(num_heads):

        first_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].min() + 1
        last_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].max() - 1

        attention = attention_weights[i]

        # Get the submatrix of high_conf_attention_in_this_head
        masked_attention = attention[first_nonzero_idx:last_nonzero_idx+1, first_nonzero_idx:last_nonzero_idx + 1]

        head_masked_attention.append(masked_attention)

    return head_masked_attention

def get_batch_head_attention(attention_weights, threshold, attention_masks):
    # attention_weights is a PyTorch tensor with shape [batch_size, heads, sequences, sequences]
    # threshold is a float number used to select high confidence attention

    batch_size, num_heads, _, _ = attention_weights.shape
    high_confidence_attention = torch.zeros((batch_size, num_heads), device=attention_weights.device)

    for i in range(batch_size):

        attention_mask = attention_masks[i]
        first_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].min() + 1
        last_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].max() - 1

        for head in range(num_heads):
            attention = attention_weights[i, head]

            # Get the submatrix of high_conf_attention_in_this_head
            masked_attention = attention[first_nonzero_idx:last_nonzero_idx+1, first_nonzero_idx:last_nonzero_idx + 1]

            # Use a mask to select attention weights larger than the threshold
            mask = masked_attention > threshold
            high_confidence_attention[i, head] = mask.float().mean()

    return high_confidence_attention


def get_batch_head_sequence_attention(attention_weights, threshold, method, attention_masks):
    # attention_weights is a PyTorch tensor with shape [batch_size, heads, sequences, sequences]
    # method could be 'max', 'average', or 'high_confidence_average
    # threshold is a float number used to select high confidence attention

    batch_size, num_heads, max_length, _ = attention_weights.shape
    result = torch.zeros((batch_size, num_heads, max_length), device=attention_weights.device)

    for i in range(batch_size):

        attention_mask = attention_masks[i]
        first_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].min() + 1
        last_nonzero_idx = (attention_mask != 0).nonzero(as_tuple=True)[0].max() - 1

        for head in range(num_heads):
            for seq_i in range(max_length):
                if seq_i < first_nonzero_idx or seq_i > last_nonzero_idx:
                    continue
                attention = attention_weights[i, head, seq_i, :]

                # Get the submatrix of high_conf_attention_in_this_head
                masked_attention = attention[first_nonzero_idx:last_nonzero_idx+1]

                if method == 'max':
                    result[i, head, seq_i] = masked_attention.max()
                elif method == 'average':
                    result[i, head, seq_i] = masked_attention.mean()
                elif method == 'high_confidence_average':
                    mask = masked_attention > threshold
                    if mask.sum().item() > 0:
                        result[i, head, seq_i] = masked_attention[mask].mean()
    return result
