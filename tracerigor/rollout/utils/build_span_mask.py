import torch

def _build_span_mask(
    input_ids_1d: torch.Tensor,
    start_tokens: list[int],
    end_tokens: list[int],
) -> torch.Tensor:
    device = input_ids_1d.device
    L = input_ids_1d.shape[0]
    mask = torch.zeros(L, dtype=torch.float32, device=device)
    if L == 0:
        return mask

    ids = input_ids_1d.tolist()
    ns = len(start_tokens)
    ne = len(end_tokens)

    start_positions = []
    end_positions_last = []

    if ns > 0 and L >= ns:
        for i in range(0, L - ns + 1):
            if ids[i:i + ns] == start_tokens:
                start_positions.append(i)
    if ne > 0 and L >= ne:
        for i in range(0, L - ne + 1):
            if ids[i:i + ne] == end_tokens:
                end_positions_last.append(i + ne - 1)

    if len(start_positions) == 0 and len(end_positions_last) == 0:
        return mask

    device = input_ids_1d.device
    s_indices = torch.tensor(start_positions, device=device, dtype=torch.long)
    e_indices = torch.tensor(end_positions_last, device=device, dtype=torch.long)

    if e_indices.numel() > 0 and (s_indices.numel() == 0 or e_indices[0] < s_indices[0]):
        s_indices = torch.cat([torch.tensor([-1], device=device), s_indices])
    if s_indices.numel() > e_indices.numel():
        e_indices = torch.cat([e_indices, torch.tensor([L - 1], device=device)])

    for s_idx, e_idx in zip(s_indices, e_indices):
        s = int(s_idx.item())
        e = int(e_idx.item())
        if s < 0 and e < L:
            mask[: e + 1] = 1.0
        elif s >= 0 and e < L:
            mask[s : e + 1] = 1.0
        elif s >= 0 and e >= L:
            mask[s:] = 1.0
        elif s < 0 and e >= L:
            mask[:] = 1.0
    return mask

# For tests, pretend <think> = [101, 102], </think> = [201, 202]
start_tokens = [101, 102]
end_tokens = [201, 202]

def print_case(name, ids):
    ids_t = torch.tensor(ids, dtype=torch.long)
    mask = _build_span_mask(ids_t, start_tokens, end_tokens)
    print(f"{name}:")
    print("  ids: ", ids)
    print("  mask:", mask.tolist())
    print()

# 1) Normal single span: <think> x y </think>
print_case("normal span", [
    10, 101, 102, 30, 31, 201, 202, 11
])
# expected mask = 0, 1,1,1,1,1,1,0

# 2) Truncated start: we see only </think> in this chunk
print_case("truncated start", [
    30, 31, 201, 202, 11
])
# expected: we are inside a span from before: mask[0..end(</think>)] = 1

# 3) Truncated end: we see only <think> in this chunk
print_case("truncated end", [
    10, 101, 102, 30, 31
])
# expected: mask from <think> to end = 1

# 4) Two spans
print_case("two spans", [
    10, 101, 102, 30, 201, 202, 40, 101, 102, 50, 201, 202, 60
])
# expected: mask = 0, 1,1,1,1,1,0, 1,1,1,1,1,0