"""
Knowledge graph preprocessor for CARTE pretraining.
"""

import torch
from torch.utils.data import Dataset


def _create_pos(edge_index, edge_type, perturb_idx):
    mask = torch.zeros(edge_index.size(1), dtype=torch.bool)
    mask[perturb_idx] = True
    return edge_index[:, ~mask], edge_type[~mask]


def _truncate_graph(edge_index, edge_type):
    p = 0.2 if edge_type.size(0) >= 10 else 0.1
    num_perturb = 2 if torch.bernoulli(torch.tensor(p)) == 1 else 1
    perturb_idx = torch.randint(edge_type.size(0), (num_perturb,)).unique()
    edge_index_, edge_type_ = _create_pos(edge_index, edge_type, perturb_idx)
    return edge_index_, edge_type_


def _create_neg(edge_index, edge_type, perturb_idx, ent_list, ent_per_rel):
    candidate_list = ent_per_rel[edge_type[perturb_idx].item()]
    replace_ent = candidate_list[torch.randint(candidate_list.size(0), (1,))]
    while torch.isin(ent_list, replace_ent).nonzero().size(0) != 0:
        replace_ent = candidate_list[torch.randint(candidate_list.size(0), (1,))]
    edge_index_ = edge_index.clone()
    edge_index_[1, perturb_idx] = replace_ent
    return edge_index_, edge_type


def _remove_duplicates(edge_index):
    nnz = edge_index.size(1)
    num_nodes = edge_index.max().item() + 1
    idx = edge_index.new_empty(nnz + 1)
    idx[0] = -1
    idx[1:] = edge_index[0]
    idx[1:].mul_(num_nodes).add_(edge_index[1])
    idx[1:], perm = torch.sort(
        idx[1:],
    )

    mask = idx[1:] > idx[:-1]
    edge_index = edge_index[:, perm]
    edge_index = edge_index[:, mask]
    return edge_index


class CARTEKGPreprocessor:

    def __init__(
        self,
        data_kg_dir: str,
        num_hops: int = 1,
        num_sample: int = 128,
        num_pos: int = 7,
        max_rels: int = 30,
    ):
        super(CARTEKGPreprocessor, self).__init__()

        # Check if the appropriate sizes
        assert num_sample % 2 == 0
        assert (num_pos + 1) % 2 == 0
        assert num_sample / (num_pos + 1) % 2 == 0

        self.data_kg_dir = data_kg_dir
        self.num_hops = num_hops
        self.num_sample = num_sample
        self.num_pos = num_pos
        self.max_rels = max_rels

        self._set_preprocessor()

    def _set_preprocessor(self):

        self.edge_attr_total = torch.load(
            f"{self.data_kg_dir}/edge_attr_total.pt",
            weights_only=True,
            mmap=True,
        )
        self.x_total = torch.load(
            f"{self.data_kg_dir}/x_total.pt",
            weights_only=True,
            mmap=True,
        )
        self.edge_index = torch.load(
            f"{self.data_kg_dir}/edge_index.pt",
            weights_only=True,
            mmap=True,
        )
        self.edge_type = torch.load(
            f"{self.data_kg_dir}/edge_type.pt",
            weights_only=True,
            mmap=True,
        )
        self.ent_per_rel = torch.load(
            f"{self.data_kg_dir}/ent_per_rel.pt",
            weights_only=True,
            mmap=True,
        )

        slice_idx = torch.cumsum(torch.bincount(self.edge_index[0]), dim=0)
        self.slice_idx = torch.hstack([torch.zeros(1, dtype=torch.long), slice_idx])
        self.num_neg_ = int((self.num_sample / (self.num_pos + 1)) - 1)

        return None

    def __call__(self, center_indices):

        if isinstance(center_indices, int):
            center_indices = [center_indices]

        if isinstance(center_indices, list) == False:
            center_indices = center_indices.tolist()

        batch = [self._preprocess_sample(ent_idx) for ent_idx in center_indices]

        return batch

    def _preprocess_sample(self, ent_idx):

        # Obatin edge_index, and edge_type for the ent_idx
        # Need to fix later for multiple hops
        idx_slice1 = self.slice_idx[ent_idx]
        idx_slice2 = self.slice_idx[ent_idx + 1]
        edge_index = self.edge_index[:, idx_slice1:idx_slice2].clone()
        edge_type = self.edge_type[idx_slice1:idx_slice2].clone()

        # Control for max. number of rels
        if edge_type.size(0) > self.max_rels:
            idx_keep = torch.randperm(edge_type.size(0))[: self.max_rels]
            edge_index = edge_index[:, idx_keep]
            edge_type = edge_type[idx_keep]
        else:
            pass
        self.ent_list_ = edge_index.unique()

        sample_original = self._preprocess_data(
            ent_idx, edge_index, edge_type, "original"
        )
        sample_perturbed = [
            self._preprocess_data(ent_idx, edge_index, edge_type, "pos")
            for _ in range(self.num_pos)
        ]
        sample_perturbed = sum(sample_perturbed, [])

        return self._collate(sample_original + sample_perturbed)

    def _preprocess_data(self, ent_idx, edge_index, edge_type, perturb_type):

        if perturb_type == "pos":
            edge_index_, edge_type_ = _truncate_graph(edge_index, edge_type)
        else:
            edge_index_, edge_type_ = edge_index, edge_type

        # Generate positives and negatives
        data_pos = [
            self._preprocess_transformer(ent_idx, edge_index_, edge_type_, perturb_type)
        ]
        data_neg = [
            self._preprocess_transformer(ent_idx, edge_index_, edge_type_, "neg")
            for _ in range(self.num_neg_)
        ]

        return data_pos + data_neg

    def _preprocess_transformer(self, ent_idx, edge_index, edge_type, perturb_type):

        # Set the negative
        if perturb_type == "neg":
            perturb_idx = torch.randint(edge_type.size(0), (1,))
            edge_index_, edge_type_ = _create_neg(
                edge_index,
                edge_type,
                perturb_idx,
                self.ent_list_,
                self.ent_per_rel,
            )
        else:
            edge_index_, edge_type_ = edge_index, edge_type

        mask_x = torch.hstack([torch.tensor(ent_idx), edge_index_[1]])
        mask_rel = torch.hstack([torch.tensor(0), edge_type_])

        # Node features
        x = torch.ones(mask_x.size(0) + 1, self.x_total.size(1))
        x[1:, :] = self.x_total[mask_x]

        # Edge features
        edge_attr = torch.ones(mask_rel.size(0) + 1, self.x_total.size(1))
        edge_attr[1:, :] = self.edge_attr_total[mask_rel]

        # Masks for transformers
        pad_size = self.max_rels - x.size(0) + 2
        pad_mask = torch.zeros((self.max_rels + 2,), dtype=bool)

        if pad_size > 0:
            pad_mask[-pad_size:] = True

        pad_emb = -1 * torch.ones((pad_size, x.size(1)))
        x = torch.vstack((x, pad_emb))
        edge_attr = torch.vstack((edge_attr, pad_emb))

        # Target value
        if perturb_type == "neg":
            y = torch.tensor([0])
        else:
            y = torch.tensor([1])

        return x, edge_attr, pad_mask, y

    def _collate(self, sample):
        x = torch.stack([x for (x, _, _, _) in sample], dim=0)
        edge_attr = torch.stack([edge_attr for (_, edge_attr, _, _) in sample], dim=0)
        mask = torch.stack([mask for (_, _, mask, _) in sample], dim=0)
        y = torch.stack([y for (_, _, _, y) in sample], dim=1).reshape(
            1, self.num_sample
        )
        y = y[:, 1:]

        return x, edge_attr, mask, y


class CARTEKGIndexIterator:

    def __init__(
        self,
        data_kg_dir: str,
        num_rel: int = 4,
        num_batch: int = 2,
    ):
        super(CARTEKGIndexIterator, self).__init__()

        self.data_kg_dir = data_kg_dir
        self.num_rel = num_rel
        self.num_batch = num_batch

        self._set_idx_iterator()

    def _set_idx_iterator(self):

        edge_index = torch.load(
            f"{self.data_kg_dir}/edge_index.pt",
            weights_only=True,
            mmap=True,
        )
        edge_type = torch.load(
            f"{self.data_kg_dir}/edge_type.pt",
            weights_only=True,
            mmap=True,
        )

        # Indices and weights for sampling
        count_index = torch.vstack([edge_index[0], edge_type])
        count_index = _remove_duplicates(count_index)
        count = count_index[0].bincount()
        self.idx_keep = (count > self.num_rel - 1).nonzero().squeeze()

        count = count.to(torch.float) - 1
        self.count_head = count[self.idx_keep]
        self.count_head_ = count[self.idx_keep].clone()

    def sample_index(self):

        idx_count = torch.multinomial(self.count_head_, self.num_batch)
        idx_sample = self.idx_keep[idx_count]
        self.count_head_[idx_count] -= 1
        if self.count_head_.sum().item() < self.num_batch:
            self.count_head_ = self.count_head.clone()

        return idx_sample.tolist()


class KGDataset(Dataset):
    """PyTorch Dataset used for dataloader."""

    def __init__(self, num_steps, kg_preprocessor, idx_iterator):

        self.num_steps = num_steps
        self.kg_preprocessor = kg_preprocessor
        self.idx_iterator = idx_iterator

    def __len__(self):
        return self.num_steps

    def __getitems__(self, idx):
        idx_sample = self.idx_iterator.sample_index()
        return self.kg_preprocessor(idx_sample)


def kg_batch_collate(batch):
    num_batch = len(batch)
    num_sample = batch[0][0].size(0)
    num_rel = batch[0][0].size(1)
    num_dim = batch[0][0].size(2)

    x = torch.stack([item[0] for item in batch]).reshape(
        num_batch * num_sample, num_rel, num_dim
    )
    edge_attr = torch.stack([item[1] for item in batch]).reshape(
        num_batch * num_sample, num_rel, num_dim
    )
    mask = torch.stack([item[2] for item in batch]).reshape(
        num_batch * num_sample, num_rel
    )
    y = torch.stack([item[3] for item in batch]).reshape(num_batch, num_sample - 1)

    return x, edge_attr, mask, y


### Used for multi-hops
# def _extract_edge_indices(center_indices, slice_idx):
#     edge_indices = torch.cat(
#         [torch.arange(slice_idx[x], slice_idx[x + 1]) for x in center_indices], dim=0
#     )
#     return edge_indices

# if len([ent_idx]) == 1:
#     center_indices = [ent_idx]
# else:
#     center_indices = None # Need to fix later for multiple hops

# edge_indices = _extract_edge_indices(center_indices, self.slice_idx)
# edge_index = self.edge_index[:, edge_indices].clone()
# edge_type = self.edge_type[edge_indices].clone()
