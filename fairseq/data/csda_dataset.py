# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)

def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if (
            alignment[:, 0].max().item() >= src_len - 1
            or alignment[:, 1].max().item() >= tgt_len - 1
        ):
            logger.warning("alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True
        )
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1.0 / align_weights.float()

    id = torch.LongTensor([s["id"] for s in samples])
    cxt_tokens = merge(
        "context",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["context"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    cxt_lengths = torch.LongTensor(
        [s["context"].ne(pad_idx).long().sum() for s in samples]
    )
    cxt_lengths, sort_order = cxt_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    cxt_tokens = cxt_tokens.index_select(0, sort_order)

    z_tokens = merge(
        "latent",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["latent"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    z_tokens = z_tokens.index_select(0, sort_order)
    z_lengths = torch.LongTensor(
        [s["latent"].ne(pad_idx).long().sum() for s in samples]
    ).index_select(0, sort_order)

    prev_output_tokens = None
    if samples[0].get("target", None) is not None:
        res_target = merge(
            "target",
            left_pad=left_pad_target,
            pad_to_length=pad_to_length["target"]
            if pad_to_length is not None
            else None,
        )
        res_target = res_target.index_select(0, sort_order)
        res_lengths = torch.LongTensor(
            [s["target"].ne(pad_idx).long().sum() for s in samples]
        ).index_select(0, sort_order)


        ntokens = res_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
        elif input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target",
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
                pad_to_length=pad_to_length["target"]
                if pad_to_length is not None
                else None,
            )
    else:
        #  don't step in here
        assert 1 == 0
        ntokens = h_src_lengths.sum().item()



    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": cxt_tokens,
            "src_lengths": cxt_lengths,
            "z_tokens": z_tokens,
            "z_lengths": z_lengths,
        },
        "target": res_target,
    }
    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens.index_select(
            0, sort_order
        )

    if samples[0].get("alignment", None) is not None: # this branch do not go in
        assert 1 == 0
        bsz, tgt_sz = batch["target"].shape
        h_src_sz = batch["net_input"]["h_src_tokens"].shape[1]
        f_src_sz = batch["net_input"]["f_src_tokens"].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += torch.arange(len(sort_order), dtype=torch.long) * tgt_sz
        if left_pad_history:
            offsets[:, 0] += h_src_sz - h_src_lengths
        # not sure
        # if left_pad_future:
        #     offsets[:, 0] += f_src_sz - f_src_lengths
        if left_pad_target:
            offsets[:, 1] += tgt_sz - tgt_lengths

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(
                sort_order, offsets, src_lengths, tgt_lengths
            )
            for alignment in [samples[align_idx]["alignment"].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch["alignments"] = alignments
            batch["align_weights"] = align_weights

    if samples[0].get("constraints", None) is not None:
        # Collate the packed constraints across the samples, padding to
        # the length of the longest sample.
        lens = [sample.get("constraints").size(0) for sample in samples]
        max_len = max(lens)
        constraints = torch.zeros((len(samples), max(lens))).long()
        for i, sample in enumerate(samples):
            constraints[i, 0 : lens[i]] = samples[i].get("constraints")
        batch["constraints"] = constraints

    return batch

class CSDAPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        cxt,
        cxt_sizes,
        cxt_dict,
        z,
        z_sizes,
        res,
        res_sizes,
        res_dict,
        left_pad_source=False,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        cxt_lang_id=None,
        z_lang_id=None,
        res_lang_id=None,
        pad_to_multiple=1,
    ):

        assert len(cxt) == len(res), "Context and response must contain the same number of examples"
        assert len(z) == len(res), "latent and response must contain the same number of examples"

        self.cxt = cxt
        self.z = z
        self.res = res
        self.cxt_sizes = np.array(cxt_sizes)
        self.z_sizes = np.array(z_sizes)
        self.res_sizes = np.array(res_sizes)

        assert self.res_sizes is not None and self.cxt_sizes is not None and self.z_sizes is not None 
        
        self.sizes = (
            np.vstack((self.cxt_sizes, self.z_sizes, self.res_sizes)).T
        )
        self.cxt_dict = cxt_dict
        self.res_dict = res_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else cxt_dict.eos()
        self.cxt_lang_id = cxt_lang_id
        self.z_lang_id = z_lang_id
        self.res_lang_id = res_lang_id

        if num_buckets > 0:
            print("i dont know what does this branch do")
            assert num_buckets != 0

            from fairseq.data import BucketPadLengthDataset

            self.cxt = BucketPadLengthDataset(
                self.cxt,
                sizes=self.cxt_sizes,
                num_buckets=num_buckets,
                pad_idx=self.cxt_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.cxt_sizes = self.cxt.sizes

            self.z = BucketPadLengthDataset(
                self.z,
                sizes=self.z_sizes,
                num_buckets=num_buckets,
                pad_idx=self.cxt_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.z_sizes = self.z.sizes

            logger.info("bucketing context lengths: {}".format(list(self.cxt.buckets)))
            logger.info("bucketing latent lengths: {}".format(list(self.z.buckets)))
            
            self.res = BucketPadLengthDataset(
                self.res,
                sizes=self.res_sizes,
                num_buckets=num_buckets,
                pad_idx=self.res_dict.pad(),
                left_pad=self.left_pad_target,
            )
            self.res_sizes = self.res.sizes
            logger.info(
                "bucketing response lengths: {}".format(list(self.res.buckets))
            )

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.cxt)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        cxt_item = self.cxt[index]
        z_item = self.z[index]
        res_item = self.res[index]


        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.res_dict.eos()
            if self.res[index][-1] != eos:
                res_item = torch.cat([self.res[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.cxt_dict.bos()
            if self.h_tgt[index][0] != bos:
                res_item = torch.cat([torch.LongTensor([bos]), self.res[index]])

            bos = self.cxt_dict.bos()
            if self.cxt[index][0] != bos:
                cxt_item = torch.cat([torch.LongTensor([bos]), self.cxt[index]])
            if self.z[index][0] != bos:
                z_item = torch.cat([torch.LongTensor([bos]), self.z[index]])

        if self.remove_eos_from_source:
            eos = self.cxt_dict.eos()
            if self.cxt[index][-1] == eos:
                cxt_item = self.cxt[index][:-1]
            if self.z[index][-1] == eos:
                z_item = self.z[index][:-1]

        example = {
            "id": index,
            "context": cxt_item,
            "latent": z_item,
            "target": res_item,
        }
        if self.align_dataset is not None:
            example["alignment"] = self.align_dataset[index]
        if self.constraints is not None:
            example["constraints"] = self.constraints[index]
        return example

    def __len__(self):
        return len(self.cxt)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        result = collate(
            samples,
            pad_idx=self.cxt_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.cxt_lang_id is not None or self.z_lang_id is not None or self.res_lang_id is not None:
            cxt_tokens = res["net_input"]["cxt_tokens"]
            z_tokens = res["net_input"]["z_tokens"]
            assert cxt_tokens.size(0) == z_tokens.size(0)
            bsz = cxt_tokens.size(0)
            if self.cxt_lang_id is not None:
                res["net_input"]["cxt_lang_id"] = (
                    torch.LongTensor([[self.cxt_lang_id]]).expand(bsz, 1).to(cxt_tokens)
                )
            if self.z_lang_id is not None:
                res["net_input"]["z_lang_id"] = (
                    torch.LongTensor([[self.z_lang_id]]).expand(bsz, 1).to(z_tokens)
                )
            if self.res_lang_id is not None:
                res["res_lang_id"] = (
                    torch.LongTensor([[self.res_lang_id]]).expand(bsz, 1).to(res_tokens)
                )

        return result

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            max(self.cxt_sizes[index],self.z_sizes[index]),
            max(self.cxt_sizes[index],self.res_sizes[index])
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.cxt_sizes[index],
            self.z_sizes[index],
            self.res_sizes[index],
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then future length, then history length
            indices = indices[np.argsort(self.res_sizes[indices], kind="mergesort")]
            indices = indices[np.argsort(self.z_sizes[indices], kind="mergesort")]
            indices = indices[np.argsort(self.cxt_sizes[indices], kind="mergesort")]
            return indices
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.cxt, "supports_prefetch", False) and \
               getattr(self.z, "supports_prefetch", False) and \
               getattr(self.res, "supports_prefetch", False)

    def prefetch(self, indices):
        assert 1 == 0 # i dont know what does this function do
        self.h_src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """

        return data_utils.filter_csda_dataset_indices_by_size(
            self.cxt_sizes,
            self.z_sizes,
            self.res_sizes,
            indices,
            max_sizes,
        )