# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from argparse import Namespace

import numpy as np
from fairseq import metrics, options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    CSDAPairDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
from fairseq.tasks import LegacyFairseqTask, register_task

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)

def load_dialog_pair_dataset(
        data_path,
        split,
        z,
        cxt,
        cxt_dict,
        res,
        res_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        append_source_id=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    cxt_datasets = []
    res_datasets = []
    z_datasets = []

    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")

        if split_exists(split_k, cxt, res, cxt, data_path):
            cxt_prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, cxt, res))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )
        cxt_dataset = data_utils.load_indexed_dataset(
            cxt_prefix + cxt, cxt_dict, dataset_impl
        )
        if truncate_source:
            cxt_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(cxt_dataset, cxt_dict.eos()),
                    max_source_positions - 1,
                ),
                cxt_dict.eos(),
            )
        cxt_datasets.append(cxt_dataset)

        if split_exists(split_k, z, res, z, data_path):
            z_prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, z, res))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )
        z_dataset = data_utils.load_indexed_dataset(
            z_prefix + z, cxt_dict, dataset_impl
        )
        if truncate_source:
            z_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(z_dataset, cxt_dict.eos()),
                    max_source_positions - 1,
                ),
                cxt_dict.eos(),
            )
        z_datasets.append(z_dataset)

        res_dataset = data_utils.load_indexed_dataset(
            cxt_prefix + res, res_dict, dataset_impl
        )
        if res_dataset is not None:
            res_datasets.append(res_dataset)



        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, cxt, res, len(cxt_datasets[-1])
            )
        )
        
        if not combine:
            break

    assert len(cxt_datasets) == 1 and len(res_datasets) == 1 and len(z_datasets) == 1

    cxt_dataset = cxt_datasets[0]
    res_dataset = res_datasets[0]
    z_dataset = z_datasets[0]

    if prepend_bos:
        assert hasattr(cxt_dict, "bos_index") and hasattr(res_dict, "bos_index") 

        cxt_dataset = PrependTokenDataset(cxt_dataset, cxt_dict.bos())
        z_dataset = PrependTokenDataset(z_dataset, cxt_dict.bos())
        res_dataset = PrependTokenDataset(res_dataset, res_dict.bos())

    eos = None
    if append_source_id:
        cxt_dataset = AppendTokenDataset(
            cxt_dataset, cxt_dict.index("[{}]".format(cxt))
        )
        z_dataset = AppendTokenDataset(
            z_dataset, cxt_dict.index("[{}]".format(z))
        )
        res_dataset = AppendTokenDataset(
            res_dataset, res_dict.index("[{}]".format(res))
        )

        eos = res_dict.index("[{}]".format(res))

    align_dataset = None
    assert load_alignments is False
    # if load_alignments:
    #     align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
    #     if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
    #         align_dataset = data_utils.load_indexed_dataset(
    #             align_path, None, dataset_impl
    #         )

    res_dataset_sizes = res_dataset.sizes if res_dataset is not None else None

    return CSDAPairDataset(
        cxt_dataset,
        cxt_dataset.sizes,
        cxt_dict,
        z_dataset,
        z_dataset.sizes,
        res_dataset,
        res_dataset_sizes,
        res_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("csda")
class CSDATask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner; \
                            however, valid and test data are always in the first directory to \
                            avoid the need for repeating them in all directories')
        parser.add_argument('--z', default='z', metavar='SRC',
                            help='x1 x2 x3 r1 ra')
        parser.add_argument('--cxt', default='cxt', metavar='SRC',
                            help='x1 x2 x3')
        parser.add_argument('--res', default='res', metavar='TARGET',
                            help='r1')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the future target on the left')
        parser.add_argument('--set-history', default='True', type=str, metavar='BOOL',
                            help='')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--truncate-source', action='store_true', default=False,
                            help='truncate source to max-source-positions')
        parser.add_argument('--num-batch-buckets', default=0, type=int, metavar='N',
                            help='if >0, then bucket source and target lengths into N '
                                 'buckets and pad accordingly; this is useful on TPUs '
                                 'to minimize the number of compilations')

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        # fmt: on

    def __init__(self, args, cxt_dict, res_dict):
        super().__init__(args)
        self.cxt_dict = cxt_dict
        self.res_dict = res_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)
        # data : str "data-bin"
        paths = utils.split_paths(args.data)
        assert len(paths) > 0

        if args.cxt is None or args.z is None or args.res is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        # load dictionaries
        # z_dict_path = os.path.join(paths[0], "dict.{}.txt".format(args.z))
        # z_dict = cls.load_dictionary(z_dict_path)
        # logger.info("load [{}] dictionary from {}".format(args.z, z_dict_path))

        cxt_dict_path = os.path.join(paths[0], "dict.{}.txt".format(args.cxt))
        cxt_dict = cls.load_dictionary(cxt_dict_path)
        logger.info("load [{}] dictionary from {}".format(args.cxt, cxt_dict_path))

        res_dict_path = os.path.join(paths[0], "dict.{}.txt".format(args.res))
        res_dict = cls.load_dictionary(res_dict_path)
        logger.info("load [{}] dictionary from {}".format(args.res, res_dict_path))

        # assert z_dict.pad() == cxt_dict.pad()
        # assert z_dict.eos() == cxt_dict.eos()
        # assert z_dict.unk() == cxt_dict.unk()
        assert cxt_dict.pad() == res_dict.pad()
        assert cxt_dict.eos() == res_dict.eos()
        assert cxt_dict.unk() == res_dict.unk()

        # logger.info("[{}] dictionary: {} types".format(args.z, len(z_dict)))
        logger.info("[{}] dictionary: {} types".format(args.cxt, len(cxt_dict)))
        logger.info("[{}] dictionary: {} types".format(args.res, len(res_dict)))

        return cls(args, cxt_dict, res_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        if split != getattr(self.args, "train_subset", None):
            # if not training data set, use the first shard for valid and test
            paths = paths[:1]
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        z, cxt, res = self.args.z, self.args.cxt, self.args.res

        self.datasets[split] = load_dialog_pair_dataset(
            data_path,
            split,
            z,
            cxt,
            self.cxt_dict,
            res,
            self.res_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            truncate_source=self.args.truncate_source,
            num_buckets=self.args.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.args.required_seq_len_multiple,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        return LanguagePairDataset(
            src_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )

    def build_model(self, args):
        from fairseq import models, quantization_utils
        model = super().build_model(args)
        if getattr(args, "eval_bleu", False):
            assert getattr(args, "eval_bleu_detok", None) is not None, (
                "--eval-bleu-detok is required if using --eval-bleu; "
                "try --eval-bleu-detok=moses (or --eval-bleu-detok=space "
                "to disable detokenization, e.g., when using sentencepiece)"
            )
            detok_args = json.loads(getattr(args, "eval_bleu_detok_args", "{}") or "{}")
            self.tokenizer = encoders.build_tokenizer(
                Namespace(
                    tokenizer=getattr(args, "eval_bleu_detok", None), **detok_args
                )
            )

            gen_args = json.loads(getattr(args, "eval_bleu_args", "{}") or "{}")
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                counts = [i.cpu() for i in counts]
                totals = [i.cpu() for i in totals]
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=meters["_bleu_sys_len"].sum,
                        ref_len=meters["_bleu_ref_len"].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.cxt_dict

    # @property
    # def z_dictionary(self):
    #     """Return the source :class:`~fairseq.data.Dictionary`."""
    #     return self.z_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.res_dict

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.res_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.res_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])
