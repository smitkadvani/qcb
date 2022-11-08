#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function

from optparse import OptionParser
import json
import os
import pdb
import pickle
import random
import sys
import time

import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pysam
from skimage.measure import block_reduce
import seaborn as sns

sns.set(style="ticks", font_scale=1.3)

import tensorflow as tf

if tf.__version__[0] == "1":
    tf.compat.v1.enable_eager_execution()
gpus = tf.config.experimental.list_physical_devices("GPU")
# for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

from basenji import seqnn
from basenji import stream
from basenji import dna_io

from akita_utils import ut_dense

"""
akita_motif_scd.py

Compute Squared Contact Difference (SCD) scores for motifs in TSV motif file.

"""

################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <tsv_file>"
    parser = OptionParser(usage)
    parser.add_option(
        "-f",
        dest="genome_fasta",
        default=None,
        help="Genome FASTA for sequences [Default: %default]",
    )
    parser.add_option(
        "-l",
        dest="plot_lim_min",
        default=0.1,
        type="float",
        help="Heatmap plot limit [Default: %default]",
    )
    parser.add_option(
        "--plot-freq",
        dest="plot_freq",
        default=100,
        type="int",
        help="Heatmap plot freq [Default: %default]",
    )
    parser.add_option(
        "-m",
        dest="plot_map",
        default=False,
        action="store_true",
        help="Plot contact map for each allele [Default: %default]",
    )
    parser.add_option(
        "-o",
        dest="out_dir",
        default="scd",
        help="Output directory for tables and plots [Default: %default]",
    )
    parser.add_option(
        "-p",
        dest="processes",
        default=None,
        type="int",
        help="Number of processes, passed by multi script",
    )
    parser.add_option(
        "--rc",
        dest="rc",
        default=False,
        action="store_true",
        help="Average forward and reverse complement predictions [Default: %default]",
    )
    parser.add_option(
        "--shifts",
        dest="shifts",
        default="0",
        type="str",
        help="Ensemble prediction shifts [Default: %default]",
    )
    parser.add_option(
        "--stats",
        dest="scd_stats",
        default="SCD",
        help="Comma-separated list of stats to save. [Default: %default]",
    )
    parser.add_option(
        "-t",
        dest="targets_file",
        default=None,
        type="str",
        help="File specifying target indexes and labels in table format",
    )
    parser.add_option(
        "--batch-size",
        dest="batch_size",
        default=None,
        type="int",
        help="Specify batch size",
    )
    parser.add_option(
        "--head-index",
        dest="head_index",
        default=0,
        type="int",
        help="Specify head index (0=human 1=mus) ",
    )

    parser.add_option(
        "--mut-method",
        dest="mutation_method",
        default="mask",
        type="str",
        help="specify mutation method",
    )

    parser.add_option(
        "--use-span",
        dest="use_span",
        default=False,
        action="store_true",
        help="specify if using spans",
    )

    parser.add_option(
        "--motif-width", dest="motif_width", default=18, type="int", help="motif width"
    )
    (options, args) = parser.parse_args()

    if len(args) == 3:
        # single worker
        params_file = args[0]
        model_file = args[1]
        motif_file = args[2]

    elif len(args) == 5:
        # multi worker
        options_pkl_file = args[0]
        params_file = args[1]
        model_file = args[2]
        motif_file = args[3]
        worker_index = int(args[4])

        # load options
        options_pkl = open(options_pkl_file, "rb")
        options = pickle.load(options_pkl)
        options_pkl.close()

        # update output directory
        options.out_dir = "%s/job%d" % (options.out_dir, worker_index)

    else:
        parser.error("Must provide parameters and model files and TSV file")

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)
    if options.plot_map:
        plot_dir = options.out_dir
    else:
        plot_dir = None

    options.shifts = [int(shift) for shift in options.shifts.split(",")]
    options.scd_stats = options.scd_stats.split(",")

    random.seed(44)

    #################################################################
    # read parameters and targets

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_train = params["train"]
    params_model = params["model"]

    if options.batch_size is None:
        batch_size = params_train["batch_size"]
    else:
        batch_size = options.batch_size
    print(batch_size)
    mutation_method = options.mutation_method
    if not mutation_method in ["mask", "permute"]:
        raise ValueError("undefined mutation method:", mutation_method)
    motif_width = options.motif_width
    use_span = options.use_span
    if options.use_span:
        print("using SPANS")

    if options.targets_file is not None:
        targets_df = pd.read_csv(options.targets_file, sep="\t", index_col=0)
        target_ids = targets_df.identifier
        target_labels = targets_df.description

    #################################################################
    # setup model

    # load model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, head_i=options.head_index)
    seqnn_model.build_ensemble(options.rc, options.shifts)
    seq_length = int(params_model["seq_length"])

    # dummy target info
    if options.targets_file is None:
        num_targets = seqnn_model.num_targets()
        target_ids = ["t%d" % ti for ti in range(num_targets)]
        target_labels = [""] * len(target_ids)

    #################################################################
    # load motifs

    # filter for worker motifs
    if options.processes is not None:
        # determine boundaries from motif file
        seq_coords_full = pd.read_csv(motif_file, sep="\t")

        num_motifs_total = len(seq_coords_full)
        worker_bounds = np.linspace(
            0, num_motifs_total, options.processes + 1, dtype="int"
        )

        seq_coords_df = seq_coords_full.loc[
            worker_bounds[worker_index] : worker_bounds[worker_index + 1], :
        ]

    else:
        # read motif positions from csv
        seq_coords_df = pd.read_csv(motif_file, sep="\t")

    num_motifs = len(seq_coords_df)

    # open genome FASTA
    genome_open = pysam.Fastafile(options.genome_fasta)

    # define sequence generator
    def mask_central_seq(seq_1hot, motif_width=20):
        seq_1hot_perm = seq_1hot.copy()
        mask_inds = np.arange(
            seq_length // 2 - motif_width // 2, seq_length // 2 + motif_width // 2
        )
        seq_1hot_perm[mask_inds, :] = 0
        return seq_1hot_perm

    def permute_central_seq(seq_1hot, motif_width=20):
        seq_1hot_perm = seq_1hot.copy()
        central_inds = np.arange(
            seq_length // 2 - motif_width // 2, seq_length // 2 + motif_width // 2
        )
        mask_inds = np.random.permutation(central_inds)
        seq_1hot_perm[mask_inds, :] = seq_1hot[central_inds, :].copy()
        return seq_1hot_perm

    def mask_spans(seq_1hot, spans):
        seq_1hot_perm = seq_1hot.copy()
        for s in spans:
            seq_1hot_perm[s[0] : s[1], :] = 0
        return seq_1hot_perm

    def permute_spans(seq_1hot, spans):
        seq_1hot_perm = seq_1hot.copy()
        spans_flat = np.array([]).astype(int)
        for s in spans:
            spans_flat = np.hstack((spans_flat, np.arange(s[0], s[1])))
        spans_permuted = np.random.permutation(spans_flat)
        seq_1hot_perm[spans_permuted, :] = seq_1hot[spans_flat, :].copy()
        return seq_1hot_perm

    def split_span(span_string):
        spans = []
        for j in span_string.split(","):
            spans.append([int(j.split("-")[0]), int(j.split("-")[1])])
        return spans

    def fetch_centered_padded_seq(chrom, start, end):
        mid = (start + end) // 2
        start_centered, end_centered = int(mid - seq_length // 2), int(
            mid + seq_length // 2
        )
        seq_dna = genome_open.fetch(chrom, start_centered, end_centered).upper()
        chromosome_length = genome_open.get_reference_length(chrom)
        pad_upstream = "N" * max(-start_centered, 0)
        pad_downstream = "N" * max(end_centered - chromosome_length, 0)
        return pad_upstream + seq_dna + pad_downstream

    def seqs_gen(motif_width):
        for s in seq_coords_df.itertuples():
            list_1hot = []
            seq_dna = fetch_centered_padded_seq(s.chrom, s.start, s.end)
            wt_1hot = dna_io.dna_1hot(seq_dna)
            list_1hot.append(wt_1hot)

            if use_span:
                spans = split_span(s.span)
                spans = np.array(spans) - start
                if mutation_method == "mask":
                    list_1hot.append(mask_spans(wt_1hot, spans))
                elif mutation_method == "permute":
                    list_1hot.append(permute_spans(wt_1hot, spans))

            else:  ## just mask central motif
                if mutation_method == "mask":
                    list_1hot.append(mask_central_seq(wt_1hot, motif_width=motif_width))
                elif mutation_method == "permute":
                    list_1hot.append(
                        permute_central_seq(wt_1hot, motif_width=motif_width)
                    )

            for seq_1hot in list_1hot:
                yield seq_1hot

    #################################################################
    # setup output

    scd_out = initialize_output_h5(
        options.out_dir, options.scd_stats, seq_coords_df, target_ids, target_labels
    )

    print("initialized")

    #################################################################
    # predict SNP scores, write output

    write_thread = None

    # initialize predictions stream
    preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen(motif_width), batch_size)

    # predictions index
    pi = 0

    for si in range(num_motifs):
        # get predictions
        ref_preds = preds_stream[pi]
        pi += 1
        alt_preds = preds_stream[pi]
        pi += 1

        # process SNP
        write_snp(
            ref_preds,
            alt_preds,
            scd_out,
            si,
            seqnn_model.diagonal_offset,
            options.scd_stats,
            plot_dir,
            options.plot_lim_min,
            options.plot_freq,
        )

    """Write SNP predictions to HDF."""

    genome_open.close()
    scd_out.close()


def initialize_output_h5(out_dir, scd_stats, seq_coords_df, target_ids, target_labels):
    """Initialize an output HDF5 file for SCD stats."""

    num_targets = len(target_ids)
    num_motifs = len(seq_coords_df)

    scd_out = h5py.File("%s/scd.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes
    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            scd_out.create_dataset(key, data=seq_coords_df[key].values.astype("S"))
        else:
            scd_out.create_dataset(key, data=seq_coords_df[key])

    # initialize scd stats
    for scd_stat in scd_stats:
        if "INS" not in scd_stat:
            scd_out.create_dataset(
                scd_stat,
                shape=(num_motifs, num_targets),
                dtype="float16",
                compression=None,
            )
        else:
            scd_out.create_dataset(
                "ref_" + scd_stat,
                shape=(num_motifs, num_targets),
                dtype="float16",
                compression=None,
            )
            scd_out.create_dataset(
                "alt_" + scd_stat,
                shape=(num_motifs, num_targets),
                dtype="float16",
                compression=None,
            )

    return scd_out


def _insul_diamond_central(mat, window=10):
    """calculate insulation in a diamond around the central pixel"""
    N = mat.shape[0]
    if window > N // 2:
        raise ValueError("window cannot be larger than matrix")
    mid = N // 2
    lo = max(0, mid + 1 - window)
    hi = min(mid + window, N)
    score = np.nanmean(mat[lo : (mid + 1), mid:hi])
    return score


def insul_diamonds_scores(mats, window=10):
    num_targets = mats.shape[-1]
    scores = np.zeros((num_targets,))
    for ti in range(num_targets):
        scores[ti] = _insul_diamond_central(mats[:, :, ti], window=window)
    return scores


def write_snp(
    ref_preds,
    alt_preds,
    scd_out,
    si,
    diagonal_offset,
    scd_stats=["SSD", "SCD"],
    plot_dir=None,
    plot_lim_min=0.1,
    plot_freq=100,
):
    """Write SNP predictions to HDF."""

    # increase dtype
    ref_preds = ref_preds.astype("float32")
    alt_preds = alt_preds.astype("float32")

    # sum across length
    ref_preds_sum = ref_preds.sum(axis=0)
    alt_preds_sum = alt_preds.sum(axis=0)

    # compare reference to alternative via mean subtraction
    if "SCD" in scd_stats:
        # sum of squared diffs
        diff2_preds = (ref_preds - alt_preds) ** 2
        sd2_preds = np.sqrt(diff2_preds.sum(axis=0))
        scd_out["SCD"][si, :] = sd2_preds.astype("float16")

    if "SSD" in scd_stats:
        # sum of squared diffs
        ref_ss = (ref_preds**2).sum(axis=0)
        alt_ss = (alt_preds**2).sum(axis=0)
        s2d_preds = np.sqrt(alt_ss) - np.sqrt(ref_ss)
        scd_out["SSD"][si, :] = s2d_preds.astype("float16")

    if np.any((["INS" in i for i in scd_stats])):
        ref_map = ut_dense(ref_preds, diagonal_offset)
        alt_map = ut_dense(alt_preds, diagonal_offset)
        for stat in scd_stats:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])
                scd_out["ref_" + stat][si, :] = insul_diamonds_scores(
                    ref_map, window=insul_window
                )
                scd_out["alt_" + stat][si, :] = insul_diamonds_scores(
                    alt_map, window=insul_window
                )

    if (plot_dir is not None) and (np.mod(si, plot_freq) == 0):
        print("plotting ", si)
        # TEMP: average across targets
        ref_preds = ref_preds.mean(axis=-1, keepdims=True)
        alt_preds = alt_preds.mean(axis=-1, keepdims=True)

        # convert back to dense
        ref_map = ut_dense(ref_preds, diagonal_offset)
        alt_map = ut_dense(alt_preds, diagonal_offset)

        # with h5py.File('%s/s%d_maps.h5' % (plot_dir, si), 'w') as map_h5:
        #  map_h5.create_dataset('ref', data=ref_map, dtype='float16')
        #  map_h5.create_dataset('alt', data=alt_map, dtype='float16')

        for ti in range(1):  # ref_preds.shape[-1]):
            ref_map_ti = ref_map[..., ti]
            alt_map_ti = alt_map[..., ti]

            # TEMP: reduce resolution
            ref_map_ti = block_reduce(ref_map_ti, (2, 2), np.mean)
            alt_map_ti = block_reduce(alt_map_ti, (2, 2), np.mean)

            vmin = min(ref_map_ti.min(), ref_map_ti.min())
            vmax = max(alt_map_ti.max(), alt_map_ti.max())

            vmin = min(-plot_lim_min, vmin)
            vmax = max(plot_lim_min, vmax)

            _, (ax_ref, ax_alt, ax_diff) = plt.subplots(1, 3, figsize=(21, 6))
            sns.heatmap(
                ref_map_ti,
                ax=ax_ref,
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )
            sns.heatmap(
                alt_map_ti,
                ax=ax_alt,
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )
            sns.heatmap(
                alt_map_ti - ref_map_ti,
                ax=ax_diff,
                center=0,
                cmap="PRGn",
                xticklabels=False,
                yticklabels=False,
            )
            plt.tight_layout()
            plt.savefig("%s/s%d_t%d.pdf" % (plot_dir, si, ti))
            plt.close()


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
