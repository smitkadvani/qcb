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
Derived from akita_scd.py

Compute insertion scores for paired motif insertions from a tsv file with chrom,start,end,strand for positions of motifs used as the core & flanking sequence. 

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

    ## insertion-specific options
    parser.add_option(
        "--background-file",
        dest="background_file",
        default=None,
        help="file with insertion seqs in fasta format",
    )
    parser.add_option(
        "--spacer-bp",
        dest="spacer_bp",
        default=0,
        type="int",
        help="Specify spacing between insertions",
    )
    parser.add_option(
        "--num-inserts",
        dest="num_inserts",
        default=6,
        type="int",
        help="Specify number of insertions",
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

    background_seqs = []
    with open(options.background_file, "r") as f:
        for line in f.readlines():
            if ">" in line:
                continue
            background_seqs.append(dna_io.dna_1hot(line.strip()))

    df_site_pairs  = seq_coords_df
    core_len = (df_site_pairs["end_core"] - df_site_pairs["start_core"]).unique()
    flank_len = (df_site_pairs["end_flank"] - df_site_pairs["start_flank"]).unique()
    if not (len(core_len) == 1) and (len(flank_len) == 1) and (core_len==flank_len):
        raise ValueError("current code works with fixed and equal size for core & flank motifs before padding.")
    
    spacer_bp = options.spacer_bp
    num_inserts = options.num_inserts

 
    def seqs_gen(df_site_pairs, genome_open, spacer_bp, num_inserts):
        # iterate over paired motif positions, insert the core into the flanking sequence
        # then insert num_insert copies of this cassette into the background sequence
        for s in df_site_pairs.itertuples():
          for background_seq in background_seqs:
            seq_1hot = background_seq.copy()
            seq_1hot_core = dna_io.dna_1hot(genome_open.fetch(
                                s.chrom_core, s.start_core-s.pad_core, s.end_core+s.pad_core).upper())
            if s.strand_core == '-': seq_1hot_core = dna_io.hot1_rc(seq_1hot_core)
            seq_1hot_flank = dna_io.dna_1hot(genome_open.fetch(
                                s.chrom_flank, s.start_flank-s.pad_flank, s.end_flank+s.pad_flank).upper())
            if s.strand_flank == '-': seq_1hot_flank = dna_io.hot1_rc(seq_1hot_flank)
            seq_1hot_insert = seq_1hot_flank.copy()
            seq_1hot_insert[s.pad_flank - s.pad_core: (-s.pad_flank+s.pad_core)] = seq_1hot_core
            insert_length = len(seq_1hot_insert)
            multi_insert_length = num_inserts * (insert_length+spacer_bp)
            offsets = [ (seq_length//2 - multi_insert_length//2 + i * (insert_length+spacer_bp)) for i in range(num_inserts)]
            for offset in offsets:
               seq_1hot[offset:offset+insert_length] = seq_1hot_insert.copy()
            yield seq_1hot


    #################################################################
    # setup output

    scd_out = initialize_output_h5(
        options.out_dir,
        options.scd_stats,
        seq_coords_df,
        background_seqs,
        target_ids,
        target_labels,
    )

    print("initialized")

    #################################################################
    # predict SNP scores, write output

    write_thread = None

    # initialize predictions stream
    preds_stream = stream.PredStreamGen(
        seqnn_model, seqs_gen(seq_coords_df, genome_open, spacer_bp, num_inserts), batch_size
    )

    # predictions index
    pi = 0
    for si in range(num_motifs):
        # get predictions
        for bi in range(len(background_seqs)):
            preds = preds_stream[pi]
            pi += 1
            # process SNP
            write_snp(
                preds,
                scd_out,
                si,
                bi,
                pi,
                seqnn_model.diagonal_offset,
                options.scd_stats,
                plot_dir,
                options.plot_lim_min,
                options.plot_freq,
            )

    """Write SNP predictions to HDF."""

    genome_open.close()
    scd_out.close()


def initialize_output_h5(
    out_dir, scd_stats, seq_coords_df, background_seqs, target_ids, target_labels
):
    """Initialize an output HDF5 file for SCD stats."""

    num_targets = len(target_ids)
    num_motifs = len(seq_coords_df)
    num_background_seqs = len(background_seqs)

    scd_out = h5py.File("%s/scd.h5" % out_dir, "w")
    seq_coords_df_dtypes = seq_coords_df.dtypes

    for key in seq_coords_df:
        if seq_coords_df_dtypes[key] is np.dtype("O"):
            scd_out.create_dataset(key, data=seq_coords_df[key].values.astype("S"))
        else:
            scd_out.create_dataset(key, data=seq_coords_df[key])

    # initialize scd stats
    for scd_stat in scd_stats:
        if scd_stat in seq_coords_df.keys():
            raise KeyError("check input tsv for clashing score name")
        scd_out.create_dataset(
            scd_stat,
            shape=(num_background_seqs, num_motifs, num_targets),
            dtype="float16",
            compression=None,
        )

    return scd_out


def write_snp(
    ref_preds,
    scd_out,
    si,
    bi,
    pi,
    diagonal_offset,
    scd_stats=["SCD"],
    plot_dir=None,
    plot_lim_min=0.1,
    plot_freq=100,
    
):
    """Write SNP predictions to HDF."""

    # increase dtype
    ref_preds = ref_preds.astype("float32")

    # compare reference to alternative via mean subtraction
    if "SCD" in scd_stats:
        # sum of squared diffs
        sd2_preds = np.sqrt((ref_preds**2).sum(axis=0))
        scd_out["SCD"][bi, si, :] = sd2_preds.astype("float16")

    if np.any((["INS" in i for i in scd_stats])):
        ref_map = ut_dense(ref_preds, diagonal_offset)
        for stat in scd_stats:
            if "INS" in stat:
                insul_window = int(stat.split("-")[1])
                scd_out[stat][bi, si, :] = insul_diamonds_scores(
                    ref_map, window=insul_window
                )

    if (plot_dir is not None) and (np.mod(pi, plot_freq) == 0):
        print("plotting ", si)
        # convert back to dense
        ref_map = ut_dense(ref_preds, diagonal_offset)
        _, axs = plt.subplots(1, ref_preds.shape[-1], figsize=(24, 4))
        for ti in range(ref_preds.shape[-1]):
            ref_map_ti = ref_map[..., ti]
            # TEMP: reduce resolution
            ref_map_ti = block_reduce(ref_map_ti, (2, 2), np.mean)
            vmin = min(ref_map_ti.min(), ref_map_ti.min())
            vmax = max(ref_map_ti.max(), ref_map_ti.max())
            vmin = min(-plot_lim_min, vmin)
            vmax = max(plot_lim_min, vmax)
            sns.heatmap(
                ref_map_ti,
                ax=axs[ti],
                center=0,
                vmin=vmin,
                vmax=vmax,
                cmap="RdBu_r",
                xticklabels=False,
                yticklabels=False,
            )

        plt.tight_layout()
        plt.savefig("%s/s%d.pdf" % (plot_dir, si))
        plt.close()


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


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
