#!/usr/bin/env python

"""
This script generates a tsv of boundary CTCFs that can be used as input to akita_motif_scd.py.

This requires the following inputs:

- CTCF motif positions as a jaspar tsv
- insulation profile with called boundaries as a tsv, currently at 10kb resolution
- chromosome lengths as a chrom.sizes file
- model sequence length as json

First, boundaries are filtered:
- boundaries on non-autosomal chromosomes are dropped
- boundaries closer than model seq_length //2 to the start or end of chromosomes are dropped

Second, CTCF motifs are intersected with boundaries

Third, a dataFrame of candidate mutations is generated. For each boundary this includes:
- the entire boundary
- all CTCF sites overlapping that boundary 
- each individual CTCF site overlapping that boundary

Disjoint spans are indicated as a comma-separated list of intervals, e.g. "5-10,15-20".

Output for default arguments saved in /project/fudenber_735/tensorflow_models/akita/v2/analysis/ jun17, 2022.

"""
from optparse import OptionParser
import json
import bioframe
import numpy as np
import pandas as pd
import os
from akita_utils import filter_by_chrmlen


################################################################################
# main
################################################################################
def main():
    usage = "usage: %prog [options] <params_file> <model_file> <vcf_file>"
    parser = OptionParser(usage)

    parser.add_option(
        "--params-file",
        dest="params_file",
        default="/project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--jaspar-file",
        dest="jaspar_file",
        default="/project/fudenber_735/motifs/mm10/jaspar/MA0139.1.tsv.gz",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--chrom-sizes-file",
        dest="chrom_sizes_file",
        default="/project/fudenber_735/genomes/mm10/mm10.chrom.sizes.reduced",
        help=" [Default: %default]",
    )

    parser.add_option(
        "--boundary-file",
        dest="boundary_file",
        default="/project/fudenber_735/GEO/bonev_2017_GSE96107/distiller-0.3.1_mm10/results/coolers/features/bonev2017.HiC_ES.mm10.mapq_30.1000.window_200000.insulation",
        help=" [Default: %default]",
    )
    parser.add_option(
        "--boundary-strength-thresh",
        dest="boundary_strength_thresh",
        default=0.25,
        type=float,
        help="threshold on boundary strengths [Default: %default]",
    )
    parser.add_option(
        "--boundary-insulation-thresh",
        dest="boundary_insulation_thresh",
        default=0.00,
        type=float,
        help="threshold on boundary insulation score [Default: %default]",
    )

    parser.add_option(
        "--boundary-output-tsv",
        dest="boundary_output_tsv",
        default="boundaries.motifs.ctcf.mm10.tsv",
        type="str",
        help="Output filename for boundary tsv [Default: %default]",
    )

    (options, args) = parser.parse_args()

    if os.path.exists(options.boundary_output_tsv) is True:
        raise ValueError("boundary file already exists!")

    # get model seq_length
    with open(options.params_file) as params_open:
        params_model = json.load(params_open)["model"]
    seq_length = params_model["seq_length"]
    if seq_length != 1310720:
        raise Warning("potential incompatibilities with AkitaV2 seq_length")

    # load CTCF motifs
    ctcf_motifs = bioframe.read_table(options.jaspar_file, schema="jaspar", skiprows=1)
    ctcf_motifs = ctcf_motifs[~ctcf_motifs.chrom.isin(["chrX", "chrY", "chrM"])]
    ctcf_motifs.reset_index(drop=True, inplace=True)

    # load boundaries and use standard filters for their strength
    boundaries = pd.read_csv(
        options.boundary_file,
        sep="\t",
    )
    window_size = options.boundary_file.split("window_")[1].split(".")[0]
    boundary_key, insulation_key = (
        f"boundary_strength_{window_size}",
        f"log2_insulation_score_{window_size}",
    )
    boundaries = boundaries.iloc[
        (boundaries[boundary_key].values > options.boundary_strength_thresh)
        * (boundaries[insulation_key].values < options.boundary_insulation_thresh)
    ]

    boundaries = boundaries[~boundaries.chrom.isin(["chrX", "chrY", "chrM"])]
    boundaries = filter_by_chrmlen(
        boundaries,
        options.chrom_sizes_file,
        seq_length,
    )
    boundaries.reset_index(drop=True, inplace=True)

    # intersect CTCFs and boundaries, discard extraneous information
    df_overlap = bioframe.overlap(
        boundaries, ctcf_motifs, suffixes=("", "_2"), return_index=True
    )
    df_overlap["span"] = (
        df_overlap["start_2"].astype(str) + "-" + df_overlap["end_2"].astype(str)
    )
    df_keys = [
        "chrom",
        "start",
        "end",
        "span",
        "score_2",
        "strand_2",
        "index_2",
        insulation_key,
        boundary_key,
        "index",
    ]
    df_overlap = df_overlap[df_keys]

    df_out = _generate_boundary_mutation_df(df_overlap)

    # reformat and save the resulting dataFrame
    df_out[[insulation_key, boundary_key]] = df_out[
        [insulation_key, boundary_key]
    ].astype("float16")
    df_out.rename(
        columns={"index": "boundary_index", "index_2": "num_ctcf"}, inplace=True
    )
    df_out.reset_index(inplace=True, drop=True)
    df_out["index"] = np.arange(len(df_out))
    df_out.to_csv(options.boundary_output_tsv, sep="\t", index=False)
    print("saved")


def _generate_boundary_mutation_df(df_overlap):

    """
    For each boundary generate the following set of mutations, specified as spans:
    - every individual CTCF site overlapping a boundary,
    - all  individual CTCF site overlapping a boundary,
    - every individual CTCF site overlapping a boundary

    Returns
    -------
    df_out : pd.DataFrame
        dataframe with spans indiciating the desired positions for mutation

    """

    df_list = []
    for ind in np.unique(df_overlap["index"].values):
        d = df_overlap.iloc[df_overlap["index"].values == ind].copy()
        if d["index_2"].isna().sum():
            # if there are no overlapping CTCFs, append the whole boundary to the list of mutations
            d["span"] = d["start"].astype(str) + "-" + d["end"].astype(str)
            d["index_2"] = 0
            df_list.append(d)

        else:
            # if there are overlapping CTCFs, append each individual site
            d["index_2"] = len(d)
            df_list.append(d)
            if len(d) > 1:
                # if there are >1 overlapping CTCFs, append the joint span of all sites
                d_join = pd.DataFrame(d.iloc[0:1].copy())
                d_join["span"] = ",".join(d["span"].values)
                d_join[["strand_2", "score_2"]] = pd.NA
                df_list.append(d_join)

            d_all = pd.DataFrame(d.iloc[0:1].copy())
            d_all["span"] = d_all["start"].astype(str) + "-" + d_all["end"].astype(str)
            d_all[["strand_2", "score_2"]] = pd.NA
            df_list.append(d_all)

    df_out = pd.concat(df_list, axis=0)
    return df_out


################################################################################
# __main__
################################################################################
if __name__ == "__main__":
    main()
