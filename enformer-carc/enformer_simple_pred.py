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

import tensorflow as tf
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
from basenji import stream

gpus = tf.config.experimental.list_physical_devices('GPU')


# @title `Enformer`, `EnformerScoreVariantsNormalized`, `EnformerScoreVariantsPCANormalized`,
SEQUENCE_LENGTH = 393216
preds_length = 896

class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)

class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)

def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}

#for gpu in gpus:
#  tf.config.experimental.set_memory_growth(gpu, True)
print(gpus)

'''
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)
  parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-o',dest='out_dir',
      default='preds',
      help='Output directory [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--batch-size', dest='batch_size',
      default=None, type='int',
      help='Specify batch size')
    
    
  parser.add_option('--species',dest='species',
      default='human',
      help='species to predict [Default: %default]')

  (options, args) = parser.parse_args()

  if len(args) == 3:
    # single worker
    params_file = args[0]
    model_file = args[1]
    bed_file = args[2]

  elif len(args) == 5:
    # multi worker
    options_pkl_file = args[0]
    params_file = args[1]
    model_file = args[2]
    bed_file = args[3]
    worker_index = int(args[4])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide parameters and model files and QTL VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  else:
    plot_dir = None

  options.shifts = [int(shift) for shift in options.shifts.split(',')]

  random.seed(44)

  #################################################################
  # read parameters and targets


  if options.batch_size is None:
    batch_size = params_train['batch_size']
  else: 
    batch_size = options.batch_size
  print(batch_size)

  if options.targets_file is not None:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description

  #################################################################
  # setup model

  # load model
  enformer_model = Enformer(model_file)

  # dummy target info
  if options.targets_file is None:
    num_targets_human = 5313
    num_targets_mouse = 1643
    if options.species == 'human':
        num_targets = num_targets_human
    else:
        num_targets = num_targets_mouse

    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  # load motifs

  # filter for worker motifs
  if options.processes is not None:
    seq_coords_full = pd.read_csv(bed_file, sep='\t')
    num_motifs_total = len(seq_coords_full)
    worker_bounds = np.linspace(0, num_motifs_total, options.processes+1, dtype='int')

    seq_coords_df = seq_coords_full.loc[worker_bounds[worker_index]:worker_bounds[worker_index+1],:]
  else:
    # read motif positions from csv
    seq_coords_df = pd.read_csv(bed_file, sep='\t')

  num_motifs = len(seq_coords_df)

  # open genome FASTA
  fasta_extractor = FastaStringExtractor(options.genome_fasta)
    
  def seqs_gen():
    for s in seq_coords_df.itertuples():
      target_interval = kipoiseq.Interval(s.chrom, s.start, s.end)  # @param
      print((s.chrom, s.start, s.end))
      #
      seq_1hot = one_hot_encode(
          fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
      yield seq_1hot

    
  #################################################################
  # setup output
  out_h5 = initialize_output_h5(options.out_dir, seq_coords_df, target_ids, target_labels)
  print('initialized')

  #################################################################
  # predict SNP scores, write output

  write_thread = None

  # initialize predictions stream
  preds_stream = stream.PredStreamSonnet(enformer_model, seqs_gen(), batch_size, species = options.species)

  # predictions index
  pi = 0
  for si in range(num_motifs):
    # get predictions
    ref_preds = preds_stream[pi]
    pi += 1
    # process SNP
    write_snp(ref_preds, out_h5, si)

  """Write SNP predictions to HDF."""
  out_h5.close()
  fasta_extractor.close()

def initialize_output_h5(out_dir, seq_coords_df, target_ids, target_labels):

  num_targets = len(target_ids)
  num_seqs = len(seq_coords_df)

  out_h5 = h5py.File('%s/scd.h5' % out_dir, 'w')
  seq_coords_df_dtypes= seq_coords_df.dtypes
  for key in seq_coords_df:
    if (seq_coords_df_dtypes[key] is np.dtype('O')):
      out_h5.create_dataset(key, data=seq_coords_df[key].values.astype('S'))
    else:
      out_h5.create_dataset(key, data=seq_coords_df[key])

  out_h5.create_dataset('preds', shape=(num_seqs, preds_length, num_targets), dtype='float16')

  return out_h5

def write_snp(ref_preds, out_h5, si, ):
  """Write predictions to HDF."""
  ref_preds = ref_preds.astype('float32')
  out_h5['preds'][si] = ref_preds


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
