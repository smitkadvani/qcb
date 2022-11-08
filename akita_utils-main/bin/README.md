Scripts for making larger-scale mutations with Akita. Generally outputs scores in h5 format. FUnctions have similar backgrounds, with some variations for sepcific mutation strategies (e.g. mutating an input sequence vs. inserting into a background sequence).

Many of them can be run with the following syntax:
```python akita_motif_scd.py -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 4 -m -o ins_test  --stats SCD,INS-16,INS-64 /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 motif_positions.8.mm10.tsv```

For spreading tasks across multiple GPUs (this assumes slurm_gf is in your pythonpath):
```python akita_motif_scd_multi.py -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --batch-size 8 -m -q gpu -p 2 --time 15:00 --num_cpus 4 -o scd_ins_sonmezer --gres gpu --constraint '[xeon-6130|xeon-2640v4]' --stats INS-16,INS-32,INS-64,INS-128,INS-256,SCD,SSD /project/fudenber_735/backup/DNN_HiC/human-mouse_5-16-21/f0_c0/train/params.json /project/fudenber_735/backup/DNN_HiC/human-mouse_5-16-21/f0_c0/train/model1_best.h5 motif_positions.8.mm10.tsv```

You can add a utility folder to your python path as follows:

```export PYTHONPATH="${PYTHONPATH}:/home1/myname/repositories/utility/"```
