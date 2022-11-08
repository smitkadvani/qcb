### Commands
1. 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234 (only the "<<" orientations)
    - generating tsv table
        ```
        python generate_log-spacing_df.py --num-strong 10 --num-weak 10 --backgrounds-indices 1,2,3,4 --orientation-string "<<" --filename 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234 --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2022-10-11_200_10W_10S_spacings_0-1000_right_bg1234.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 16 --max_proc 16
        ```

2. 2022-10-12_200_100W_100S_spacings_0-1000_(orientation)_bg12 (all orientations)
    - generating tsv table
        ```
        python generate_log-spacing_df.py --num-strong 100 --num-weak 100 --orientation-string ">>" --backgrounds-indices 1,2 --filename 2022-10-12_200_100W_100S_spacings_0-1000_right_bg12 --verbose
        ```
    - Akita predictions
        ```
        python multiGPU-virtual_symmetric_experiment.py /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/params.json /project/fudenber_735/tensorflow_models/akita/v2/models/f0c0/train/model1_best.h5 2022-10-12_200_100W_100S_spacings_0-1000_right_bg12.tsv -f /project/fudenber_735/genomes/mm10/mm10.fa --head-index 1 --model-index 1 --batch-size 4  --stats SCD,INS-16 -p 16 --max_proc 16
        ```

