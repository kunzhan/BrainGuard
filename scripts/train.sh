#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

train_type=vision
save_path=./logs/$train_type/
data_root=/data/natural-scenes-dataset # data path

python -u main.py \
-ls 1 \
-gr 600 \
--cuda_id '{"server":-1, "1": 0, "2": 1, "5": 2, "7": 3}' \
-tp $train_type \
-p 24 \
-lbs 50 \
--data_root $data_root \
2>&1 | tee $save_path/$now.log
