#!/bin/sh
python process_kitting_dataset_to_fit_this_implementation.py
python main.py --batch_size 16 --mode train --data_path kitting_exp_skill_3.npy
python main.py --batch_size 16 --mode test --data_path kitting_exp_skill_3.npy
