#!/bin/bash
python main.py --N_data 1000 --Nitr 80000 --nsym 64 --sps 4 --step_num 800 --step_num_dt 4 --power_dbm 0 --lamio 1 --fft_flag 0 --loss l1 --noise_db 20
python main.py --N_data 1000 --Nitr 80000 --nsym 32 --sps 8 --step_num 800 --step_num_dt 4 --power_dbm 0 --lamio 1 --fft_flag 0 --loss l1 --noise_db 20
python main.py --N_data 1000 --Nitr 80000 --nsym 128 --sps 2 --step_num 800 --step_num_dt 4 --power_dbm 0 --lamio 1 --fft_flag 0 --loss l1 --noise_db 20