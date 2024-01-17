#python main.py --dataset CUB-200-2011 --gpu_ids 0 --log_dir ./runs/pipnet_cub_cnext26
python main.py --dataset CARS --gpu_ids 1 --log_dir ./runs/pipnet_cars_cnext26
#python main.py --dataset pets --gpu_ids 1 --log_dir ./runs/pipnet_pets_cnext26 --lr_block 0.0001 --lr_net 0.0001