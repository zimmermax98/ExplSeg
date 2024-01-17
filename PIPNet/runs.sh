python main.py --task classification --dataset CUB-200-2011 --gpu_ids 0 --seed 1 --run_name cub_cnext26
python eval.py --task classification --dataset CUB-200-2011 --gpu_ids 0 --seed 1 --run_name cub_cnext26
#python main.py --task classification --dataset CARS --gpu_ids 0 --run_name cars_cnext26
#python main.py --task classification --dataset pets --gpu_ids 0 --run_name pets_cnext26 --lr_block 0.0001 --lr_net 0.0001
python main.py --task segmentation --dataset VOC --gpu_ids 0 --seed 1 --run_name voc_cnext26_pool3_stride1
python eval.py --task segmentation --dataset VOC --gpu_ids 0 --seed 1 --run_name voc_cnext26_pool3_stride1