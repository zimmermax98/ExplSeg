### Cityscapes
python -m segmentation.preprocess_cityscapes 8
python -m segmentation.img_to_numpy
python -m segmentation.train cityscapes_kld_imnet run_cityscapes_test
python -m segmentation.run_pruning cityscapes_kld_imnet run_cityscapes_test
python -m segmentation.train cityscapes_kld_imnet run_cityscapes_test --pruned
python -m segmentation.eval_valid run_cityscapes_test pruned
python -m segmentation.eval_test run_cityscapes_test pruned

### Pascal
#python -m segmentation.preprocess_pascal 8
#python -m segmentation.img_to_numpy
#python -m segmentation.train pascal_kld_imnet run_pascal_voc
#python -m segmentation.run_pruning pascal_kld_imnet run_pascal_voc
#python -m segmentation.train pascal_kld_imnet run_pascal_voc --pruned
#python -m segmentation.eval_valid run_pascal_voc pruned
#python -m segmentation.eval_test run_pascal_voc pruned