mimic : LR = 0.1
nih   : LR = 



mimic : 
	python main_mlc.py \
  --dataname mimic \
  --dataset_dir /path/to/mimic \
  --backbone resnet50 \
  --optim SGD \
  --lr 0.1 \
  --momentum 0.9 \
  --wd 1e-4 \
  --batch-size 256 \
  --enable_splicemix \
  --splicemix_prob 0.5 \
  --splicemix_mode SpliceMix-CL \
  --output ./output/mimic_resnet50_rolt_splicemix