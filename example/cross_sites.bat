@echo off

python knowledge_transfer.py --city SZH --max_site00 --eval_percentage 20 --auxiliary all --model frets --pred_type sitsitebal_epoch 100 --deploy_epoch 20

echo All experiments completed.
