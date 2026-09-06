@echo off

set max_sitehoice=25 50 100 150 200

for %%c in (%max_sitehoice%) do (
    python knowledge_transfer.py --city AMS+JHB+LOA+MEL+SPO+SZH --eval_city SZH --auxiliary all --model frets --pred_type city --global_epoch 100 --deploy_epoch 20 --max_site%c|| (
                    echo Error occurred on max_site%c
                    pause
                    exit /b 1
                )
)
echo All experiments completed.
