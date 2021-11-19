for lr in 0.001 0.0001
do
    for model in resnet50 efficientnet0
    do
        exp_name=exp_model_"$model"_lr_"$lr"
        python Main.py --csv_data_path Data/data/images.csv \
                        --data_path Data/images \
                        --batch_size 16 \
                        --lr $lr \
                        --save_path Data/models/ \
                        --write_logs \
                        --log_dir Data/logs/ \
                        --experiment_name $exp_name \
                        --weight_loss \
                        --model_type $model &
    done
done