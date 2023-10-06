for dir in "avalanche"  "repo" "random" "repo_100"
do
    for data_dir in "train" "test" "training"
    do
        for sub_dir in "class_key_counts" "key_class_counts" "heatmaps" "task_key_counts" "key_task_counts"
        do
            if [ "$dir" = "random" ] && [ "$data_dir" = "training" ]
            then
                break
            fi
            if [ "$dir" = "repo_100" ] && [ "$data_dir" = "training" ]
            then
                break
            fi
            echo "Removing images in ./$dir/$data_dir/$sub_dir/"
            rm ./$dir/$data_dir/$sub_dir/*
        done
    done
    echo "Removing images in ./$dir/reduction/"
    rm ./$dir/reduction/*
    echo ""


    # if [ "$dir" = "repo_100" ]
    # then
    #     n_classes=100
    #     for ((i = 0; i < n_classes; i++)); do
    #         dir_name="class_$i"
    #         echo "Removing images in ./$dir/saliency_maps/$dir_name/"
    #         rm ./$dir/saliency_maps/$dir_name/*
    #     done
    # fi
done