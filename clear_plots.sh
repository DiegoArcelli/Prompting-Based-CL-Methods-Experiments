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
            echo "Removing images in ./plots/$dir/$data_dir/$sub_dir/"
            rm ./plots/$dir/$data_dir/$sub_dir/*
        done
    done
    echo "Removing images in ./plots/$dir/reduction/"
    rm ./plots/$dir/reduction/*
    echo ""

    n_classes=100
    for ((i = 0; i < n_classes; i++)); do
        dir_name="class_$i"
        echo "Removing images in ./plots/$dir/saliency_maps/$dir_name/"
        rm ./plots/$dir/saliency_maps/$dir_name/*
    done
    
done