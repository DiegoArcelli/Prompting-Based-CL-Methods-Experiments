for dir in "batchwise" "no_batchwise" "repo"
do
    for sub_dir in "class_key_counts" "key_class_counts" "heatmaps" "task_key_counts" "key_task_counts"
    do
        echo "Removing images in ./$dir/$sub_dir/"
        rm ./$dir/$sub_dir/*
    done
done