python3.9 get_keys_plots.py --dataset train --model avalanche
python3.9 get_keys_plots.py --dataset train --model repo
python3.9 get_keys_plots.py --dataset train --model random

python3.9 get_keys_plots.py --dataset test --model avalanche
python3.9 get_keys_plots.py --dataset test --model repo
python3.9 get_keys_plots.py --dataset test --model random

python3.9 get_keys_plots.py --dataset training --model repo
python3.9 get_keys_plots.py --dataset training --model avalanche

python3.9 visualize_embeddings.py --reduction_alg tsne --model avalanche
python3.9 visualize_embeddings.py --reduction_alg tsne --model repo
python3.9 visualize_embeddings.py --reduction_alg tsne --model random

python3.9 visualize_embeddings.py --reduction_alg pca --model avalanche
python3.9 visualize_embeddings.py --reduction_alg pca --model repo
python3.9 visualize_embeddings.py --reduction_alg pca --model random