#!/bin/bash
#SBATCH --job-name=safe-transformer-via-shielding
#SBATCH --partition gpgpuB
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/vol/bitbucket/ky723/workspaces/shenji/shenji/logs/%x_%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hg623

export PATH=/vol/bitbucket/ky723/workspaces/shenji/shenji/training_o3/.venv/bin/:$PATH

source /vol/bitbucket/ky723/workspaces/shenji/shenji/training_o3/.venv/bin/activate

source /vol/cuda/12.2.0/setup.sh

echo "=== GPU ==="
/usr/bin/nvidia-smi
echo -e "\n=== CPU ==="
lscpu | grep 'Model name'
echo -e "\n=== Memory & Uptime ==="
free -h
uptime
echo -e "==========\n"

cd /vol/bitbucket/ky723/workspaces/shenji/shenji
echo "train started"
# python -m training_o3.preprocess_pgn --pgn-archive ../lichess-training/lichess_db_standard_rated_2025-05.pgn.zst --out-dir ./data --shard-size 100000000 --max-games 200000
python -m training_o3.train --config training_o3/config.yaml --data data
