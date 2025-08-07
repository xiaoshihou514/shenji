# Readme

## Run preprocess

```python
python -m preprocess_pgn.py \
      --pgn-archive /data/lichess_db_2025-07.pgn.zst \
      --out-dir ./data \
      --shard-size 1_000_000 --max-games 5_000_000
```

## Run training

```python
python -m training_o3.train --config config.yaml --data ./data
tensorboard --logdir checkpoints/tb
# after training
python -m training_o3.evaluate \
       --config config.yaml \
       --data ./data \
       --checkpoint checkpoints/epoch_010.pt
```
