# Readme

## Run training

```python
python -m chess_transformer.train --config configs/default.yaml --data ./data
tensorboard --logdir checkpoints/tb
# after training
python -m chess_transformer.evaluate \
       --config configs/default.yaml \
       --data ./data \
       --checkpoint checkpoints/epoch_010.pt
```
