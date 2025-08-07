# Readme

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
