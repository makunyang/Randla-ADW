

## 1 install
```
Python 3.6, Tensorflow 2.6, CUDA 11.4 and cudnn
```
## 2 Download data-SUM
```
https://3d.bk.tudelft.nl/opendata/
```

## 3 Prepare the dataset
```
Organize the SUM dataset in the format of semantic3D.
run data_prepare.py
```
## 4 train

```
run main_SUM.py
```


## 5 vis
python ins_vis.py -p /home/ma/Desktop/TTT/randla-net-tf2-main/data/semantic3d/original_ply/1.ply -l /home/ma/Desktop/TTT/randla-net-tf2-main/test/Log_2025-06-18_08-49-29/predictions/1.labels --min_samples 10 --eps 5  --visualize

