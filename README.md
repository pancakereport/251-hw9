# 251-hw9

This homework was completed by Alan Jian and Silas Gifford. 

In this homework we demonstrate that training using two GPUs was faster than training with one GPU. One GPU took roughly 45 minutes for a single epoch whereas two GPUS took roughly 28 minutes. Both approaches achieved similar loss and accuracies as documented in `ddp_tensorboard.png` and `one_vm_tensorboard.png` after one epoch on two GPUs and two epochs on one GPU.

To reproduce for DDP, update the `URL` variable to the correct address and copy `imagenet_ddp.ipynb` into a second container. Everything can stay the same but update `RANK` to be `1`.

Useful commands: 
`docker run --rm -v ~/data:/data --net=host --gpus=all -ti nvcr.io/nvidia/pytorch:23.02-py3 bash` to create a container 
`jupyter lab --ip=0.0.0.0 --allow-root` to open jupyter lab within a container
`tensorboard --logdir=/data/runs` run within a container to access tensorboard
