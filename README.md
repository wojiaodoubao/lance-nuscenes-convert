This tool transforms the nuScenes dataset into a lance dataset.

## What is nuScenes dataset.
The nuScenes dataset (pronounced /nuːsiːnz/) is a public large-scale dataset for autonomous driving developed by the team at Motional (formerly nuTonomy). [nuScenes](https://www.nuscenes.org/nuscenes)

The best way to understand nuScene is using the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit). It provides
a jupyter tutorial, see [tutorial](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/tutorials/nuscenes_tutorial.ipynb) for more details.
```shell
jupyter notebook ./python-sdk/tutorials/nuscenes_tutorial.ipynb
```

## Environment

### 1. Prepare nuScenes dataset.
The nuScenes dataset could be downloaded from [nuScenes](https://www.nuscenes.org/nuscenes#download).
Download the mini dataset (with 10 scenes) for test or the full dataset (with 1000 scenes) for full transformation. 

### 2. Install nuscenes-devkit.

#### Install with pip. (Recommended)
```shell
pip install nuscenes-devkit
```

#### Advanced install.

See [Install doc](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md)

### 3. Install pylance
```shell
pip install pylance
```

## Transform
```shell
python nuscenes_convert.py {nuscenes_root} {nuscenes_version} {lance_root}
```