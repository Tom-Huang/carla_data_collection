# README

## Usage

The code is for CARLA version 0.9.4. It needs to be modified to adapt to other versions of CARLA.

### Data Generation

1. Copy files in "poseprocesseffect" to "carla_root_folder/Unreal/CarlaUE4/Plugins/Carla/Content/PostProcessingMaterials"

2. put "collect_albedo_depth_normal.py" to "carla_root_folder/PythonAPI/examples" and run it using terminal. You can modify the code to generate the lidar point cloud and the semantic image for front view, left view, right view and rear view simultaneously(just uncomment some lines).

3. data will be created in folder "carla_root_folder/PythonAPI/examples/images"

### Point cloud processing

After generation of the semantic image for front view, left view, right view and rear view and the lidar point cloud at the same time step, you may want to generate the semantic point cloud(point cloud with different colors according to their semantic meanings). The usage is as below:

1. Go to folder `lidar_processing`.

```python
cd ./lidar_processing
```

2. If the four views of data are generated with the code in Data Generation as .npy format, you have to run the `npy2png.py` file first.

```python
python npy2png.py
```

3. If the four views of data are already .png format or after you do the step 1, you can run the following to generate semantic lidar points. The results will be in `./data/for_training/2019-05-31_22_05/Lidar/labeled/`.

```python
python add_lidar_label.py
```

The semantic point cloud will have the following look:

![original lidar](./imgs/lidar.png)
original lidar

![labeled lidar](./imgs/lidar_labeled.png)
labeled lidar
