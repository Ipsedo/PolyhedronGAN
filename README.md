# PolyhedronGAN
Generate 3D polyhedron object with GAN

## modif voxelfuse

`voxel_model.py` at line 2088 -> comment

`voxel_model.py` at line 2086 redirect to `/dev/null`
```python
command_string = command_string + ' output.geo -3 -format msh > /dev/null'
```

`voxel_model.py` at line 1436 : remove tqdm
