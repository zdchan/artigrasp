# ArtiGrasp: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation

## [Paper](https://arxiv.org/pdf/2309.03891.pdf) | [Project Page](https://eth-ait.github.io/artigrasp/)

<img src="docs/image/teaser.jpg" /> 

### Contents

1. [Info](#info)
2. [Installation](#installation)
3. [Demo](#demo)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Citation](#citation)
7. [License](#license)

## Info

This code was tested with Python 3.8 and gcc 9.4.0 on Ubuntu 20.04. The repository comes with all the features of the [RaiSim](https://raisim.com/) physics simulation, as ArtiGrasp is integrated into RaiSim.

The ArtiGrasp related code can be found in the [raisimGymTorch](./raisimGymTorch) subfolder. There are six environments (see [envs](./raisimGymTorch/raisimGymTorch/env/envs/)). [compose_eval](./raisimGymTorch/raisimGymTorch/env/envs/compose_eval) is for the quantitative evaluation of the Dynamic Object Grasping and Articulation task. [fixed_arti_evaluation](./raisimGymTorch/raisimGymTorch/env/envs/fixed_arti_evaluation) is for the quantitative evaluation of grasping and articulation with fixed object base.  [floating_evaluation](./raisimGymTorch/raisimGymTorch/env/envs/floating_evaluation) is for the quantitative evaluation of grasping and articulation with free object base.  [general_two](./raisimGymTorch/raisimGymTorch/env/envs/general_two), [left_fixed](./raisimGymTorch/raisimGymTorch/env/envs/left_fixed) and [multi_obj_arti](./raisimGymTorch/raisimGymTorch/env/envs/multi_obj_arti) are the training environments for two hand cooperation with free object base, left hand policy with fixed object base, right hand policy with fixed object base respectively.

## Installation


For good practice for Python package management, it is recommended to use virtual environments (e.g., `virtualenv` or `conda`) to ensure packages from different projects do not interfere with each other.

For installation, see and follow our documentation of the installation steps under [docs/INSTALLATION.md](./docs/INSTALLATION.md). Note that you need to get a valid, free license for the RaiSim physics simulation and an activation key via this [link](https://docs.google.com/forms/d/e/1FAIpQLSc1FjnRj4BV9xSTgrrRH-GMDsio_Um4DmD0Yt12MLNAFKm12Q/viewform). 

## Demo

We provide some pre-trained models to view the output of our method. They are stored in [this folder](./raisimGymTorch/data_all/). 

+ For interactive visualizations, you need to run

  ```Shell
  raisimUnity/linux/raisimUnity.x86_64
  ```

  and check the Auto-connect option.

+ To randomly choose an object and visualize the generated sequences, run

  ```Shell
  python raisimGymTorch/env/envs/general_two/runner_eval.py
  ```

## Training

- For the pre-training phase of the right hand policy with fixed-base objects, run

  ```Shell
  python raisimGymTorch/env/envs/multi_obj_arti/runner.py -re (Load the checkpoint. Otherwise start from scratch)
  ```

- For the pre-training phase of the left hand policy with fixed-base objects, run

  ```Shell
  python raisimGymTorch/env/envs/left_fixed/runner.py -re (Load the checkpoint. Otherwise start from scratch)
  ```

- For the fine-tuning phase of two hand policies with free-base objects for cooperation, run

  ```Shell
  python raisimGymTorch/env/envs/general_two/runner.py -re (Load the checkpoint. Otherwise start from scratch)
  ```

## Evaluation

- For grasping and articulation of free-base objects, run

  ```Shell
  python raisimGymTorch/env/envs/floating_evaluation/runner_eval.py -obj '<obj_name>' -test (otherwise for training set) -grasp (otherwise for articulation)
  ```

- For articulation of fixed-base objects, run

  ```Shell
  python raisimGymTorch/env/envs/fixed_arti_evaluation/runner_eval.py -obj '<obj_name>' -test (otherwise for training set)
  ```

- For Dynamic Object Grasping and Articulation task (except ketchup), run

  ```Shell
  python raisimGymTorch/env/envs/composed_eval/compose_eval.py -obj '<obj_name>' -test (otherwise for training set)
  ```

- For Dynamic Object Grasping and Articulation task (ketchup), run

  ```Shell
  python raisimGymTorch/env/envs/composed_eval/ketchup_eval.py -obj '<obj_name>' -test (otherwise for training set)
  ```

## Citation

To cite us, please use the following:

```
@inProceedings{zhang2024artigrasp,
  title={{ArtiGrasp}: Physically Plausible Synthesis of Bi-Manual Dexterous Grasping and Articulation},
  author={Zhang, Hui and Christen, Sammy and Fan, Zicong and Zheng, Luocheng and Hwangbo, Jemin and Song, Jie and Hilliges, Otmar},
  booktitle={International Conference on 3D Vision (3DV)},
  year={2024}
}
```

## License

See the following [license](LICENSE.md).







