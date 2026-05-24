GE^2

### Environment with Conda

        $ conda install -c "nvidia/label/cuda-11.3.1" cuda-toolkit
        $ conda install -c conda-forge cudnn # if needed
        $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

### Build and Install

        $ mkdir build
        $ cd build
        $ cmake ..
        $ make gege -j           # build only python bindings
        $ make pip-install -j    # install pip package 

### Run Commands

        $ gege_preprocess --dataset twitter --output_dir datasets/twitter -ds 0.9 0.05 0.05 --num_partition 16
        $ CUDA_VISIBLE_DEVICES=0,1 gege_train gege/configs/fb15k.yaml

### Reproduction Notes

- [FB86M_C32_SINGLE_GPU_REPRO.md](FB86M_C32_SINGLE_GPU_REPRO.md) records the
  ARC c32 Freebase86M p32 q4 single-GPU 10-epoch timing and
  eval-only-from-checkpoint workflow.
- [FB86M_2GPU_RUN_TAGS.md](FB86M_2GPU_RUN_TAGS.md) tags the c32/ARC
  Freebase86M 2-GPU runs as correct, invalid, or current repro attempts, with
  the timing/config differences recorded.


#### Acknowledgements

We reuse most of components in [Marius](https://github.com/marius-team/marius) because they are well-developed.
