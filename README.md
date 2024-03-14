# MoST (CVPR 2024)

This is the official implementation of "MoST: Motion Style Transformer between Diverse Action Contents" [[paper]](https://arxiv.org/abs/2403.06225) [[project]](https:)




![framework](https://github.com/Boeun-Kim/MoST/blob/main/figures/overview.png)



 ## Dependencies

We tested our code on the following environment.

- CUDA 11.4

- python 3.8.10

  

Install python libraries with:

```
pip install -r requirements.txt
```



## Data preparation

1. Download raw motion dataset **mocap_xia.zip** to `./data/` from the following link.

   - https://deepmotionediting.github.io/style_transfer 

   

2. Unzip the dataset

   ```
   cd ./data
   unzip mocap_xia.zip
   ```

   

3. Preprocess the data

   ```
   python xia_preprocess.py
   ```

   Three folders will be generated.

   - preprocessed_xia : Preprocessed training data and distribution data.
   - preprocessed_xia_test: Copies of testing files.
   - preprocessed_xia_gt: Copies of training files.



## Training

Sample arguments for training:

(please refer to `arguments.py` for detailed arguments.)

```
python train.py --save_path results
```



- Pretrained weights can be downloaded via this [link]( https://drive.google.com/file/d/1yhkAoyDLJHRsJE5HWcyoJ2tprsyZ3msF/view?usp=sharing.). 

  Make a directory `./pretrained/` and download **xia_pretrained.pth** into `./pretrained/`.



## Demo

Sample arguments for demonstration:

(please refer to `arguments.py` for detailed arguments.)

```
python demo.py \
    --model_path pretrained/xia_pretrained.pth
    --demo_datapath data/preprocessed_xia_test
    --cnt_clip angry_13_000 --sty_clip strutting_16_000 
```

**generated_motion.bvh** will be created.



## Evaluation

Arguments for calculate Content Consistency (CC):

```
python evaluation_CC.py \
	--model_path pretrained/xia_pretrained.pth
```



Arguments for calculate Style Consistency ++(SC++):

```
python evaluation_SCpp.py \
	--model_path pretrained/xia_pretrained.pth
```



## Reference

Part of our code is based on [GL-Transformer](https://github.com/Boeun-Kim/GL-Transformer), [Aberman et. al.](https://deepmotionediting.github.io/style_transfer), and [Park et. al.](https://github.com/soomean/Diverse-Motion-Stylization)

Thanks to the great resources.



## Citation

Please cite our work if you find it useful.

```
@misc{kim2024most,
      title={MoST: Motion Style Transformer between Diverse Action Contents}, 
      author={Boeun Kim and Jungho Kim and Hyung Jin Chang and Jin Young Choi},
      year={2024},
      eprint={2403.06225},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
