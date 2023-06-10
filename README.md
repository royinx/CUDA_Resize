
# Cupy, CUDA Bilinear interpolation

Ultra fast Bilinear interpolation in image resize with CUDA.


`lerp.py` : Concept and code base (*single thread, may take a while to run). <br/>
`resize_ker.cu` : CUDA test case in `C`. <br/>
`resize.py` : Cupy example <br/>

(*PyCUDA(deprecated) is no longer support , use cupy instead )

Requirements:
>- GPU (compute capability: 3.0 or above, testing platform: 7.5)
>- CUDA driver
>- Docker and nvidia docker
---
Pros:
- support Batch image.
- no shared object .so and .dll binary file
- Install cupy and use
- Compatible to `Numpy` library
- pass the GPU array to TensorRT directly. 

Cons:
- still need the concept of CUDA programming
- SourceModule have to write in C CUDA, including all CUDA kernel and device code

---
### Build 

```bash
git clone https://github.com/royinx/CUDA_Resize.git 
cd CUDA_Resize
docker build -t lerp_cuda .
docker run -it --runtime=nvidia -v ${PWD}:/py -w /py lerp_cuda bash 

# For Cupy implementation
python3 resize.py

# For concept
python3 lerp.py

# For CUDA kernel testing
nvcc resize_free.cu -o resize_free.o && ./resize_free.o

# For benmarking
wget http://images.cocodataset.org/zips/val2017.zip
python3 benchmark.py
```
<details><summary> Advance </summary>

```bash
docker run -it --privileged --runtime=nvidia -p 20072:22 -v ${PWD}:/py -w /py lerp_cuda bash 
sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
nvcc resize_free.cu -o resize_free.o
nsys profile ./resize_free.o

ncu -o metrics /bin/python3 resize_free.py  > profile_log
ncu -o metrics /bin/python3 resize_free.py
```
Remark: Development platform is in dockerfile.opencv with OpenCV in C for debugging

Function Working well in pycuda container, you dont need to build OpenCV.
</details>

### Benchmark
#### AWS g4dn.xlarge (Tesla T4)
> ratio = T4 (ms) / Xeon Platinum 8259CL (ms)
![](benchmark/g4dn.png)

> ms/img on T4 
![](benchmark/t4.png)


#### 2080ti
> ratio = 2080ti (ms) / Ryzen 3950x (ms)
![](benchmark/2080ti.png)


<!-- 
|in\out (times)|(1920, 1080)|(960, 540)|(480, 270)|(240, 135)|(120, 67)|(60, 33)|(30, 16)|
|:-----------:|:----------:|:--------:|:--------:|:--------:|:-------:|:------:|:------:|
|(1920, 1080) |1.70|2.25|6.26|7.26|5.33|6.51|3.36|
|(960, 540)   |2.58|2.02|3.34|9.19|6.83|5.31|3.47|
|(480, 1080)  |2.54|1.72|1.60|5.39|7.21|4.85|2.75|
|(240, 1080)  |2.67|1.74|3.97|1.52|6.55|7.77|5.34|
|(120, 1080)  |2.79|1.70|3.03|4.13|0.79|3.97|4.40|
|(60, 1080)   |2.59|1.57|2.36|3.19|3.55|0.86|4.48|
|(30, 1080)   |2.59|1.58|2.59|2.85|2.94|3.62|1.06|
 -->
 
<!-- 

(1920, 1080)    ->      (1920, 1080)    :        CPU: 1.1610580560000017         | CUDA: 0.6842355520000183
(1920, 1080)    ->      (960, 540)      :        CPU: 0.38829656900009013        | CUDA: 0.17241025099997387
(1920, 1080)    ->      (480, 270)      :        CPU: 0.3424055989997896         | CUDA: 0.054677475000175946
(1920, 1080)    ->      (240, 135)      :        CPU: 0.15193816399983007        | CUDA: 0.02091631899997992
(1920, 1080)    ->      (120, 67)       :        CPU: 0.06044440600021517        | CUDA: 0.011333408000041345
(1920, 1080)    ->      (60, 33)        :        CPU: 0.027777105000154734       | CUDA: 0.004265433999989909
(1920, 1080)    ->      (30, 16)        :        CPU: 0.010086882999871705       | CUDA: 0.0030035350000616745
(960, 540)      ->      (1920, 1080)    :        CPU: 1.7609010599999237         | CUDA: 0.6830883400000403
(960, 540)      ->      (960, 540)      :        CPU: 0.3343937270000197         | CUDA: 0.16587755199998355
(960, 540)      ->      (480, 270)      :        CPU: 0.14595758299992667        | CUDA: 0.04363917599994238
(960, 540)      ->      (240, 135)      :        CPU: 0.13546343800010163        | CUDA: 0.014742367000167178
(960, 540)      ->      (120, 67)       :        CPU: 0.051735301000007894       | CUDA: 0.007577149999974608
(960, 540)      ->      (60, 33)        :        CPU: 0.020995430000084525       | CUDA: 0.003951062000055572
(960, 540)      ->      (30, 16)        :        CPU: 0.010350153999866052       | CUDA: 0.002982846999771027
(480, 270)      ->      (1920, 1080)    :        CPU: 1.7658809170001177         | CUDA: 0.6943260289999671
(480, 270)      ->      (960, 540)      :        CPU: 0.2858867139998438         | CUDA: 0.16626066599985734
(480, 270)      ->      (480, 270)      :        CPU: 0.07003998500010766        | CUDA: 0.04367567899976166
(480, 270)      ->      (240, 135)      :        CPU: 0.07154810400015776        | CUDA: 0.01326615799996489
(480, 270)      ->      (120, 67)       :        CPU: 0.04758754900012718        | CUDA: 0.006600883000032809
(480, 270)      ->      (60, 33)        :        CPU: 0.019298720999813668       | CUDA: 0.003976251000153752
(480, 270)      ->      (30, 16)        :        CPU: 0.008601805000239438       | CUDA: 0.003122667000070578
(240, 135)      ->      (1920, 1080)    :        CPU: 1.7806759940001484         | CUDA: 0.6674624740001036
(240, 135)      ->      (960, 540)      :        CPU: 0.27071022400002676        | CUDA: 0.15557714400006262
(240, 135)      ->      (480, 270)      :        CPU: 0.16034231799972076        | CUDA: 0.0403976719998127
(240, 135)      ->      (240, 135)      :        CPU: 0.01737634100015839        | CUDA: 0.01142766900022707
(240, 135)      ->      (120, 67)       :        CPU: 0.027595563999852857       | CUDA: 0.004216191999944385
(240, 135)      ->      (60, 33)        :        CPU: 0.01613739500032807        | CUDA: 0.0020755900002313865
(240, 135)      ->      (30, 16)        :        CPU: 0.006980756999610094       | CUDA: 0.0013075480001134565
(120, 67)       ->      (1920, 1080)    :        CPU: 1.7407769839998082         | CUDA: 0.6237445080001862
(120, 67)       ->      (960, 540)      :        CPU: 0.2684578630000942         | CUDA: 0.15776679299995067
(120, 67)       ->      (480, 270)      :        CPU: 0.12010822800039023        | CUDA: 0.039634367000076054
(120, 67)       ->      (240, 135)      :        CPU: 0.046079689999942275       | CUDA: 0.011151117000054
(120, 67)       ->      (120, 67)       :        CPU: 0.0030908759999874746      | CUDA: 0.003898979000041436
(120, 67)       ->      (60, 33)        :        CPU: 0.007907080999871141       | CUDA: 0.0019918509999570233
(120, 67)       ->      (30, 16)        :        CPU: 0.005631100000073275       | CUDA: 0.0012808260001975214
(60, 33)        ->      (1920, 1080)    :        CPU: 1.7397525290002704         | CUDA: 0.6712527989999444
(60, 33)        ->      (960, 540)      :        CPU: 0.2618644579999909         | CUDA: 0.16646550899997692
(60, 33)        ->      (480, 270)      :        CPU: 0.1029992480000601         | CUDA: 0.043565018999856875
(60, 33)        ->      (240, 135)      :        CPU: 0.03758387700020194        | CUDA: 0.011800254000036148
(60, 33)        ->      (120, 67)       :        CPU: 0.013579181000068274       | CUDA: 0.0038238310000906495
(60, 33)        ->      (60, 33)        :        CPU: 0.0015284880000763224      | CUDA: 0.0017852409999932206
(60, 33)        ->      (30, 16)        :        CPU: 0.003489628000124867       | CUDA: 0.0007788719999552995
(30, 16)        ->      (1920, 1080)    :        CPU: 1.7044012149999617         | CUDA: 0.6583124620002536
(30, 16)        ->      (960, 540)      :        CPU: 0.26106682399995407        | CUDA: 0.16568029600023237
(30, 16)        ->      (480, 270)      :        CPU: 0.11399052499962181        | CUDA: 0.04398374100014735
(30, 16)        ->      (240, 135)      :        CPU: 0.03368437599999652        | CUDA: 0.011832932000061192
(30, 16)        ->      (120, 67)       :        CPU: 0.011452169999756734       | CUDA: 0.003898591999927703
(30, 16)        ->      (60, 33)        :        CPU: 0.005362819000197305       | CUDA: 0.0014827379998223478
(30, 16)        ->      (30, 16)        :        CPU: 0.00131592600030217        | CUDA: 0.001240176999999676 -->