# When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup>, *[Brian Price](https://www.brianpricephd.com/)*<sup>2</sup>, *[Scott Cohen](https://research.adobe.com/person/scott-cohen/)*<sup>2</sup>, and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1</sup>
<br></br><sup>1</sup>York University &nbsp;&nbsp; <sup>2</sup>Adobe Research

![WB_sRGB_fig1](https://user-images.githubusercontent.com/37669469/76103171-3d3bf600-5f9f-11ea-9267-db077e7ddb51.jpg)

Reference code for the paper [When Color Constancy Goes Wrong:
Correcting Improperly White-Balanced Images.](http://openaccess.thecvf.com/content_CVPR_2019/papers/Afifi_When_Color_Constancy_Goes_Wrong_Correcting_Improperly_White-Balanced_Images_CVPR_2019_paper.pdf) Mahmoud Afifi, Brian Price, Scott Cohen, and Michael S. Brown, CVPR 2019. If you use this code or our dataset, please cite our paper:
```
@inproceedings{afifi2019color,
  title={When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images},
  author={Afifi, Mahmoud and Price, Brian and Cohen, Scott and Brown, Michael S},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={1535--1544},
  year={2019}
}
```

The original source code of our paper was written in Matlab. We also provide a Python version of our code. We tried to make both versions identical.
However, there is no guarantee that the Python version will give exactly the same results. 
The differences should be due to rounding errors when we converted our model to Python or differences between Matlab and OpenCV in reading compressed images.



#### Quick start

##### 1. Matlab:
[![View Image white balancing on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/73428-image-white-balancing)
1. Run `install_.m`
2. Run `demo.m` to process a single image or `demo_images.m` to process all images in a directory.
3. Check `evaluation_examples.m` for examples of reporting errors using different evaluation metrics. Also, this code includes an example of how to hide the color chart for Set1 images.

##### 2. Python:
1. Requirements: numpy, opencv-python, and skimage (skimage is required for evaluation code only).
2. Run `demo.py` to process a single image or `demo_images.py` to process all images in a directory.
3. Check `evaluation_examples.py` for examples of reporting errors using different evaluation metrics. Also, this code includes an example of how to hide the color chart for Set1 images.


#### Graphical user interface
We provide a Matlab GUI to help tuning our parameters in an interactive way. Please, check `demo_GPU.m`.
<p align="center">
  <img src="https://user-images.githubusercontent.com/37669469/76103283-6c526780-5f9f-11ea-9f2c-ad9d87d95fb7.gif">
</p>




#### Code/GUI parameters and options
1. `K`: Number of nearest neighbors in the KNN search (Sec. 3.4 in the paper) -- change its value to enhance the results.
2. `sigma`: The fall-off factor for KNN blending (Eq. 8 in the paper) -- change its value to enhance the results.
3. `device`: GPU or CPU (provided for Matlab version only).
4. `gamut_mapping`: Mapping pixels in-gamut either using scaling (`gamut_mapping= 1`) or clipping  (`gamut_mapping= 2`). In the paper, we used the clipping options to report our results, 
but the scaling option gives compelling results in some cases (esp., with high-saturated/vivid images). 
5. `upgraded_model` and `upgraded`: To load our upgraded model, use `upgraded_model=1` in Matlab or `upgraded=1` in Python. The upgraded model has new training examples. In our paper results, we did not use this model.

### Dataset

![dataset](https://user-images.githubusercontent.com/37669469/80766673-f3413d80-8b13-11ea-98f2-9dcebaa481d2.png)

In the paper, we mentioned that our dataset contains over 65,000 images. We further added two additional sets of rendered images, for a total of 105,638 rendered images. 
You can download our dataset from [here](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html). You can also download the dataset from the following links:

Input images: [Part1](https://ln2.sync.com/dl/df390d230/bcxms94b-fh7wiwb2-cjv22e95-ijqq8pry) | [Part2](https://ln2.sync.com/dl/a91b94bf0/frnsyykq-z3hhmjkj-adrxqj3h-v6v8637z/view/default/9967673500008) | [Part3](https://ln2.sync.com/dl/98719b4f0/i9zh42sd-7isdxqvh-rbrhgxbc-z7adicv4) | [Part4](https://ln2.sync.com/dl/07b36ff40/xrfe55mc-zjda4wp7-67jxgug4-7cjw5qda) | [Part5](https://ln2.sync.com/dl/7f8be8910/bwnahjub-ttystr9d-dnvu2wuj-gez7enha) | [Part6](https://ln2.sync.com/dl/a80481330/27zamddw-e6zezbpt-erqt5e3a-5x7we5uj) | [Part7](https://ln2.sync.com/dl/c647defb0/k824nusp-nb964z7f-xd6q79i7-v7j8w3z9) | [Part8](https://ln2.sync.com/dl/b0433ce80/4gbk7q9q-b96s62vi-qektmg5t-akqhueen) | [Part9](https://ln2.sync.com/dl/271048960/f2c4gr6m-9frsuuc7-g5r47tzh-4s8m55tk) | [Part10](https://ln2.sync.com/dl/21ce83f60/v36jwspj-e4mw2vtb-s6ifkgmv-jzc8mvya)

Input images [a single ZIP file]: [Download (PNG lossless compression)](https://ln2.sync.com/dl/21ce83f60/v36jwspj-e4mw2vtb-s6ifkgmv-jzc8mvya) | [Download (JPEG)](https://ln2.sync.com/dl/823095230/w94kcz2k-778ezdij-7xanis7k-67wtt6b7) | [Google Drive Mirror (JPEG)](https://drive.google.com/file/d/12UhutFIMgnm27Eo6zrieat_kwbneh8Lw/view?usp=sharing)

Input images (without color chart pixels): [Part1](https://ln2.sync.com/dl/bd8d95590/jnd4k56e-firy4vq7-8rdjucac-zfr8a47f/view/default/9967673050008) | [Part2](https://ln2.sync.com/dl/e99ba85e0/3t3wyk8n-u5c5cc7v-xr5yzh9x-wz69u97d) | [Part3](https://ln2.sync.com/dl/76cf59c80/hk7vazpq-g3tqrnt2-3ptcqw8y-fmwtdzzx) | [Part4](https://ln2.sync.com/dl/428149ef0/r5e6ahwr-ubhqugd6-bendw5ac-cdyvif99) | [Part5](https://ln2.sync.com/dl/5bc462790/y2nkwaue-z6jvs798-7gps6k8m-nhq7z89b) | [Part6](https://ln2.sync.com/dl/c659fee90/unka53m7-gxf2hmpw-ts3fqewc-9a7ekhf6) | [Part7](https://ln2.sync.com/dl/945b316e0/xzsq94w2-k4t4bfut-a7r2qh2d-y683fgk8) | [Part8](https://ln2.sync.com/dl/997b2b460/ig8rnuhc-e488k3y2-9j7iwva5-vv4siwp4) | [Part9](https://ln2.sync.com/dl/d69b8cb70/455f389w-jpzt2pm8-2f7pgdz8-g4dwqexm) | [Part10](https://ln2.sync.com/dl/c35a43450/gdrfdgz2-a34fjigz-5pwmgcth-2hw3ztvb)

Input images (without color chart pixels) [a single ZIP file]: [Download (PNG lossless compression)](https://ln2.sync.com/dl/c35a43450/gdrfdgz2-a34fjigz-5pwmgcth-2hw3ztvb) | [Download (JPEG)](https://ln2.sync.com/dl/69186ed90/vhk63ik9-mfun6pmz-y4nd4hqu-bnfrxv53) | [Google Drive Mirror (JPEG)](https://drive.google.com/file/d/1p8X-328dHw0KxkEgKfUHiDd-sV1e0kKV/view?usp=sharing)

Augmented images (without color chart pixels): [Download](https://ln2.sync.com/dl/fd890f450/qptvg83f-h5evnawu-62ksiv99-jjmtiwyv) (rendered with additional/rare color temperatures)

Ground-truth images: [Download](https://ln2.sync.com/dl/1f607c380/ypyw5z4p-q765pviu-rc8tzi2n-4pyyep8h)

Ground-truth images (without color chart pixels): [Download](https://ln2.sync.com/dl/afb9c68a0/kzbvche9-wfqfddjx-462f8xdv-pncntp8g/view/default/9967672880008)

Metadata files: [Input images](https://ln2.sync.com/dl/1ecab3360/e452ufey-6q23a2mn-bgnxu5x8-cu2hmj8f/view/default/9967672840008) | [Ground-truth images](https://ln2.sync.com/dl/e386982f0/9t49ej9n-db6bmkr9-gaactnii-kbyua7gn)

Folds: [Download](https://ln2.sync.com/dl/16e553bc0/s7eyufdq-h4i82udv-m4t3jp73-cc98jeze)


### Online demo
Try the interactive [demo](http://130.63.97.192/WB_for_srgb_rendered_images/demo.php) by uploading your photo or paste a URL for a photo from the web.


### Working with videos
You can use the provided code to process video frames separately (some flickering may occur as it does not consider temporal coherence in processing).

https://user-images.githubusercontent.com/37669469/125736626-dbcebab6-5c1d-4873-b081-640172c094ab.mp4



### Project page
For more information, please visit our [project page](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/index.html)

### Commercial Use
This software and the dataset are provided for research purposes only. A license must be obtained for any commercial application.


### Related Research Projects
- [White-Balance Augmenter](https://github.com/mahmoudnafifi/WB_color_augmenter): An augmentation technique based on camera WB errors (ICCV 2019).
- [Deep White-Balance Editing](https://github.com/mahmoudnafifi/Deep_White_Balance): A multi-task deep learning model for post-capture white-balance correction and editing (CVPR 2020).
- [Interactive White Balancing](https://github.com/mahmoudnafifi/Interactive_WB_correction):A simple method to link the nonlinear white-balance correction to the user's selected colors to allow interactive white-balance manipulation (CIC 2020).
