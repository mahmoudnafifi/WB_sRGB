# When Color Constancy Goes Wrong: Correcting Improperly White-Balanced Images
*[Mahmoud Afifi](https://sites.google.com/view/mafifi)*<sup>1</sup>, *[Brian Price](https://www.brianpricephd.com/)*<sup>2</sup>, *[Scott Cohen](https://research.adobe.com/person/scott-cohen/)*<sup>2</sup>, and *[Michael S. Brown](http://www.cse.yorku.ca/~mbrown/)*<sup>1</sup>
<br></br><sup>1</sup>York University  <sup>2</sup>Adobe Research

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
  <img https://user-images.githubusercontent.com/37669469/76103283-6c526780-5f9f-11ea-9f2c-ad9d87d95fb7.gif>
</p>


#### Code/GUI parameters and options
1. `K`: Number of nearest neighbors in the KNN search (Sec. 3.4 in the paper) -- change its value to enhance the results.
2. `sigma`: The fall-off factor for KNN blending (Eq. 8 in the paper) -- change its value to enhance the results.
3. `device`: GPU or CPU (provided for Matlab version only).
4. `gamut_mapping`: Mapping pixels in-gamut either using scaling (`gamut_mapping= 1`) or clipping  (`gamut_mapping= 2`). In the paper, we used the clipping options to report our results, 
but the scaling option gives compelling results in some cases (esp., with high-saturated/vivid images). 
5. `upgraded_model` and `upgraded`: To load our upgraded model, use `upgraded_model=1` in Matlab or `upgraded=1` in Python. The upgraded model has new training examples. In our paper results, we did not use this model. However, our online [demo](http://130.63.97.192/WB_for_srgb_rendered_images/demo.php) uses it. 

### Dataset
In our paper, we mentioned that our dataset is over 65,000 images. We added two additional sets of rendered images, for a total of 105,638 rendered images. 
You can download our dataset from [here](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/dataset.html).


### Online demo
Try the interactive [demo](http://130.63.97.192/WB_for_srgb_rendered_images/demo.php) by uploading your photo or paste a URL for a photo from the web.


### Project page
For more information, please visit our [project page](http://cvil.eecs.yorku.ca/projects/public_html/sRGB_WB_correction/index.html)

This software is provided for research purposes only. A license must be obtained for any commercial application.
