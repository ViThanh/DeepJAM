# Joint Alignment of Multivariate Quasi-Periodic Functional Data Using Deep Learning (DeepJAM)

## Summary
This repository contains the implementation of the DeepJAM algorithm ([Pham et al., 2023](#Pham2023)), using R implementations of keras ([Cholet et al., 2015](#Chollet2015), [Allaire & Chollet, 2022](#Allaire2022keras)) and tensorflow ([Abadi et al., 2015](#Abadi2015), [Allaire & Tang, 2022](#Allaire2022tensorflow)). Furthermore, extra packages are needed for this implementation, namely Python module `neurokit2` ([Makowski et al. 2021](#Makowski2021)) and tensorflow package `tensorflow_probability`. Refer to the [`reticulate`](https://rstudio.github.io/reticulate/articles/python_packages.html)  and `keras` vignettes for instructions on installing extra packages.

The file [Example.R](Example.R) contains a use example of the algorithm on simulated data that is placed in the [Data](Data/) subdirectory. The [Models](Models/) subdirectory contains the pre-trained neural network models that can be loaded into the environment. All the functions and libraries necessary for the algorithm can be found in the [Code](Code/) subdirectory.

## Simulations
For the example, we simulated univariate and multivariate functions, as well as electrocardiograms (ECG). We have not performed any parameter tuning for these example models, as the intention is to present a proof-of-concept and demonstrate the use of the code. When defining the model, we need to specify the number of periods observed in the data. In the following subsection, we will show what happens when the number of periods is chosen incorrectly.

### Univariate data
We simulated univariate functional data with 5 quasi-periods. The figure below shows the observed and aligned data. The thick red line is the true template, the thick black line is the cross-sectional mean of the observed data, the thick black dashed line is the cross-sectional mean of the aligned data, and the thick black dotted line is the extension of the estimated common template.

When correctly specifying the number of periods, the model aligns the data to the true template (second row). However, if we are not sure how many periods are in the data, we can simply select the number of periods to be one and still obtain good enough alignment (third row). Notice that the estimated common mean differs slightly from the common template. Finally, when choosing a number of periods that is not a divisor of the true number of periods (fourth row), the algorithm cannot find the correct template, which makes sense.
![](Results/univariate.svg)

### Multivariate data
In the multivariate case, we simulated 3-dimensional functions with 3 quasi-periods and additional amplitude variability. The figure below shows the observed and aligned data. As before, the thick red line is the true template, the thick black line is the cross-sectional mean of the observed data, the thick black dashed line is the cross-sectional mean of the aligned data, and the thick black dotted line is the extension of the estimated common template.
![](Results/multivariate.svg)

### ECG data
For the ECG data example, we simulated the data using the `neurokit2` Python module ([Makowski et al. 2021](#Makowski2021)). We simulated 12-lead ECGs and selected signals corresponding to four heartbeats. The figure below shows the observed and aligned data. Similarly to previous examples, the thick black line is the cross-sectional mean of the observed data, the thick black dashed line is the cross-sectional mean of the aligned data, and the thick black dotted line is the extension of the estimated common template.
![](Results/ECG.svg)

## References
<a id="Abadi2015"></a>
Abadi, M., Agarwal, A., Barham, P., Brevdo, E., Chen, Z., Citro, C., et al. (2015).
TensorFlow: Large-scale machine learning on heterogeneous systems. https://www.tensorflow.org/

<a id="Allaire2022keras"></a>
Allaire, J., & Chollet, F. (2022). Keras: R interface to ’keras’ [R package version
2.10.0]. https://CRAN.R-project.org/package=keras

<a id="Allaire2022tensorflow"></a>
Allaire, J., & Tang, Y. (2022). Tensorflow: R interface to ’tensorflow’ [R package
version 2.10.0]. https://CRAN.R-project.org/package=tensorflow

<a id="Chollet2015"></a>
Chollet, F., et al. (2015). Keras. https://keras.io

<a id="Makowski2021"></a>
Makowski, D., Pham, T., Lau, Z. J., Brammer, J. C., Lespinasse, F., Pham, H., et al. (2021). NeuroKit2: A python toolbox for neurophysiological signal processing. Behavior Research Methods, 53 (4), 1689–
1696. https://doi.org/10.3758/s13428-020-01516-y

<a id="Pham2023"></a>
Pham, V. T., Nielsen, J. B., Kofoed, K. F., Kühl, J. T., & Jensen, A. K. (2023). Joint alignment of multivariate quasi-periodic functional data using deep learning. https://doi.org/10.48550/arXiv.2312.09422
