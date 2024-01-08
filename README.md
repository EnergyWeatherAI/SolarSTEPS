# SolarSTEPS

SolarSTEPS is a probabilistic surface solar radiation (SSR) nowcasting model based on PySTEPS (https://github.com/pySTEPS/pysteps). 
This repository contains the code for using SolarSTEPS to make predictions and the code used to compute the metrics in our paper (https://www.sciencedirect.com/science/article/pii/S030626192301139X).

We recommend to download the model and run in a Python 3.9 environment, 3.10 should work too.

To make the proababilistic forecast reproducible, the ensemble members are assigned with a seed as shown in Forecast.ipynb.

![](https://github.com/albertocarpentieri/SolarSTEPS/blob/main/SS_movie.gif)
On the left a 1-hour lead time probabilistic forecast with SolarSTEPS and on the right the correspondent satellite-derived images. The pixels represent the probability of having a clear sky (yellowish).
