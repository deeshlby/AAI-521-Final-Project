# AAI-521-Final-Project
### Authors: Dina Shalaby, Lishi Wang, Jeffrey Thomas
### Group Name : Group 2

# Predicting Breast Cancer in Mammograms


## I. Problem Definition


### Introduction: 

Breast cancer remains one of the most common and lethal cancers affecting women worldwide. Early detection is crucial for effective treatment and improved survival rates. Traditional methods of detection, primarily through mammography, rely heavily on the expertise of radiologists to interpret complex and nuanced images. However, even for skilled professionals, this task can be challenging. Inaccuracies in diagnosis can have significant negative impacts, not only on the course of treatment but also on the psychological well-being of patients.

In recent years, the integration of artificial intelligence (AI) into healthcare has opened new avenues for enhancing diagnostic accuracy. Specifically, the application of computer vision algorithms in interpreting mammograms presents a promising tool to supplement the limitations of human analysis. Our project, "Predicting Breast Cancer in Mammograms," aims to harness the power of AI to assist in the advancements needed for breast cancer detection.

Leveraging a comprehensive dataset from the RSNA Screening Mammography Breast Cancer Detection competition on Kaggle, which includes over 54,000 mammogram images, our project seeks to develop several deep learning models capable of accurately identifying signs of breast cancer. This endeavor is not just a technological challenge; the goal carries the potential to impact public health by improving the accuracy of breast cancer screening and diagnosis. By reducing the rates of false positives and false negatives, we aim to continue the work of improving the early detection of breast cancer, thereby increasing the chances of successful treatment and survival for women worldwide.

This report outlines our journey through this project, detailing the steps from problem definition to model development, validation, and the analysis of results. We hope our findings will offer valuable hints into the capabilities of AI in medical imaging and pave the way for more advanced, accurate, and accessible diagnostic tools in the fight against breast cancer.** \
**


### Problem Statement:

The primary problem our project addresses is the challenge of accurately detecting breast cancer in mammogram images. Mammography, while being the standard screening tool for breast cancer, presents inherent difficulties due to the complex nature of breast tissue images. The variations in breast tissue density, overlapping tissues, and subtle signs of early cancer make interpretation challenging. We aim to develop a model that can accurately identify features of mammograms presenting malignant and non-malignant growths. \



### Importance of the Problem

Breast cancer is a major health concern, with early detection being a key factor in improving survival rates. The accurate interpretation of mammograms is therefore critical. Current screening methods are not infallible, and the high rate of false positives and false negatives indicates a need for improvement. False positives can lead to unnecessary biopsies, anxiety, and physical discomfort. On the other hand, false negatives can delay the detection of cancer, reducing the chances of successful treatment. As millions of women undergo mammography annually, even a small improvement in accuracy can significantly impact patient outcomes.


### Role of Computer Vision

Computer vision algorithms can be a potentially significant tool to address these challenges. By applying deep learning neural networks, it is possible to analyze mammogram images with a level of detail and consistency challenging for human eyes. These algorithms can learn from vast amounts of data, identifying subtle patterns and indicators of breast cancer that might be overlooked in manual screening.

The role of computer vision in this context is multifaceted:



1. **Enhanced Accuracy**: AI models can potentially reduce the rate of false positives and false negatives, improving diagnostic accuracy.
2. **Consistency in Screening**: Unlike human practitioners, who may be influenced by fatigue or subjective bias, computer vision algorithms can provide consistent interpretations of mammograms.
3. **Early Detection**: By identifying subtle signs of cancer that are difficult to detect visually, AI can aid in earlier detection, leading to timely and more effective treatment.
4. **Aid to Radiologists**: These algorithms can serve as a second opinion for radiologists and oncologists, adding an extra layer of analysis to enhance confidence in diagnostic decisions.

In summary, the integration of computer vision into mammography represents a significant stride towards more accurate, reliable, and early detection of breast cancer. This technological advancement has the potential to revolutionize breast cancer screening and diagnosis, ultimately contributing to improved patient outcomes.


# II. EDA (Exploratory Data Analysis) and Pre-Processing


## Data Exploration: 

The dataset consists of 54,713 mammogram images, making it a significantly large and valuable resource for training and testing machine learning models. The large volume of data is crucial for developing robust models capable of generalizing well across different cases.

The images are provided in DICOM (Digital Imaging and Communications in Medicine) format. This is a standard format for medical imaging, which not only includes the image itself but also metadata about the patient and the imaging process.

This dataset is particularly valuable for developing AI models due to its diversity, size, and the richness of the accompanying metadata. It provides a realistic and challenging environment for testing the effectiveness of computer vision algorithms in detecting breast cancer from mammogram images.

A critical aspect to consider in our analysis of the RSNA Screening Mammography Breast Cancer Detection competition dataset is the issue of data imbalance. The percentage of positive cases is very low, approximately 2.1%.

The following page has some data visualizations.


## Pre-Processing Requirements:

The preprocessing of the RSNA Screening Mammography Breast Cancer Detection dataset was a multi-step process, critical for preparing the images for effective analysis by the classifier models. The pre-processing approach utilized a YOLO (You Only Look Once) model to assist in the extraction of the area of interest for the classifier, adapted from the first place submission to the RSNA competition. Below is a detailed description of each step involved in the pre-processing pipeline:


### **DICOM Image Extraction**

The raw data was in DICOM format, which required extraction of the image array and specific metadata for each image.

Key metadata extracted included the window center, window width, the voi-lut function, and the type of monochrome (either Monochrome1 or Monochrome2).


### Image Inversion

Depending on whether the image was Monochrome1 or Monochrome2, an image may need to be inverted. This step ensured that every image was set on a black background, instead of the potential that some were white.


### Preparation for YOLO Model Training

The images were prepared for the YOLO model, which was trained on an external dataset to identify and draw a bounding box around the breast area in each image.

This preparation involved resizing the images isotropically to maintain aspect ratio, converting them to float32 data type, and normalizing the pixel values to a range of 0 to 1.


### Bounding Box Identification and Resizing

Once the YOLO model identified the bounding box around the breast in each image, this box was resized back to match the dimensions of the original image.


### Image Cropping

The images were then cropped to the region within the bounding box. This step focused the analysis on the relevant breast area, eliminating unnecessary background and reducing computational load.


### Windowing

The images underwent a windowing process, which is critical in medical imaging for enhancing the visibility of specific tissues. The appropriate VOI LUT (Value of Interest Lookup Table) function was used based on the metadata.

In cases where the window center was not provided, a min-max scaler was employed to adjust the pixel intensities, further enhancing image contrast and detail.


### Image Stacking and Conversion

Finally, the processed images were stacked into three channels to mimic the format required for typical CNN (Convolutional Neural Network) inputs.

The images were then converted to the uint8 data type and saved as PNG files for subsequent use in the classifier modeling phase.

Each step in this preprocessing pipeline was based upon the approach of Hong Dang Nguyen and others that were higher performers in the Kaggle Competition (Nguyen, n.d.). The specific implementation was streamlined for our project, and designed to ensure that the images were optimally prepared for the deep learning models. 

The process ensured that the relevant features in the mammograms were highlighted and standardized, which is crucial for the accuracy and efficiency of the subsequent model training and analysis.


# III. Modeling Methods, Validation, and Performance Metrics


## Modeling Approach: 

We are comparing the performance of different computer vision architectures for the breast cancer problem. Since breast cancer detection does not require real time operation in a power limited device such as a cell phone, we can use larger and more complex models that can provide higher accuracy.

 

As a baseline, we are evaluating the performance of ConvNext which is a CNN approach used by the First Place winner of the Kaggle competition (Nguyen, n.d.). This is an ensemble learning implementation combining the previously outlined YOLO image preprocessing pipeline with a ConvNext classifier model. In addition, we evaluated the performance of a  vision transformer based architecture, in particular ViT (Dosovitskiy et al., 2021). Finally, another classifier implementation was proposed with UNet, a neural network structure that has been widely used in medical imaging. Some implementations were able to achieve very high accuracy in identifying malignant lesions (Soulami et al., 2021).


### ConvNext

ConvNeXt is a convolutional neural network (ConvNet) model, proposed in the paper "A ConvNet for the 2020s" (Liu et al, 2022). This model architecture was inspired by the design of Vision Transformers, though maintaining only convolutional neural nets.

The ConvNeXt model is structured into several versions with varying sizes, including Tiny, Small, Base, Large, and XLarge. These models are tailored for different computational and performance needs, with each version offering a balance between efficiency and accuracy. For this project, we ran the Small variation.


### Vision Transformers (ViT)

Transformers are a powerful deep neural network architecture that initially appeared in NLP use cases, and it is the default architecture used in NLP and LLMs now. A few years ago, several researchers demonstrated that the same idea of NLP transformers can be applied to vision tasks and can achieve high accuracy. The idea is to divide the image into patches, and treat each patch as a token. The reason vision Transformers are becoming more powerful than CNNs, is that a CNN is a local filter that learns local information. Transformers, on the other hand, have an attention mechanism that allows the model to learn global features in the image, and therefore only focus on local features that matter the most. For a problem like cancer detection, the attention mechanism can help improve the performance.

**UNET:**

UNet is an encoder-decoder neural network structured in a U shape. The encoder down-samples the image with convolutional filters to learn the features of the image. The decoder up-samples the image into a segmentation mask. It utilizes skip connections that are used to pass information directly from the encoder to the decoder at various levels of encoding resolution. UNet has been widely used in medical imaging to identify regions of interest in those images, including mammograms (Soulami et al., 2021). This implementation was decided against due to it needing either pre-prepared masks, or implementing it as an end-to-end solution.


### Framework

We are using the vision deep learning package TIMM [[Pytorch Image Models (timm) | timmdocs (fast.ai)](https://timm.fast.ai/)], which includes several pre-trained vision CNN and Transformer models. 

For the rest of the data pipeline, we followed a similar approach to the First Place Winner and only changed the classification network. In particular, we highlight the change we did in the following figure



Original design from the First Place winner using ConvNext

## Validation Techniques:

We used train-validation split to split the data into 80% train and 20% validation after randomization. For testing, we ran predictions on the test data, though the full test dataset from Kaggle is hidden until submission. As of the submission of this report, our submission is still processing.


## Performance Metrics:

We calculate the following metrics:

* Precision
* Recall
* Accuracy
* F1 score
* ROC-AUC

Since this is a classification problem, we need to find the optimal threshold. We designed a function for sweeping the decision threshold, and used F1 score as a metric to find the optimal threshold. The reason for using F1 score is that it is a balanced metric between false positive and false negative, and both errors are required to be minimized for such a medical use case as we explained in the Introduction section.


# IV. Modeling Results and Findings

## Comparative Analysis: 


<table>
  <tr>
   <td>
   </td>
   <td>ConvNext - Small
<p>
<em>Full Dataset</em>
<p>
35 Epochs
   </td>
   <td>MaxxViT-Nano
<p>
<em>Full Dataset</em>
<p>
13 Epochs
   </td>
  </tr>
  <tr>
   <td>Precision
   </td>
   <td>26.2%
   </td>
   <td>4.9%
   </td>
  </tr>
  <tr>
   <td>Recall
   </td>
   <td>35.8%
   </td>
   <td>23.0%
   </td>
  </tr>
  <tr>
   <td>Accuracy
   </td>
   <td>96.1%
   </td>
   <td>87.7%
   </td>
  </tr>
  <tr>
   <td>F1 Score
   </td>
   <td>30.2%
   </td>
   <td>8.1%
   </td>
  </tr>
  <tr>
   <td>AUC
   </td>
   <td><strong>82.5%</strong>
   </td>
   <td>62.7%
   </td>
  </tr>
</table>



<table>
  <tr>
   <td>
   </td>
   <td>ConvNext
<p>
<em>Balanced Subset</em>
<p>
1 Epoch
   </td>
   <td>ConvNext+ hyperpara
<p>
<em>Balanced Subset</em>
<p>
1 Epoch
   </td>
   <td> MaxxViT- Nano
<p>
<em>Balanced Subset</em>
<p>
1 Epoch
   </td>
   <td>MaxxViT- Nano+ hyperpara
<p>
<em>Balanced Subset</em>
<p>
1 Epoch
   </td>
  </tr>
  <tr>
   <td>Precision
   </td>
   <td>42.5%
   </td>
   <td>43.5%
   </td>
   <td>42.5%
   </td>
   <td>42.5%
   </td>
  </tr>
  <tr>
   <td>Recall
   </td>
   <td>100%
   </td>
   <td>100%
   </td>
   <td>100%
   </td>
   <td>100%
   </td>
  </tr>
  <tr>
   <td>Accuracy
   </td>
   <td>42.5%
   </td>
   <td>45%
   </td>
   <td>42.5%
   </td>
   <td>42.5%
   </td>
  </tr>
  <tr>
   <td>F1 Score
   </td>
   <td>59.6%
   </td>
   <td>60%
   </td>
   <td>59.6%
   </td>
   <td>59.6%
   </td>
  </tr>
  <tr>
   <td>AUC
   </td>
   <td>51.4%
   </td>
   <td>54.4%
   </td>
   <td>57.3%
   </td>
   <td>52.4%
   </td>
  </tr>
</table>


The results show that with hyperparameter optimization, the baseline ConvNext can improve 1% in precision and 3% in AUC over the baseline. The ViT model is able to achieve close to 6% improvement in AUC over the baseline ConvNext used in the First Place model. This significant improvement using ViT is an encouraging sign that vision Transformer can play a strong role in medical computer vision applications. Note that, due to training compute complexity, we used the ViT nano model. It would be expected that with ViT-large, the performance gains can be even larger, however the use of larger models caused training crashes on Colab PRo+ even with using A100 GPU.

Training the ConvNext longer and on the full dataset produced the most complete results. 


## Challenges and Differences: 

Two major challenges in this project is handling large image sizes (almost 2 mega-pixel images) which result in an overall data size of 314GB. This was not possible to store on an ordinary laptop and needed to use a server with large storage. 

The second challenge is that the image size being very large, resulted in challenges during training. Out of memory errors and crashes during training were frequent. We had to upgrade to Colab Pro+ and purchase extra compute units multiple times to be able to use the A100 GPU with 80GB system memory. These challenges limited our ability to do extensive training and also was a limitation when we tried to use larger ViT models and a hybrid ViT-base-Resnet50 model. Ideally, we would have trained on a cluster of 8 or 16 A100 with 80MB memory each, to train on a larger number of epochs.


# V. Conclusions: 
 
**Project Objectives Fulfillment: **

We were able to implement an end-to-end image preprocessing and classification model. The model was trained on the data and was able to perform the classification task.


## Future Directions: 

To advance our work and achieve optimal performance, more experimentation with model architectures and ensembles need to be explored to yield a more performant model. Continued exploration of more efficient approaches would also be beneficial.

The implementation and findings here could also serve as a starting point for exploring other kinds of imaging like CT scans, or utilizing the model to do transfer learning on images of other kinds of cancer.


# References



1. Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). A ConvNet for the 2020s. arXiv. [https://arxiv.org/abs/2201.03545](https://arxiv.org/abs/2201.03545)
2. Hugging Face. (n.d.). PyTorch Image Models. GitHub. Retrieved November 15, 2023, from [https://github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
3. Nguyen, H.D. (n.d.). kaggle_rsna_breast_cancer. GitHub. Retrieved November 15, 2023, from [https://github.com/dangnh0611/kaggle_rsna_breast_cancer](https://github.com/dangnh0611/kaggle_rsna_breast_cancer)
4. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. arXiv. [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
5. Soulami, K. B., Kaabouch, N., Saidi, M. N., & Tamtaoui, A. (2021b). Breast cancer: One-stage automated detection, segmentation, and classification of digital mammograms using unet model based-semantic segmentation. _Biomedical Signal Processing and Control_, _66_, 102481. https://doi.org/10.1016/j.bspc.2021.102481 
6. Ashwath, B. (2020, September 26). _UNet with pretrained Resnet50 encoder (pytorch)_. Kaggle. [https://www.kaggle.com/code/balraj98/unet-with-pretrained-resnet50-encoder-pytorch](https://www.kaggle.com/code/balraj98/unet-with-pretrained-resnet50-encoder-pytorch)