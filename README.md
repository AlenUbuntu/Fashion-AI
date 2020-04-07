# Fashion-AI 
Fashion-AI is a PyTorch code base that implements various sate-of-the-art algorithms related to fashion AI, e.g., compatibility prediction, outfit recommendation, etc.

## Install
See [INSTALL.md]()

## Papers Implemented
[1] &nbsp; **Context-Aware Visual Compatibility Prediction** </br>
CVPR, 2019, [[pdf](https://arxiv.org/abs/1902.03646)]  </br>
[2] &nbsp; **Learning Binary Code for Personalized Fashion Recommendation.** </br>
CVPR, 2019, [[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Lu_Learning_Binary_Code_for_Personalized_Fashion_Recommendation_CVPR_2019_paper.pdf)] </br>
[3] &nbsp; **DeepFashion2: A Versatile Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of Clothing Images.** </br>
CVPR, 2019, [[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Ge_DeepFashion2_A_Versatile_Benchmark_for_Detection_Pose_Estimation_Segmentation_and_CVPR_2019_paper.pdf)]

## Table of Contents
* [Compatibility Prediction / Outfit Recommendation](#Compatibility_PredictionOutfit_Recommendation) </br>
Paper [1]
* [Visual Retrieval / Personalized Fashion Recommendation](#Visual_RetrievalPersonalized_Fashion_Recommendation) </br>
Paper [2]
* [Fashion Image Object Detection/Classification/Parsing/Segmentation/Attribute Manipulation/Landmark Detection](#fashion-image-object-detectionclassificationparsingsegmentationattribute-manipulationlandmark-detection) </br>
Paper [3]

## Compatibility Prediction / Outfit Recommendation
### Context-aware Visual Compatibility Prediction
#### Performance Comparison on Polyvore
<table>
  <tr>
    <th rowspan="2">Method</th>
    <th colspan="2">Fill-In-The-Blank (FITB) Accuracy</th>
    <th colspan="2">Compatibility AUC</th>
  </tr>
  <tr>
    <td>Orig.(Easy)</td>
    <td>Res.(Hard)</td>
    <td>Orig.(Easy)</td>
    <td>Res.(Hard)</td>
  </tr>
  <tr>
    <td>Siamese Net</td>
    <td>54.2</td>
    <td>54.4</td>
    <td>0.85</td>
    <td>0.85</td>
  </tr>
  <tr>
    <td>Bi-LSTM</td>
    <td>68.6</td>
    <td>64.9</td>
    <td>0.90</td>
    <td>0.94</td>
  </tr>
  <tr>
    <td>TA-CSN</td>
    <td>86.1</td>
    <td>65.0</td>
    <td>0.98</td>
    <td>0.93</td>
  </tr>
  <tr>
    <td>Context (Orig.) (K=0)</td>
    <td>62.2</td>
    <td>47.0</td>
    <td>0.86</td>
    <td>0.76</td>
  </tr>
  <tr>
    <td>Context (Ours) (K=0)</td>
    <td>64.6</td>
    <td>48.3</td>
    <td>0.86</td>
    <td>0.77</td>
  </tr>
  <tr>
    <td>Context (Orig.) (K=3)</td>
    <td>95.9</td>
    <td>90.9</td>
    <td>0.99</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>Context (Ours) (K=3)</td>
    <td>95.1</td>
    <td>91.1</td>
    <td>0.99</td>
    <td>0.98</td>
  </tr>
  <tr>
    <td>Context (Orig.) (K=15)</td>
    <td>96.9</td>
    <td>92.7</td>
    <td>0.99</td>
    <td>0.99</td>
  </tr>
  <tr>
    <td>Context (Ours) (K=15)</td>
    <td>96.3</td>
    <td>92.7</td>
    <td>0.99</td>
    <td>0.99</td>
  </tr>
</table>

Please refer [here]() for details.
