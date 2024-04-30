# Practical Project Proposal -  A System Translating H&E Images to IHC Images Based on Diffusion Model

**Group Member**

Xichu Yu

Maiqi Zhou

Yuchen Li

## Introduction

Breast cancer continues to be a prominent cause of global female mortality. Efficacious treatment plans and precise diagnoses pivot critically on the comprehensive evaluation of histopathological images, specifically, Hematoxylin and Eosin (H&E) and Immunohistochemistry (IHC) stained slices. To enhance our approach in this crucial area, our project proposes to develop a system capable of translating H&E images into IHC images, harnessing the capacity of diffusion models. With this endeavor, we try to improve the current narrative around breast cancer diagnosis and treatment planning.

## Related Work

Broad research advancements in the realm of breast cancer diagnostics have harnessed the utility of Hematoxylin and Eosin (H&E) and Immunohistochemistry (IHC) stained images. A study in this context includes "Breast Cancer Immunohistochemical Image Generation: a Benchmark Dataset and Challenge Review," which emphasizes the potential of transforming H&E images into IHC counterparts, optimizing HER2 expression evaluation in diagnostics .

In the field of medical imaging, analogous generative models are employed extensively to synthesize anatomical images and augment data sets for diagnostic algorithm training. For instance, studies like "Diffusion Models for Medical Image Synthesis" have demonstrated the successful replication of intricate tissue textures and staining patterns, typical of pathological images .

Leveraging insights from these pivotal works, our project centers on the use of diffusion models for the purpose of translating H&E images directly into their IHC iterations.

## Dataset

Our project proposes to harness the Breast Cancer Immunohistochemical (BCI) benchmark, aiming to synthesize IHC data from paired Hematoxylin and Eosin (H&E) stained images. The BCI dataset provides a pool of 9,746 images (4,873 pairs), partitioned into 3,896 training pairs and 977 testing pairs. This diverse collection captures various HER2 expression levels, ensuring adaptability and broad applicability of our model in breast cancer diagnostics.

## Objectives

1. Develop a diffusion model capable of generating high-fidelity IHC images.
2. Validate the accuracy of the model against existing datasets of IHC images.
3. Implement a user-friendly interface for users to utilize the generated images.

## -- Design

1. Upload H&E image.
2. Preprocess the image in backend.
3. Translate to IHC image by model inference.
4. Show the image and provide download option.

## Time Schedule

1. milestone on 14.05: Dataset prepared, software system and methodology determined.
2. milestone on 04.06: Methodology designed.
3. milestone on 11.06: Software system designed, intermediate results displayed.
4. milestone on 24.06: Draft final report handed.
5. milestone on 23.07: Final report and code handed.

## Reference

1. Zhu, C., Liu, S., Yu Z., Xu, F., Aggarwal, A., Corredor, G., Madabhushi, A., Qu, Q., Fan, H., Li, F., Li, Y., Guan, X., Zhang, Y., Singh, V. K., Akram, F., Sarker, M. M. K., Shi, Z. & Jin, M. (2023). Breast Cancer Immunohistochemical Image Generation: A Benchmark Dataset and Challenge Review. https://arxiv.org/abs/2305.03546 

2. Kazerouni, A., Aghdam, E. K., Heidari, M., Azad, R., Fayyaz, M., Hacihaliloglu, I., & Merhof, D. (2023). Diffusion models in medical imaging: A comprehensive survey. Medical Image Analysis, 88, 102846. https://doi.org/10.1016/j.media.2023.102846
