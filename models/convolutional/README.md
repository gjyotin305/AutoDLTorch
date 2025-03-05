# Convolutional Models

## RoadMap

- [x] CNN
- [x] Resnet
- [ ] YoLO
- [ ] InceptionNet
- [ ] MobileNet

## Some Notes on Convolutional Models

- **Receptive Field**: It is the region of the original input that feeds into the hidden unit in the network.  
    - **Example**: Consider a convolutional network where each **convolutional kernel** has size **3**, the hidden units in the first layer take a weighted sum of the three closest inputs, so the **receptive field** is **3** → the units in the second layer take the weighted sum of the three closest inputs → **receptive field** is **5**.  
    - In this way, the **receptive field** of units in successive layers **increases**, and **information** from across the input is gradually **integrated**.  

- **Shared Weights**: A **single convolutional filter** is applied across the entire **input**, reducing the number of **parameters**.  
    - This helps **detect features** consistently across different **regions**.  

- **Translation Invariance**: CNNs can **recognize features** regardless of their **positions** in the image.  
    - **Pooling layers** further reinforce this property by **downsampling spatial dimensions**.  

- **Sparse Connectivity**: They connect only a **subset of neurons** (via **small kernels**), leading to **more efficient computations**.  
