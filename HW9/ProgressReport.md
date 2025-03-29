# Progress Report

### 2025/03/27
Download the dataset and the HW manual

### 2025/03/28
Went through Slide 29 through 62 of the Week 8 slide deck on Semantic Segmentation.

Understand the GAN material on Slides 60 through 81 of the Week 11 slide deck.

Gain an understanding of just the top-level ideas on Slides 125 through 165 of diffusion. 
1. Why does diffusion modeling require two Markov Chains? 
A: Diffusion modeling uses two Markov chains to first gradually destroy data and then learn to reverse that process to generate new data.

2. What is the difference between the forward q-chain and the reverse p-chain? 
A: The forward q-chain adds noise step-by-step, while the reverse p-chain learns to denoise step-by-step.

3. Why does injecting Gaussian noise make it easier to train a diffusion based data modeler? 
A: Because it turns a complex data distribution into a simple, tractable one, enabling the model to learn in small, manageable steps.

### 2025/03/29

