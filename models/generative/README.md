## Generative Modeling

### GANs (Generative Adverserial Networks)

- It is a deep neural network that can learning from the training data and generate new data with the same characteristics as the training data.
- These consist of two parts mainly:
    - The generator : It is trained to produce synthetic data.
    - The discriminator : It is trained to distinguish the generators data from actual examples.

- Intuitively, the generator maps random noise through a model to produce a sample, and the discriminator decides whether the sample is real or not.

### VAE (Variational AutoEncoders)

- Motivation: The data points in our training set, may not cover the whole latent space. In the worst case scenario theoretically our autoencoder can just put all the data points in a straight line, effectively enumerating them.
Thus, if we wanted to generate an image and sampled a point from the latent space that belongs to a hole, we wont get a valid output. WE HAVE TO REGULARIZE OUR LATENT SPACE, so that the whole manifold of images is mapped to the whole latent space, preferably in a smooth way.

VAE makes sure that the latent space has a Gaussian distribution, so that by gradually moving from one point of latent space to its neighbour, we get a meaningful gradually changing output.

Loss function: $\mathbb{E}_{q(z)} \log p(\mathbf{x} | \mathbf{z}) - KL(q(\mathbf{z}) \parallel p(\mathbf{z}))$

The first term characterizes the quality of reconstruction of image from its latent representation. The second term is a regularizer term that gurantees that our guide (latent space) distribution stays relatively close to the prior p(z), which is usually chosen to be Gaussian.

