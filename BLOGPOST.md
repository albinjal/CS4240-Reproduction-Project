# Blog post for Deep Learning course

## Introduction
Here we add some motivation about why we chose this paper in the first place.

In this paragraph we explain the paper shortly -> what it does and tries to achieve.

## Building the VQGAN from Scratch (AVI)
The VQGAN model is an autoencoder that utilizes vector quantization as a bottleneck between the encoder and decoder. This quantization technique has a similar effect to reducing the size of layers in the middle of a stacked autoencoder or by having the encoder output the sufficient statistics of a normal distribution and feeding them to the decoder (as in a variational autoencoder). Quantization has an advantage over sampling in that it doesn't result in blurry output for the decoded image, unlike sampling which often produces blurry images.

Below you can see a visualization of the VQGAN.

![VQ_GAN](Images_blogpost/teaser.png)

The model consists of an encoder, a decoder, and a discriminator that differentiates real from fake images, making it a VQGAN rather than just a VQVAE. While the entire model was implemented from scratch, we will focus on a small segment of the code that handles vector quantization.
```
class Codebook(nn.Module):
    def __init__(self,args):
        super(Codebook,self).__init__()
        self.num_codebook_vectors = args.num_codebook_vectors
        self.latent_dim = args.latent_dim
        self.beta = args.beta

        self.embedding = nn.Embedding(self.num_codebook_vectors,self.latent_dim) # matrix with as rows the different embedding vectors

        # takes as input tensor with indices, output will be a tensor containing all the requested embedding vectors that corr with the indices
        self.embedding.weight.data.uniform_(-1.0/self.num_codebook_vectors,1.0/self.num_codebook_vectors) #the learnable weights of the module of shape (num_embeddings, embedding_dim) initialized uniformly now

    def forward(self,z):
        # z is normally of shape (batch_size,channels,height, width), after permutation its (batch_size, height,width,channels)
        z = z.permute(0,2,3,1).contiguous() # prepending latent vectors for finding the minimal distance to the codebook vectors
        z_flattened = z.view(-1,self.latent_dim)

        d = torch.sum(z_flattened**2,dim=1,keepdim=True)+\
            torch.sum(self.embedding.weight**2, dim=1)-\
            2*(torch.matmul(z_flattened,self.embedding.weight.t()))

        min_enc_indices = torch.argmin(d,dim=1)
        z_q = self.embedding(min_enc_indices).view(z.shape)
        loss = torch.mean((z_q.detach()-z)**2)+ self.beta * torch.mean((z_q-z.detach())**2)
        # above we first remove the gradient from the quantized latent vectors from the gradient flow and substract it from the original latent vector
        # in the second part we remove tha gradient from the original latent vector and keep the one of the quantized latent vector and substract them 

        z_q = z + (z_q-z).detach() # here we make sure that z_q has the gradient of z but keeps its quantized value
        z_q = z_q.permute(0,3,1,2) 

        return z_q, min_enc_indices, loss 
```

Here we construct a Codebook class with `self.num_codebook_vectors` number of codebook vectors. Each having dimension `self.latent_dim`. 
The codebook vectors are stored in a `nn.Embedding` layer, with `self.num_codebook_vectors` number of embeddings each of size `self.latent_dim`.

In the forward function we see that we start with a $z$ with dimension $[N,H,W,C]$, where $N$ is the batch size $H$ is the height of the image $W$ is the width of the image and $C$ is the number of channels in of the image. Which we then permute and reshape to get a $z_{flattened}$ which has the shape of $[N\cdot H \cdot W,C]$. We then calculate the euclidean distance of $z_{flattened}$ to every codebook vector and store these distances in $d$. After which we calculate the indices corresponding to the smallest distances, and store the result in `min_enc_indices`. The quantized latent vectors $z_q$ are then obtained through indexing the codebook with `self.embedding(min_enc_indices)`.

To relate what has happened up to this point to the image shown before, putting the code into more mathematical terms we've done the following:
$$z_{\mathbb{q}} = \arg \min_{z_i \in \mathcal{Z}} ||\hat{z}-z||$$
Where $\mathcal{Z}$ is the codebook. 

Now, the quantized $z_{\mathbb{q}}$ does not have the gradients of $z$. To ensure that they do have the same gradient we use a trick that is referred to as the straight through estimator: `z_q = z + (z_q-z).detach() `. We keep the value of $z_q$ equal to its original value but it has the exact same gradients as $z$. This is because $z$ gets canceled by $(z_q-z)$,  however the gradient will not be canceled because of the `.detach()` method that was called on it. 

All in all, the VQGAN model is a powerful tool for generating high quality images using vector quantization in the autoencoder architecture together with a discriminator. 
The code above is written in PyTorch, whereas the code on the github page was written in pytorch lightning so we also changed the framework.


## Running the Paper's VQGAN + Transformer on old data
Explain how to run the original code on the old dataset. Also explain why COCO and the motivation.

Show images generated.


## Running the Paper's VQGAN + Transformer on new data
Motivate why the Atari data.

Show images generated.



## FID Scores
Add table for FID scores.


## Conclusion
Conclude the paper -> they can actually make images, but we cannot recreate exact results due to computational power (lack thereof).