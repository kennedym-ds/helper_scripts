# Module 7: Generative AI

This module explores the exciting world of generative artificial intelligence, covering everything from traditional generative models to cutting-edge transformer architectures. You'll learn to build models that can create new data, from images to text to code.

## Learning Objectives

By the end of this module, you will:
- Understand the principles behind generative modeling
- Build Variational Autoencoders (VAEs) for data generation
- Create Generative Adversarial Networks (GANs) for realistic data synthesis
- Work with transformer architectures for text generation
- Implement diffusion models for image generation
- Apply generative AI to real-world creative and practical applications
- Understand the ethical implications and responsible AI practices

## Prerequisites

- Completion of Modules 0-6
- Strong understanding of deep learning and neural networks
- Familiarity with PyTorch or TensorFlow
- Knowledge of probability and statistics

## Module Contents

1. **Introduction to Generative AI** (`01_intro_to_generative_ai.ipynb`)
   - What is generative AI?
   - Discriminative vs. generative models
   - Applications and impact across industries
   - Ethical considerations and responsible AI

2. **Variational Autoencoders (VAEs)** (`02_variational_autoencoders.ipynb`)
   - Understanding latent variable models
   - Encoder-decoder architecture with probabilistic twist
   - VAE loss function: reconstruction + KL divergence
   - Implementing VAEs for image generation

3. **Generative Adversarial Networks (GANs)** (`03_generative_adversarial_networks.ipynb`)
   - The adversarial training paradigm
   - Generator and discriminator networks
   - Training dynamics and common challenges
   - Building your first GAN

4. **Advanced GAN Architectures** (`04_advanced_gan_architectures.ipynb`)
   - DCGAN for improved image generation
   - Progressive GANs for high-resolution images
   - StyleGAN and style transfer
   - Conditional GANs and controllable generation

5. **Transformer Architecture** (`05_transformer_architecture.ipynb`)
   - Attention is all you need: understanding transformers
   - Self-attention and multi-head attention
   - Positional encoding and layer normalization
   - Building transformers from scratch

6. **Language Models and Text Generation** (`06_language_models_text_generation.ipynb`)
   - From n-grams to neural language models
   - GPT architecture and autoregressive generation
   - Fine-tuning pre-trained language models
   - Text generation techniques and sampling strategies

7. **Diffusion Models** (`07_diffusion_models.ipynb`)
   - Understanding the diffusion process
   - Denoising diffusion probabilistic models (DDPMs)
   - Implementing basic diffusion models
   - Applications to image generation

8. **Multimodal Generative Models** (`08_multimodal_generative_models.ipynb`)
   - Text-to-image generation (CLIP, DALL-E concepts)
   - Image captioning and visual question answering
   - Cross-modal representation learning
   - Building simple multimodal applications

9. **Practical Generative AI Applications** (`09_practical_applications.ipynb`)
   - Creative applications: art, music, writing
   - Code generation and programming assistance
   - Data augmentation for machine learning
   - Content creation and automation

10. **Advanced Topics and Future Directions** (`10_advanced_topics_future.ipynb`)
    - Large language models (LLM) concepts
    - Reinforcement learning from human feedback (RLHF)
    - Constitutional AI and alignment
    - Emerging architectures and research directions

## Key Generative AI Concepts

### Generative Modeling Principles
- **Latent Space**: Hidden representations of data
- **Likelihood Estimation**: Modeling probability distributions
- **Sampling**: Generating new data from learned distributions
- **Mode Collapse**: When models fail to capture data diversity

### VAE Architecture
```python
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
```

### GAN Architecture
```python
class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
```

### Transformer Building Blocks
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def forward(self, x, mask=None):
        batch_size, seq_length, d_model = x.size()
        
        Q = self.W_q(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)
        output = self.W_o(attn_output)
        return output
```

## Practical Projects

### Project 1: Image Generation with VAEs
Build a VAE to generate handwritten digits:
- Implement encoder and decoder networks
- Understand the VAE loss function
- Visualize the latent space
- Generate new digit images

### Project 2: Face Generation with GANs
Create realistic faces using GANs:
- Design generator and discriminator architectures
- Implement adversarial training loop
- Handle training instabilities
- Evaluate generation quality

### Project 3: Text Generation with Transformers
Build a small language model:
- Implement transformer decoder
- Train on text corpus
- Generate coherent text samples
- Experiment with different sampling strategies

### Project 4: Simple Diffusion Model
Create a basic diffusion model for image generation:
- Understand the forward and reverse diffusion process
- Implement denoising network
- Train on simple datasets
- Generate new images through iterative denoising

### Project 5: Multimodal Application
Build a text-to-image search system:
- Use pre-trained CLIP embeddings
- Create image and text similarity metrics
- Build simple retrieval system
- Explore cross-modal understanding

## Advanced Generative AI Concepts

### Training Techniques
- **Progressive Training**: Gradually increase model complexity
- **Self-Supervised Learning**: Learning from unlabeled data
- **Few-Shot Generation**: Generating with minimal examples
- **Transfer Learning**: Adapting pre-trained generative models

### Evaluation Metrics
- **Inception Score (IS)**: Quality and diversity of generated images
- **Fréchet Inception Distance (FID)**: Distribution similarity
- **BLEU Score**: Text generation quality
- **Perplexity**: Language model evaluation

### Prompt Engineering
```python
# Examples of effective prompting strategies
def create_effective_prompts():
    examples = {
        "Creative Writing": "Write a short story about a robot who discovers emotions...",
        "Code Generation": "Create a Python function that sorts a list using quicksort algorithm:",
        "Image Description": "Describe this image in detail, focusing on colors, composition, and mood:",
        "Problem Solving": "Break down this complex problem into smaller steps:"
    }
    return examples
```

## Real-World Applications

### Creative Industries
- **Art and Design**: AI-generated artwork, logo design
- **Music Composition**: Melody and harmony generation
- **Writing and Content**: Blog posts, marketing copy, stories
- **Game Development**: Procedural content generation

### Business Applications
- **Data Augmentation**: Expanding training datasets
- **Synthetic Data**: Privacy-preserving data generation
- **Personalization**: Customized content creation
- **Automation**: Content generation pipelines

### Scientific Research
- **Drug Discovery**: Molecular generation
- **Materials Science**: New material properties
- **Climate Modeling**: Scenario generation
- **Protein Folding**: Structure prediction

## Ethical Considerations and Responsible AI

### Key Ethical Issues
- **Deepfakes and Misinformation**: Detecting and preventing misuse
- **Bias and Fairness**: Ensuring diverse and inclusive generation
- **Intellectual Property**: Understanding copyright implications
- **Privacy**: Protecting individual data in training sets

### Best Practices
- **Transparency**: Clear labeling of AI-generated content
- **Robustness**: Testing for edge cases and failures
- **Human Oversight**: Maintaining human control and review
- **Continuous Monitoring**: Ongoing evaluation of model behavior

## Assessment Checklist

- [ ] Can explain the fundamental principles of generative modeling
- [ ] Successfully implemented VAEs for data generation
- [ ] Built and trained GANs for realistic data synthesis
- [ ] Understands transformer architecture and attention mechanisms
- [ ] Created working text generation models
- [ ] Implemented basic diffusion models
- [ ] Built at least one multimodal generative application
- [ ] Understands ethical implications and responsible AI practices
- [ ] Completed 4+ practical generative AI projects

## Estimated Time

**Total Duration:** 6-7 days (18-22 hours)

## Advanced Resources

### Pre-trained Models
```python
# Using Hugging Face transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Using Stable Diffusion
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
```

### Research Papers to Explore
- "Attention Is All You Need" (Transformer architecture)
- "Generative Adversarial Networks" (Original GAN paper)
- "Auto-Encoding Variational Bayes" (VAE paper)
- "Denoising Diffusion Probabilistic Models" (DDPM paper)

## Future Directions

### Emerging Trends
- **Constitutional AI**: Building more aligned AI systems
- **Retrieval-Augmented Generation**: Combining generation with knowledge retrieval
- **Multimodal Large Models**: Unified models for text, image, and audio
- **Efficient Training**: Reducing computational requirements

### Research Opportunities
- **Controllable Generation**: Fine-grained control over generated content
- **Few-Shot Learning**: Generating with minimal examples
- **Cross-Domain Transfer**: Applying generative models across different domains
- **Interpretability**: Understanding what generative models learn

## Next Steps

After completing this module, you'll move to Module 8: Advanced Topics, where you'll learn about ensemble methods, time series analysis, MLOps, and how to deploy generative AI models in production environments.

You'll also work on a capstone project that combines multiple concepts from the entire learning series, potentially building a complete generative AI application!