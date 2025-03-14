# ANNz+ (The readme.md is being updated at the moment.) 
ANNz+ is an upgraded version of the artificial neural networks photometric redshift estimation algorithm (ANNz) by [Collister & Lahav (2004)](https://iopscience.iop.org/article/10.1086/383254). It improves upon the original freely available software package [ANNz](https://www.homepages.ucl.ac.uk/~ucapola/annz.html) by offering better performance and the flexibility to use multiple activation functions, ensuring its continued relevance in the photo-z community. 

### Publication & Validation  
ANNz+ has been published in [Mahmud Pathi et al. (2025)](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta), where its improved performance was rigorously tested using SDSS Legacy and PAUS-GALFORM samples.  

### Key Updates in ANNz+  
- Supports multiple activation functions for flexibility.  
- Enhanced performance compared to the original ANNz.  
- Now, it is compatible with later versions of C++, ensuring smooth execution on modern systems.  

# Activation Functions  
ANNz+ supports the following activation functions:  
- **tanh**  
- **sigmoid**  
- **Leaky ReLU**  
- **ReLU**  
- **Mish**  
- **Softplus**  
- **SiLU**  

User can choose any activation function based on their preference. However, we **recommend using tanh, Leaky ReLU, or ReLU**, as they have demonstrated the best performance in [Mahmud Pathi et al. (2025)](https://iopscience.iop.org/article/10.1088/1475-7516/2025/01/097/meta).  

# Installation Instructions  

1. Download ANNz+
   
Open a terminal and run:  
```bash
git clone https://github.com/imdadmpt/ANNzPlus
