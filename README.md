# Contextual Embeddings Study

## Requirements
    1. TensorFlow 1.12
    2. TensorFlow Hub
    3. tqdm
    4. sklearn
    5. nltk
    6. matplotlib

## Usage
To see a specific word in interactive mode:
   
    python visualize.py --word fire
    
To plot for all words:
    
    python visualize.py
    
## Dataset

Use the training set of [SNLI](https://nlp.stanford.edu/projects/snli/), 
        
    1. pick the most frequent 200 - 400 words 
       (The top 200 words have many punctuations and pronouns, and are thus not interesting to study).
    2. extract the ELMo embeddings
    3. for each word, use kmeans to cluster its ELMo representations
    4. Visualize by transform the 1024 dimensional ELMo embeddings to 3 dimensional

## Result
    See `images` for visualized clusters for each word
    
## Examples

<p>
  <img src='images/train/2.png' height='400' width='375'/>
</p>