## Reinforcement Learning to Avoid Dialogue Breakdown for Conversation System

### Requirement
pytorch==1.0.0
numpy
gensim
nltk
pyhocon
argparse
transformers

### Project Construction
- DApredict.py: DA prediction or estimation model
- NLI.py: NLI model for reward
- run_glue.py: NLI model trainer
- order_predict.py: dialogue order predict model
- utils.py: utility tools
- models.py: response generation model
- nn_blocks.py: neural network components
- preprocess.py: preprocess dataset
- evaluation.py: evaluate response generation model
- quantitative_evaluation.py: calc. BLEU, Distinct, ...
- train.py: response generation model trainer
- experiments.conf: config file
- data/
    - corpus/ :corpus
    - model/ :trained model
    - result/ : results


### Usage
1. train HRED model
```
python train.py --expr HRED_dd --gpu <gpu num>
```

2. reinforcement learning
```
python train.py --expr RL_dd --gpu <gpu num>
```

3. evaluation
```
python evaluation.py --expr RL_dd --gpu <gpu num>
python quantitative_evaluation --expr RL_dd
```

### Pretrained Model

https://drive.google.com/open?id=1g42PHKWzmmzUaA8vK02b7ge9TL_z6ZfK
