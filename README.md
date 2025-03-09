# Awesome LLM Resources

*A curated list of essential papers, codebases, and tools for mastering Large Language Models (LLMs).  
Inspired by Sebastian Raschka’s book [*Build a Large Language Model From Scratch*](https://www.manning.com/books/build-a-large-language-model-from-scratch).*

---


## Part 1: Foundational Research
1. BloombergGPT: A Large Language Model for Finance (2023) by Wu et al., [Link](https://arxiv.org/abs/2303.17564)
2. Towards Expert-Level Medical Question Answering with Large
Language Models (2023) by Singhal et al., [Link](https://arxiv.org/abs/2305.09617)
3. Attention Is All You Need (2017) by Vaswani et al., [Link](https://arxiv.org/abs/1706.03762)
4. BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding (2018) by Devlin et al., [Link](https://arxiv.org/abs/1810.04805)
5. Language Models are Few-Shot Learners (2020) by Brown et al., [Link](https://arxiv.org/abs/2005.14165)
6. An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale (2020) by Dosovitskiy et al., [Link](https://arxiv.org/abs/2010.11929)
7. RWKV: Reinventing RNNs for the Transformer Era (2023) by Peng et
al., [Link](https://arxiv.org/abs/2305.13048)
8. Hyena Hierarchy: Towards Larger Convolutional Language Models
(2023) by Poli et al., [Link](https://arxiv.org/abs/2302.10866)
9. Mamba: Linear-Time Sequence Modeling with Selective State Spaces
(2023) by Gu and Dao, [Link](https://arxiv.org/abs/2312.00752)
10. Llama 2: Open Foundation and Fine-Tuned Chat Models (2023) by
Touvron et al., [Link](https://arxiv.org/abs/2307.092881)
11. The Pile: An 800GB Dataset of Diverse Text for Language Modeling
(2020) by Gao et al., [Link](https://arxiv.org/abs/2101.00027)
12. Training Language Models to Follow Instructions with Human
Feedback (2022) by Ouyang et al., [Link](https://arxiv.org/abs/2203.02155)


## Part 2: Tokenization & Subword Techniques
13. Machine Learning Q and AI (2023) by Sebastian Raschka, [Link](https://leanpub.com/machine-learning-q-and-ai)
14. Neural Machine Translation of Rare Words with Subword Units (2015)
by Sennrich at al., [Link](https://arxiv.org/abs/1508.07909)
15. Code for the Byte pair encoding tokenizer used to train GPT-2 open-sourced by OpenAI, [Link](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
16. Interactive web UI to illustrate how the byte pair tokenizer in GPT model works, [Link](https://platform.openai.com/tokenizer)
17. A minimal implementation of a BPE tokenizer by Andrej Karpathy, [Link](https://github.com/karpathy/minbpe)
18. SentencePiece: A Simple and Language Independent Subword
Tokenizer and Detokenizer for Neural Text Processing (2018) by Kudo
and Richardson, [Link](https://aclanthology.org/D18-2012/)
19. Fast WordPiece Tokenization (2020) by Song et al., [Link](https://arxiv.org/abs/2012.15524)


## Part 3: Efficient Attention & Architecture
20. Neural Machine Translation by Jointly Learning to Align and Translate (2014) by Bahdanau, Cho, and Bengio, [Link](https://arxiv.org/abs/1409.0473)
21. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-
Awareness (2022) by Dao et al., [Link](https://arxiv.org/abs/2205.14135)
22. FlashAttention-2: Faster Attention with Better Parallelism and Work
Partitioning (2023) by Dao,, [Link](https://arxiv.org/abs/2307.08691)
23. PyTorch implementation of self-attention and causal attention that supports FlashAttention for efficiency, [Link](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
24. Dropout: A Simple Way to Prevent Neural Networks from Overfitting
(2014) by Srivastava et al., [Link](https://jmlr.org/papers/v15/srivastava14a.html)
25. Simplifying Transformer Blocks (2023) by He and Hofmann, [Link](https://arxiv.org/abs/2311.01906)


## Part 4: Normalization & GPT Models
26. Layer Normalization (2016) by Ba, Kiros, and Hinton, [Link](https://arxiv.org/abs/1607.06450)
27. On Layer Normalization in the Transformer Architecture (2020) by
Xiong et al., [Link](https://arxiv.org/abs/2002.04745)
28. ResiDual: Transformer with Dual Residual Connections (2023) by Tie
et al., [Link](https://arxiv.org/abs/2304.14802)
29. Root Mean Square Layer Normalization (2019) by Zhang and Sennrich, [Link](https://arxiv.org/abs/1910.07467)
28. Gaussian Error Linear Units (GELUs) (2016) by Hendricks and
Gimpel, [Link](https://arxiv.org/abs/1606.08415)
29. Language Models are Unsupervised Multitask Learners (2019) by
Radford et al., [Link](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
30. Language Models are Few-Shot Learners (2023) by Brown et al., [Link](https://arxiv.org/abs/2005.14165)
31. OpenAI's GPT-3 Language Model: A Technical Overview,, [Link](https://lambdalabs.com/blog/demystifying-gpt-3)
32. NanoGPT, a repository for training medium-sized GPTs, [Link](https://github.com/karpathy/nanoGPT)
33. "In the long (context) run" by Harm de Vries, [Link](https://www.harmdevries.com/post/context-length/)


## Part 5: Training Strategies & Datasets
34. L8.2 Logistic Regression Loss Function, [Link](https://www.youtube.com/watch?v=GxJe0DZvydM)
35. Pythia: A Suite for Analyzing Large Language Models Across Training
and Scaling (2023) by Biderman et al., [Link](https://arxiv.org/abs/2304.01373)
36. OLMo: Accelerating the Science of Language Models (2024) by
Groeneveld et al. [Link](https://arxiv.org/abs/2402.00838)
37. Simple and Scalable Strategies to Continually Pre-train Large Language
Models (2024) by Ibrahim et al., [Link](https://arxiv.org/abs/2403.08763)
38. GaLore: Memory-Efficient LLM Training by Gradient Low-Rank
Projection (2024) by Zhao et al., [Link](https://arxiv.org/abs/2403.03507)
39. GaLore code repository, [Link](https://github.com/jiaweizzhao/GaLore)
40. Dolma: an Open Corpus of Three Trillion Tokens for LLM Pretraining
Research by Soldaini et al. 2024, [Link](https://arxiv.org/abs/2402.00159)
41. The RefinedWeb Dataset for Falcon LLM: Outperforming Curated
Corpora with Web Data, and Web Data Only, by Penedo et al. (2023), [Link](https://arxiv.org/abs/2306.01116)
42. RedPajama by Together AI, [Link](https://github.com/togethercomputer/RedPajama-Data)
43. Hierarchical Neural Story Generation by Fan et al. (2018), [Link](https://arxiv.org/abs/1805.04833)
44. Diverse Beam Search: Decoding Diverse Solutions from Neural
Sequence Models by Vijayakumar et al. (2016), [Link](https://arxiv.org/abs/1610.02424)

## Contributing
Contributions welcome! Feel free to submit a PR or open an issue for additional resources.  
