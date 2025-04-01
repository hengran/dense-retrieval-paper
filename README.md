# dense-retrieval
## encoder-only
1. Hard-negative mining methods:
   1. [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/pdf/2004.04906) Vladimir Karpukhin et al., 2020.09 EMNLP2020
   2. [Approximate nearest neighbor negative contrastive learning for dense text retrieval](https://arxiv.org/pdf/2007.00808). Lee Xiong et al., 2020.10 ICLR2021
   3. [RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2010.08191) Yingqi Qu et al., 2020.10 NAACL2021
   4. [rocketqav2: a joint training method for dense passage retrieval and passage re-ranking](https://arxiv.org/abs/2110.07367) Ruiyang Ren et al., 2021.10 EMNLP2021
   5. [Optimizing Dense Retrieval Model Training with Hard Negatives](https://arxiv.org/abs/2104.08051) Jingtao Zhan et al., 2021.04 SIGIR2021
   6. [Conan-embedding: General Text Embedding with More and Better Negative Samples](https://arxiv.org/abs/2408.15710) Shiyu Li et al., 2024.08
3. Interaction  
   1. D-q:
      1. [DRPQ: Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval](https://arxiv.org/abs/2105.03599) Hongyin Tang et al., 2021.03  ACL2021
      2. [I3 Retriever: Incorporating Implicit Interaction in Pre-trained Language Models for Passage Retrieval](https://arxiv.org/abs/2306.02371) Qian Dong et al., 2023.07   CIKM2023
      3. [DCE: Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval](https://arxiv.org/abs/2208.04232) Zehan Li et al., 2022.08
      4. [CAPSTONE:Curriculum Sampling for Dense Retrieval with Document Expansion](https://aclanthology.org/2023.emnlp-main.651.pdf) Xingwei He et al., EMNLP2023
   2. q-D  
      1. [Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback](https://arxiv.org/abs/2108.13454) HongChien Yu et al., 2021.08 CIKM2021
3. Multi-vector
   1. [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/pdf/2004.12832)  Omar KhaŠab et al., 2020.07 SIGIR2020
     
5. pre-train methods:
   1. Auto-encoding:
       1. [Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder](https://arxiv.org/abs/2102.09206) Shuqi Lu et al., 2021.02 EMNLP2021
       2. [Retromae: Pre-training retrieval-oriented transformers via masked autoencoder](https://arxiv.org/abs/2205.12035) Shitao Xiao et al., 2022.03, EMNLP2022
       3. [ConTextual Masked Auto-Encoder for Dense Passage Retrieval](https://arxiv.org/abs/2208.07670) Xing Wu et al., 2022.08 AAAI2023
       4. [SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval](https://aclanthology.org/2023.acl-long.125/) Liang Wang et al., ACL2023
       5. [MASTER: Multi-task Pre-trained Bottlenecked Masked Autoencoders are Better Dense Retrievers](https://arxiv.org/abs/2212.07841) Kun Zhou et al., 2022.12 ECML-PKDD 2023
   2. Transformers:
      1. [Condenser: a Pre-training Architecture for Dense Retrieval](https://arxiv.org/abs/2104.08253) Luyu Gao et al., 2021.04 EMNLP2021
   3. Representative Words Prediction
      1. [PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval](https://arxiv.org/abs/2010.10137) Xinyu Ma et al., WSDM2021, 2020.10
      2. [B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval](https://arxiv.org/abs/2104.09791) SIGIR2021 2021.04
   3. Others:
      1. [How to Train Your DRAGON: Diverse Augmentation Towards Generalizable Dense Retrieval](https://arxiv.org/pdf/2302.07452) 2023.02
      2. [M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/pdf/2402.03216)  Jianlv Chen et al., 2024 07



# LLMs coming
***

## LLMs help retriever
1. W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering (2024) [paper link](https://arxiv.org/pdf/2408.08444)
2. REPLUG: Retrieval-Augmented Black-Box Language Models (NAACL 2024) [paper link](https://aclanthology.org/2024.naacl-long.463.pdf)
    **key point**: Generation helps retrieval.
3. A Case Study of Enhancing Sparse Retrieval using LLMs (WWW'24) [paper](https://dl.acm.org/doi/pdf/10.1145/3589335.3651945)

     
## LLMs for IR post-processing (relevance in retriever, and utility or usefulness in generator)
1. From the perspective of cognition：
   1. Are Large Language Models Good at Utility Judgments? (Hengran Zhang, SIGIR 2024)
   2. Iterative Utility Judgment Framework via LLMs Inspired by Relevance in Philosophy (Hengran Zhang, 2024.7)
   3. Corrective Retrieval Augmented Generation, Shi-Qi Yan, et al., Arxirv 2024
   4. Similarity is Not All You Need: Endowing Retrieval-Augmented Generation with Multi–layered Thoughts, Chunjing Gan, et al., Arxiv 2024
   5. ARKS:ActiveRetrieval in Knowledge Soup for Code Generation, Hongjin Su, et al., 2024
   6. CONTEXT-AUGMENTED CODE GENERATION USING PROGRAMMING KNOWLEDGE GRAPHS,  Iman Saberi, et.al., 2024.10.9
   7. Evaluating Retrieval Quality in Retrieval-Augmented Generation.  Alireza Salem, et al., SIGIR2024
   8. Bridging the Preference Gap between Retrievers and LLMs. Zixuan Ke et al. ACL2024
      
### LLMs in dense retrieval (LLMs as encoder)
1. Fine-tuning LLaMa for Multi-stage Text Retrieval(SIGIR 2024)
   1. add eos token
   2. use the eos hidden states to embed whole sentence
3. Making Large Language Models a Better Foundation For Dense Retrieval(2023)-> **Llama2Vec: Unsupervised Adaptation of Large Language Models for
Dense Retrieval (ACL 2024)**
   1. first work on pre-training for dense retrieval using LLMs
   2. motivation: As a result, the LLMs’ output embeddings will mainly focus on capturing the local and near-future semantic of the context. However, dense retrieval calls for embeddings to represent the global semantic of the entire context.
4. Improving Text Embeddings with Large Language Models(2023) [paper link](https://arxiv.org/pdf/2401.00368)
   1. E5-mistral-7B
   2. fine-tuning on both the generated synthetic data and a collection of 13 public datasets.
5. NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models(2024) [paper link](https://arxiv.org/pdf/2405.17428)
   1. Two stage fine-tuning task:
        1. It first applies contrastive training with instructions on retrieval datasets, utilizing in-batch negatives and curated hard negative examples.
        2. At stage-2, it blends various non-retrieval datasets into instruction tuning, which not only enhances non-retrieval task accuracy but also improves retrieval performance.
6. Repetition Improves Language Model Embeddings(2023) [paper link](https://arxiv.org/pdf/2402.15449)
   1. input the query or passage twice.
   2. improve the embedding of the last token.
7. LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders(2024) [paper link](https://arxiv.org/pdf/2404.05961)
8. NV-Retriever: Improving text embedding models with effective hard-negative mining [paper link](https://arxiv.org/pdf/2407.15831)
9. ScalingNote: Scaling up Retrievers with Large Language Models for Real-World Dense Retrieval, Suyuan Huang, et al.,  [paper link](https://arxiv.org/pdf/2411.15766)

# re-Ranking


## LLMs for reranking
1. A Setwise Approach for Effective and Highly Efficient Zero-shot Ranking with Large Language Models(SIGIR 2024) [paper](https://dl.acm.org/doi/pdf/10.1145/3626772.3657813)
      1. Listwise
      2. Zero-shot
2. PRP-Graph: Pairwise Ranking Prompting to LLMs with Graph Aggregation for Effective Text Re-ranking(ACL 2024) [paper](https://aclanthology.org/2024.acl-long.313.pdf)
      1. Pairwise
      2. Zero-shot
3. RankVicuna: Zero-Shot Listwise Document Reranking with Open-Source Large Language Models (Arxiv'23) [paper](https://arxiv.org/pdf/2309.15088)
      1. Listwise
      2. Zero-shot
4. Improving Zero-shot LLM Re-Ranker with Risk Minimization (EMNLP'2024) [paper](https://arxiv.org/pdf/2406.13331)
   1. Pointwise
   2. Zero-shot
5. RankRAG: Unifying Context Ranking with Retrieval-Augmented Generation in LLMs (Arxiv'24) [paper](https://arxiv.org/pdf/2407.02485v1)
   1. Listwise
   2. Instruct-tuning
6. Re-Ranking Step by Step: Investigating Pre-Filtering for Re-Ranking with Large Language Models (Arxiv'24) [paper](https://arxiv.org/pdf/2406.18740)
   1. Listwise
   2. zero-shot
8. Lightweight reranking for language model generations (ACL'24) [paper](https://aclanthology.org/2024.acl-long.376.pdf)
   1. Pairwise
   2. zero-shot
9. Zero-Shot Cross-Lingual Reranking with Large Language Models for Low-Resource Languages  [paper](https://aclanthology.org/2024.acl-short.59/) Mofetoluwa Adeyemi, et al., ACL'24 short
10. Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models(WWW'25) [paper](https://arxiv.org/pdf/2406.14848)
   1. Pointwise
   2. zero-shot
11. PaRaDe: Passage Ranking using Demonstrations with Large Language Models(Findings of EMNLP'23) [paper](https://aclanthology.org/2023.findings-emnlp.950.pdf)
12. APEER : Automatic Prompt Engineering Enhances Large Language Model Reranking(Arxiv'24) [paper](https://arxiv.org/pdf/2406.14449)
13. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents(EMNLP'23) [paper](https://arxiv.org/pdf/2304.09542)
14. Are Large Language Models Good at Utility Judgments? (SIGIR'24) [paper](https://arxiv.org/pdf/2403.19216)
15. JudgeRank: Leveraging Large Language Models for Reasoning-Intensive Reranking [paper](/https://arxiv.org/pdf/2411.00142)
16. FIRST: Faster Improved Listwise Reranking with Single Token Decoding [paper](https://arxiv.org/pdf/2406.15657) Revanth Gangi Reddy, et al., EMNLP2024
17. Self-Calibrated Listwise Reranking with Large Language Models [paper](https://arxiv.org/pdf/2411.04602) Ruiyang Ren, et al., WWW25
      

   
