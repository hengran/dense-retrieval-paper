# dense-retrieval
## encoder-only
1. Hard-negative mining methods:
   1. Dense Passage Retrieval for Open-Domain Question Answering
   2. Approximate nearest neighbor negative contrastive learning for dense text retrieval.
   3. RocketQA: An Optimized Training Approach to Dense Passage Retrieval for Open-Domain Question Answering
   4. rocketqav2: a joint training method for dense passage retrieval and passage re-ranking
   5. Optimizing Dense Retrieval Model Training with Hard Negatives
   6. Conan-embedding: General Text Embedding with More and Better Negative Samples (2024.08) [paper link](https://arxiv.org/abs/2408.15710)
3. Interaction  
   1. D-q:
      1. DRPQ: Improving Document Representations by Generating Pseudo Query Embeddings for Dense Retrieval
      2. I3 Retriever: Incorporating Implicit Interaction in Pre-trained Language Models for Passage Retrieval
      3. DCE: Learning Diverse Document Representations with Deep Query Interactions for Dense Retrieval
   2. q-D  
      1. Improving Query Representations for Dense Retrieval with Pseudo Relevance Feedback
5. pre-train methods:
   1. Auto-encoding:
       1. Less is More: Pre-train a Strong Text Encoder for Dense Retrieval Using a Weak Decoder
       2. Retromae: Pre-training retrieval-oriented transformers via masked autoencoder
       3. ConTextual Masked Auto-Encoder for Dense Passage Retrieval
       4. SimLM: Pre-training with Representation Bottleneck for Dense Passage Retrieval
       5. MASTER: Multi-task Pre-trained Bottlenecked Masked Autoencoders are Better Dense Retrievers
   2. Transformers:
      1. Condenser: a Pre-training Architecture for Dense Retrieval
   3. Representative Words Prediction
      1. PROP: Pre-training with Representative Words Prediction for Ad-hoc Retrieval
      2. B-PROP: Bootstrapped Pre-training with Representative Words Prediction for Ad-hoc Retrieval.



# LLMs coming
***

## LLMs help retriever
1. W-RAG: Weakly Supervised Dense Retrieval in RAG for Open-domain Question Answering (2024) [paper link](https://arxiv.org/pdf/2408.08444)
2. REPLUG: Retrieval-Augmented Black-Box Language Models (NAACL 2024) [paper link](https://aclanthology.org/2024.naacl-long.463.pdf)
    **key point**: Generation helps retrieval.
3. A Case Study of Enhancing Sparse Retrieval using LLMs (WWW'24) [paper](https://dl.acm.org/doi/pdf/10.1145/3589335.3651945)

     
## LLMs for IR post-processing
1. From the perspective of cognition：
   1. Are Large Language Models Good at Utility Judgments? (SIGIR 2024)
   2. Iterative Utility Judgment Framework via LLMs Inspired by Relevance in Philosophy (2024)
      
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
4. Improving Zero-shot LLM Re-Ranker with Risk Minimization (Arxiv'24) [paper](https://arxiv.org/pdf/2406.13331)
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
9. Leveraging Passage Embeddings for Efficient Listwise Reranking with Large Language Models(Arxiv'24) [paper](https://arxiv.org/pdf/2406.14848)
   1. Pointwise
   2. zero-shot
10. PaRaDe: Passage Ranking using Demonstrations with Large Language Models(Findings of EMNLP'23) [paper](https://aclanthology.org/2023.findings-emnlp.950.pdf)
11. APEER : Automatic Prompt Engineering Enhances Large Language Model Reranking(Arxiv'24) [paper](https://arxiv.org/pdf/2406.14449)
12. Is ChatGPT Good at Search? Investigating Large Language Models as Re-Ranking Agents(EMNLP'23) [paper](https://arxiv.org/pdf/2304.09542)
13. Are Large Language Models Good at Utility Judgments? (SIGIR'24) [paper](https://arxiv.org/pdf/2403.19216)
      

   
