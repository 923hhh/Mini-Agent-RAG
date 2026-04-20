# Domain 100 Reranker ??

- ????`data\eval\domainrag_small_batch_100_domainrag_small_batch.jsonl`
- ?????????? `Domain 100` ???????? `RERANK_MODEL`??????????
- ?????`./data/models/bge-reranker-base`
- ?????`./data/models/bge-reranker-v2-m3`

## ????

| ?? | ?? | v2-m3 | ?? |
| --- | --- | --- | --- |
| `hit_at_k` | `0.8700` | `0.8900` | `+0.0200` |
| `mrr` | `0.7298` | `0.7418` | `+0.0120` |
| `ndcg_at_k` | `0.7317` | `0.7482` | `+0.0165` |
| `top1_hit_accuracy` | `0.6600` | `0.6700` | `+0.0100` |

## ?????

| ?? | ?? | ?? | v2-m3 | ?? |
| --- | --- | --- | --- | --- |
| `conversation_qa` | `hit_at_k` | `0.9000` | `0.9000` | `+0.0000` |
| `conversation_qa` | `mrr` | `0.8083` | `0.8167` | `+0.0083` |
| `conversation_qa` | `ndcg_at_k` | `0.8315` | `0.8381` | `+0.0065` |
| `conversation_qa` | `top1_hit_accuracy` | `0.7500` | `0.7500` | `+0.0000` |
| `extractive_qa` | `hit_at_k` | `0.9000` | `0.9500` | `+0.0500` |
| `extractive_qa` | `mrr` | `0.6142` | `0.6950` | `+0.0808` |
| `extractive_qa` | `ndcg_at_k` | `0.6839` | `0.7568` | `+0.0728` |
| `extractive_qa` | `top1_hit_accuracy` | `0.5000` | `0.6000` | `+0.1000` |
| `multi-doc_qa` | `hit_at_k` | `1.0000` | `1.0000` | `+0.0000` |
| `multi-doc_qa` | `mrr` | `0.9500` | `0.9083` | `-0.0417` |
| `multi-doc_qa` | `ndcg_at_k` | `0.7991` | `0.7808` | `-0.0183` |
| `multi-doc_qa` | `top1_hit_accuracy` | `0.9000` | `0.8500` | `-0.0500` |
| `structured_qa` | `hit_at_k` | `1.0000` | `1.0000` | `+0.0000` |
| `structured_qa` | `mrr` | `0.9750` | `1.0000` | `+0.0250` |
| `structured_qa` | `ndcg_at_k` | `0.9815` | `1.0000` | `+0.0185` |
| `structured_qa` | `top1_hit_accuracy` | `0.9500` | `1.0000` | `+0.0500` |
| `time-sensitive_qa` | `hit_at_k` | `0.5500` | `0.6000` | `+0.0500` |
| `time-sensitive_qa` | `mrr` | `0.3017` | `0.2892` | `-0.0125` |
| `time-sensitive_qa` | `ndcg_at_k` | `0.3624` | `0.3655` | `+0.0031` |
| `time-sensitive_qa` | `top1_hit_accuracy` | `0.2000` | `0.1500` | `-0.0500` |

## ??

- `bge-reranker-v2-m3` ????????????????????????????
- `Recall@5` ????????? rerank ??? Top5 ????????????
