# Autoregressive Text Generation with Recurrent Neural Networks

Foundations of Generative AI (ICT4403) — Assignment 2

## Objective

This project compares four recurrent neural networks for word-level next-word prediction and text generation: a Simple RNN, a Bidirectional RNN, an LSTM, and a GRU. Each model predicts the next token from the preceding 20 tokens. The models are evaluated under the same experimental settings, then used to generate continuations from identical seed phrases at three sampling temperatures.

The implementation, plots, results, and generated samples are contained in [`FGAI_Assignment2_Text_Generation_Final.ipynb`](FGAI_Assignment2_Text_Generation_Final.ipynb).

## Dataset

The experiment uses the English **WikiText-2 raw** dataset (`wikitext-2-raw-v1`) from [Salesforce AI Research on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext). WikiText contains text from Wikipedia articles and provides separate training, validation, and test splits. The corpus was introduced by [Merity et al. (2016)](https://arxiv.org/abs/1609.07843).

Blank lines and article headings are excluded. The remaining text is lowercased and tokenized into words and punctuation. The vocabulary contains the 8,000 most frequent training tokens, including special beginning, ending, and unknown-token markers. Words outside this vocabulary are mapped to `<unk>`. The vocabulary is built from training text only.

| Split | Retained lines | Tokens after preprocessing | Selected next-word examples |
| --- | ---: | ---: | ---: |
| Train | 17,556 | 2,031,617 | 100,000 |
| Validation | 1,841 | 212,007 | 10,000 |
| Test | 2,185 | 238,286 | 10,000 |

Candidates are drawn at five-token intervals, and the stated number of examples is selected with a fixed random seed within each original split. Each example contains a 20-token input context and its immediately following target token. The reduced, reproducible sample keeps training practical while preserving a separate held-out test set.

## Models and training

All four models use a 128-dimensional embedding, a 192-dimensional output representation, dropout of 0.2, and a vocabulary prediction layer. The Bidirectional RNN uses 96 hidden units in each direction, producing a 192-dimensional combined representation. It reads both directions within the *known context only*; the target token and later text are never supplied as input.

The shared training configuration is:

| Setting | Value |
| --- | ---: |
| Context length | 20 tokens |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Maximum epochs | 8 |
| Early stopping | Validation loss, patience 2 |
| Random seed | 42 |

Cross-entropy loss is optimized with gradient clipping. The best epoch is selected using validation loss. All four models reached their best validation loss at epoch 4 and stopped after epoch 6. Test results were calculated after model selection.

## Running the notebook

The notebook was run in Google Colab with Python 3 and PyTorch. It installs the `datasets` package in its first code cell. Other libraries used are NumPy, pandas, Matplotlib, and IPython, which are available in a standard Colab environment. Internet access is needed on the first run to download WikiText-2.

1. Open [`FGAI_Assignment2_Text_Generation_Final.ipynb`](FGAI_Assignment2_Text_Generation_Final.ipynb) in Google Colab.
2. Select a Python runtime. A GPU can reduce training time; the results below were measured on a CPU.
3. Run all cells in order. The notebook loads the dataset, trains the four models, produces comparison plots, and displays all 36 generated continuations.

The saved notebook includes the outputs from the reported run. A fresh run can produce different training times depending on the runtime hardware.

## Quantitative results

Perplexity is the exponential of test cross-entropy loss; lower values indicate better next-word prediction. Accuracy is the proportion of test examples for which the highest-scoring token exactly matched the target.

| Model | Test loss ↓ | Test perplexity ↓ | Next-word accuracy ↑ | Training time, CPU | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple RNN | 5.444 | 231.26 | 18.56% | 8.07 min | 2,629,824 |
| Bidirectional RNN | 5.513 | 248.01 | 17.30% | 8.19 min | 2,611,392 |
| LSTM | 5.339 | 208.25 | **18.91%** | 15.15 min | 2,815,296 |
| GRU | **5.332** | **206.91** | 18.69% | 12.86 min | 2,753,472 |

GRU achieved the lowest test loss and perplexity. LSTM was close and had the highest exact next-word accuracy. Simple RNN trained fastest, while Bidirectional RNN had the highest perplexity in this run. Falling training loss alongside rising validation loss after epoch 4 suggests that additional training was not improving generalization.

## Sampling temperature and generated text

Each model generated 60 tokens from each of these seed phrases: `the history of the city`, `the first time the team`, and `in the early years of`. Temperatures of **0.5**, **1.0**, and **1.3** produced 36 continuations in total. The complete passages and a temperature comparison plot are displayed in the notebook.

At temperature 0.5, the models favored common phrases but often repeated wording such as “the first” or “the united states.” At 1.3, the continuations used more varied words but showed more abrupt topic changes and grammatical breaks. Temperature 1.0 gave an intermediate level of diversity, although the text still drifted from its seed topic. The distinct-1 and distinct-2 scores rose with temperature, while repeated four-token sequences became less common. These diversity measures do not by themselves measure coherence.

## Conclusion

**GRU is the best-performing model in this experiment by held-out perplexity**, with LSTM a close second and slightly better at exact next-word prediction. Simple RNN is the fastest option. The generated samples contain occasional readable fragments, but none of the models maintains a coherent topic over a long continuation. The findings are limited to one random seed, a fixed sample of WikiText-2 examples, and this training configuration.

## References

1. Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). [*Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843).
2. Salesforce AI Research. [WikiText dataset](https://huggingface.co/datasets/Salesforce/wikitext), `wikitext-2-raw-v1` configuration.
3. [PyTorch documentation](https://pytorch.org/docs/stable/nn.html).
