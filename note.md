# Exercises to do

Review `static inline` functions in header files

### Attention

- Implementation: https://github.com/HicrestLaboratory/Open-VIT-bench
- Fast softmax: https://www.youtube.com/watch?v=IpHjDoW4ffw
- Fast matmul videos: https://www.youtube.com/@tgautam03/videos
- Very fast blogpost: https://siboehm.com/articles/22/CUDA-MMM
- Flash Attention V1 blogpost: https://gordicaleksa.medium.com/eli5-flash-attention-5c44017022ad
- [CUDA notes video](https://www.youtube.com/watch?v=86FAWCzIe_4&t=32683s) also has more discussion on matrix multiply
- [More Flash attention good video](https://www.youtube.com/watch?v=NBqHVjyDFfQ) flash attention at 1 hour 19

More videos:

- [How to write a fast Softmax kernel](https://www.youtube.com/watch?v=IpHjDoW4ffw)
- Intro to GPU programming [entire playlist ep 1](https://www.youtube.com/watch?v=c8mQYGbT310)

#### Note:

Rember to pin the memory if you want to show that

- Try **CuPY** to compare performances with that too and also with torch

## Additional work

Try implementing the different version with `fences` locks etc to do `RING` communication -> In ring attention. And try different hints also perhaps for the fence

## Testings

OLMo 2 Configs (Full MHA)

| Variant | Hidden Size | Heads (Q=K=V) | Head Dim | Layers | Seq Len |
| ------- | ----------- | ------------- | -------- | ------ | ------- |
| 7B      | 4096        | 32            | 128      | 30     | 4096    |
| 32B     | 8192        | 64            | 128      | 60     | 4096    |
| 70B     | 8192        | 64            | 128      | 80     | 4096    |
