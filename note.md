# Exercises to do

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

First write the synch threads version before proceeding with the softmax

- You can also synch warp

Also keep in mind `logf` instead of `log` operation which goes to the GPU. You can look at the cuda math API intrinsics.

-> Single precision device intrinsics

For example if I want to use softmax or using `use_fastmath` flag

-> You can profile the python code with nsycompute too. To compare the attention and the flash attention implementation

-> With torch utils c++ extension test out attention also in python might be a nice thing to show

### Suggested step for me

Perhaps start working with data that is or allocated by you or as a tensor so that it can be used by torch with torch:extension and then do the python bindings

Rember to pin the memory if you want to show that

- Try **CuPY** to compare performances with that too and also with torch

- Compute it multiple times for an average and through away the first 3 for example for both compilation and warmup

### More suggestions

- Additionally you can use GDB with core dumps very useful to see what is happening to the program

## Additional work

Try implmeenting the different version with `fences` locks etc to do `RING` communication -> In ring attention. And try different hints also perhaps for the fence
