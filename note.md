# Exercises to do

- Review `static inline` functions in header files
- Change parallelisim to be each thread in a warp to iterate over the block
- Be careful of shared memory bank conflicts
- Additioanl improvements

#### Note:

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
