#import "@preview/red-agora:0.1.2": project

#show: project.with(
  title: "Tutorial: Implementing Efficient Causal Multi-Head Self-Attention for CPU and GPU",
  subtitle: "A Performance-Oriented Study of Self-Attention Kernels Across Computing Architectures",
  authors: (
    "Jacopo Zacchigna",
  ),
  school-logo: [],
  company-logo: [],
  mentors: (
    "Prof. Luca Tornatore
",
  ),
  footer-text: "DSAI",
  branch: "Advanced High Performance Computing",
  academic-year: "2025-2026",)

// Enable equation numbering and justify
#set math.equation(numbering: "(1)")
#set par(justify: true)
#show link: set text(fill: blue)

#include "sections/introduction.typ"
#include "sections/background.typ"
#include "sections/single_thread.typ"
#include "sections/multi_thread.typ"

= CUDA Implementation

In this case we will not use Tensor cores which are actually the fastest thing to compute matrix multiplication on GPU (this is because they require more complexity etc but most of all because they work with half precition to full precision output which would differe from the rest of the implementations )
Actually there are very interesting things about using tensor corses, and tensor memory which must be loaded via 4 warps in a warp-groups and are esesntially what stores the result from Tensor Core computation which instaed gets data from L1 and uses it directly without passing to regisrter and is extremly fast for certain type of opertaions like spexific matrix multplications.

Note that uccda Malloc autmatically returns 256+ byte aligned pointers 

== Some relevant notes on GPUs

*Some relevant conepts I leaned for GPU programming*: from #link("https://modal.com/gpu-glossary/device-software/warp")[this awesome site].

CPUs can also run many threads concurrently. But switches between warps happen at the speed of a single clock cycle (over 1000x faster than context switches on a CPU), again powered by the SM's Warp Schedulers . The volume of available warps and the speed of warp switches help hide latency caused by memory reads, thread synchronization, or other expensive instructions, ensuring that the arithmetic bandwidth provided by the CUDA Cores and Tensor Cores is well utilized
This latency-hiding is the secret to GPUs' strengths. CPUs seek to hide latency from end-users and programmers by maintaining large, hardware-managed caches and sophisticated instruction prediction. This extra hardware limits the fraction of their silicon area, power, and heat budgets that CPUs can allocate to computation.

Because each thread has its own private registers allocated from the register file of the SM , context switches on the GPU do not require any data movement to save or restore contexts.
And because the L1 caches on GPUs can be entirely programmer-managed and are shared between the warps scheduled together onto an SM (see cooperative thread array ), context switches on the GPU have much less impact on cache hit rates

The register file is split into 32 bit registers that can be dynamically reallocated between different data types, like 32 bit integers, 64 bit floating point numbers, and (groups of) 16 bit or smaller floating point numbers. These physical registers back the virtual registers in the Parallel Thread eXecution (PTX) intermediate representation.

RAM is generally not on the same die as the SMs , though in the latest data center-grade GPUs like the H100, it is located on a shared interposer for decreased latency and increased bandwidth . These GPUs use High-Bandwidth Memory (HBM) technology, rather than the more familiar Double Data Rate (DDR) memory in consumer GPUs and CPUs.

Instead of waiting for an instruction's results to return, when multiple warps are scheduled onto a single SM , the Warp Scheduler will select another warp to execute. This latency-hiding is how GPUs achieve high throughput and ensure work is always available for all of their cores during execution. For this reason, it is often beneficial to maximize the number of warps scheduled onto each SM , ensuring there is always an eligible warp for the SM to run
he fraction of cycles on which a warp was issued an instruction is known as the issue efficiency . The degree of concurrency in warp scheduling is known as occupancy (the ratio of the active warps to the maximum number of active warps on a device)
Performant GPU programs hide latency by interleaving the execution of many threads . This allows programs to maintain high throughput despite long instruction latencies. When one warp stalls on a slow memory operation, the GPU immediately switches to execute instructions from another eligible warp .

```nasm
LDG.E.SYS R1, [R0]        // memory load, 400 cycles
IMUL R2, R1, 0xBEEF       // integer multiply, 6 cycles
IADD R4, R2, 0xAFFE       // integer add, 4 cycles
IMUL R6, R4, 0x1337       // integer multiply, 6 cycles
```

Executed sequentially, this would take 416 cycles to complete. We can hide this latency by operating concurrently. If we assume we can issue one instruction every cycle, then, by Little's Law , if we run 416 concurrent threads , we can still finish the sequence once per cycle (on average), hiding the latency of memory from consumers of the data in R6.

Note that the equivalent of warps in other GPU programming models include _subgroups_ in WebGPU, _waves_ in DirectX, and _simdgroups_ in Metal.

A cooperative thread array (CTA) is a collection of threads scheduled onto the same Streaming Multiprocessor (SM). It is essentially what is rappresented by a block in the cuda programming model.
CTAs are the PTX /SASS implementation of the CUDA programming model 's thread blocks . CTAs are composed of one or more warps 
hreads in different CTAs cannot coordinate with each other via barriers, unlike threads within a CTA, and instead must coordinate via global memory , e.g. via atomic update instructions. Due to driver control over the scheduling of CTAs at runtime, CTA execution order is indeterminate and blocking a CTA on another CTA can easily lead to deadlock.

Shared memory is the level of the memory hierarchy corresponding to the thread block level of the thread hierarchy in the CUDA programming model . It is generally expected to be much smaller but much faster (in throughput and latency) than the global memory .

A fairly typical kernel therefore looks something like this:

- load data from global memory into shared memory
- perform a number of arithmetic operations on that data via the CUDA Cores and Tensor Cores
- optionally, synchronize threads within a thread block by means of barriers while performing those operations
- write data back into global memory, optionally preventing races across thread blocks by means of atomics

Shared memory is stored in the *L1 data cache* of the GPU's *Streaming Multiprocessor (SM)*.

When multiple threads in a warp simultaneously request memory within the same bank in shared memory but across distinct addresses, we say there is a bank conflict.

== My code ...

I initially tried malloc Managed but for some reason even though I coundn't really see it clearly from the nsyight system the results were absolutly atrocious. 
Therefore i quickly switched to a direct allocation on the gpu with CudaMalloc and CudaMemcopy.

Already here in the SASS we can see the MUFU.EX2 which is caming from the fast math ... (Check the other non fast math if it has it) and is an instruction from the GPU’s SFU (Special Function Unit

I should then explain the basic implmentation and how things are done. To then get to the magic of memory coleascing which allows for essentially (scatter-gather) load and store for warps to be extremly efficient whenever data is contigues in memory.
Note that in the case of gpu data access dooesn't have to be sequential for each thread in the warp but it just has to be conntigues and not strided (this gives huge speedups).

- v2 Added coalasced memory access

(If multiple concurrent logical accesses are serviced by a single physical burst, the access is said to be coalesced)

Since data are loaded from DRAM in burst this allows to load 32 ... at once ! 
I was told to add this: --use_fast_math
Moreover I still have some uncoaleasced memory access so I have to think how to deal with that for key_pos which would make it much faster. 
Typically, a single burst can service 128 bytes – not coincidentally, enough for each of the 32 threads in a warp to load one 32 bit float.

NOTE that we did meny optimization to explain

1. Unrolling
2. Shared memory
3. Merging the two + better unrolling levereging vector loads (4floats)

Local Memory (0.00 Inst): This is actually good. It means you have no register spilling. Your variables and the acc float4 in Step 4 are staying in registers.

Though the hit rate of L1 cache is very low L1/TEX Cache hit rate is only 37.57%.
This is likely caused by your access patterns for the Key (K) and Value (V) matrices

- Thous I should probably tile K and V but I have to be carefull of not increasaing shared memory too much becuase of  the problem I speak about later.

=== Very important note on shared memry (Bank conflicts) !
When multiple threads in a warp simultaneously request memory within the same bank in shared memory but across distinct addresses, we say there is a bank conflict

Addresses that differ by 32 × 4 = 128 bytes map to the same bank. Shared memories are roughly kilobyte scale, and so multiple addresses map onto the same bank.

If we access sequential elements of an array in shared memory, each thread in our warp will hit a different bank:
cpp

```cpp
__shared__ float data[1024];  // array in shared memory
// all 32 threads access consecutive elements of data
int tid = threadIdx.x;
float value = data[tid];  // address LSBs: 0x00, 0x04, 0x08, ...
```

All 32 accesses complete in one memory transaction because each thread hits a different bank. This is depicted on the left in the figure above.
But say we wanted our threads to access a column in a row-major shared memory array with 32 elements per row, and so we wrote:
```cpp
float value = data[tid * 32];  // address LSBs: 0x000, 0x080, 0x100 ...
// recall: floats are 4 bytes wide
```
All accesses hit the same bank, Bank 0, and so must be serialized, resulting in a 32x increase in latency, rising from on the order of ten cycles to on the order of hundreds. We could solve this bank conflict by transposing our shared memory array. For more techniques to resolve bank conflicts, see the

// NOTE:
Putting aw in Shared Memory might seem like a good idea, though it leads to  a combination of low occupancy and memory traffic bottlenecking. 
By putting aw in shared memory, shared memory requirements per block increases drastically (WARPS_PER_BLOCK * seq_len). 
If seq_len is large (e.g., 1024), each block uses 32KB of SMEM, which severely limits how many blocks can run on a single SM simultaneously

- v3 To remove the last memory access and try to get rid of this workspace we can do a nice trick which is perfoming online softmax and we can do so by ...
This gives us ... 
Moreover from this we can also use shared meomry to laod ... 

[To be completed in next sections]

= Performance Analysis and Results

[To be completed in next sections]

= Conclusion and Future Work

[To be completed in next sections]

= References
#bibliography("refs.bib")

#include "sections/appendix.typ"
