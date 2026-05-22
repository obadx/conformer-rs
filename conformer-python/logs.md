
~/muaalem-tf $ ./android_aarch64_benchmark_model --num_runs=1000 --warmup_runs=5 --graph=./models/tiny_muaalem_float32.tflite 
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Min num runs: [1000]
INFO: Min warmup runs: [5]
INFO: Graph: [./models/tiny_muaalem_float32.tflite]
INFO: Signature to run: []
INFO: Loaded model ./models/tiny_muaalem_float32.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 2168 out of 2168 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions for subgraph 0.
INFO: The input model file size (MB): 24.6229
INFO: Initialized session in 119.515ms.
INFO: Running benchmark for at least 5 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
INFO: count=9 first=87112 curr=51629 min=51629 max=87112 avg=58147.4 std=11478 p5=51629 median=51962 p95=87112

INFO: Running benchmark for at least 1000 iterations and at least 1 seconds but terminate if exceeding 150 seconds.









INFO: count=1000 first=52162 curr=49785 min=49088 max=686105 avg=63743.2 std=60716 p5=49620 median=51891 p95=68628

INFO: Inference timings in us: Init: 119515, First inference: 87112, Warmup (avg): 58147.4, Inference (avg): 63743.2
INFO: Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
INFO: Memory footprint delta from the start of the tool (MB): init=55.4961 overall=55.4961
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ ./android_aarch64_benchmark_model --num_runs=1000 --warmup_runs=5 --graph=./models/tiny_muaalem_
tiny_muaalem_float32.tflite  tiny_muaalem_int4.tflite     tiny_muaalem_int8.tflite     
~/muaalem-tf $ ./android_aarch64_benchmark_model --num_runs=5 --warmup_runs=5 --graph=./models/tiny_muaalem_int8.tflite 
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Min num runs: [5]
INFO: Min warmup runs: [5]
INFO: Graph: [./models/tiny_muaalem_int8.tflite]
INFO: Signature to run: []
INFO: Loaded model ./models/tiny_muaalem_int8.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 2136 out of 2168 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 65 partitions for subgraph 0.
ERROR: tensorflow/lite/kernels/batch_matmul.cc:353 (lhs_data->type == kTfLiteFloat32 && rhs_data->type == kTfLiteInt8) || lhs_data->type == rhs_data->type was not true.
ERROR: Node number 57 (BATCH_MATMUL) failed to prepare.
ERROR: Failed to allocate tensors!
ERROR: Benchmarking failed.
~/muaalem-tf $ ./android_aarch64_benchmark_model --num_runs=5 --warmup_runs=5 --graph=./models/tiny_muaalem_int4.tflite 
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Min num runs: [5]
INFO: Min warmup runs: [5]
INFO: Graph: [./models/tiny_muaalem_int4.tflite]
INFO: Signature to run: []
INFO: Loaded model ./models/tiny_muaalem_int4.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 2136 out of 2168 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 65 partitions for subgraph 0.
INFO: The input model file size (MB): 8.45746
INFO: Initialized session in 207.799ms.
INFO: Running benchmark for at least 5 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.
INFO: count=11 first=90710 curr=42743 min=42209 max=90710 avg=48230.2 std=13845 p5=42209 median=42611 p95=90710

INFO: Running benchmark for at least 5 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
INFO: count=24 first=42468 curr=41938 min=41938 max=42840 avg=42394.8 std=188 p5=42133 median=42419 p95=42671

INFO: Inference timings in us: Init: 207799, First inference: 90710, Warmup (avg): 48230.2, Inference (avg): 42394.8
INFO: Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
INFO: Memory footprint delta from the start of the tool (MB): init=23.5938 overall=26.6797
~/muaalem-tf $ ./android_aarch64_benchmark_model --num_runs=5 --warmup_runs=50 --graph=./models/tiny_muaalem_int4.tflite 
INFO: STARTING!
INFO: Log parameter values verbosely: [0]
INFO: Min num runs: [5]
INFO: Min warmup runs: [50]
INFO: Graph: [./models/tiny_muaalem_int4.tflite]
INFO: Signature to run: []
INFO: Loaded model ./models/tiny_muaalem_int4.tflite
INFO: Initialized TensorFlow Lite runtime.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 2136 out of 2168 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 65 partitions for subgraph 0.
INFO: The input model file size (MB): 8.45746
INFO: Initialized session in 143.669ms.
INFO: Running benchmark for at least 50 iterations and at least 0.5 seconds but terminate if exceeding 150 seconds.

INFO: count=50 first=72635 curr=42240 min=42134 max=72635 avg=43536.2 std=4813 p5=42168 median=42430 p95=49447

INFO: Running benchmark for at least 5 iterations and at least 1 seconds but terminate if exceeding 150 seconds.
INFO: count=24 first=42142 curr=42232 min=42142 max=42763 avg=42435.4 std=147 p5=42232 median=42405 p95=42729

INFO: Inference timings in us: Init: 143669, First inference: 72635, Warmup (avg): 43536.2, Inference (avg): 42435.4
INFO: Note: as the benchmark tool itself affects memory footprint, the following is only APPROXIMATE to the actual memory footprint of the model at runtime. Take the information at your discretion.
INFO: Memory footprint delta from the start of the tool (MB): init=23.5625 overall=27.5859
~/muaalem-tf $ 
~/muaalem-tf $ 
~/muaalem-tf $ 

