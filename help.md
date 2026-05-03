Help on class CompiledModel in module ai_edge_litert.compiled_model:

class CompiledModel(builtins.object)
 |  CompiledModel(c_model_ptr)
 |
 |  Python wrapper for the C++ CompiledModelWrapper.
 |
 |  This class provides methods to load, inspect, and execute machine learning
 |  models using the LiteRT runtime.
 |
 |  Methods defined here:
 |
 |  __init__(self, c_model_ptr)
 |      Initializes the CompiledModel with a C++ model pointer.
 |
 |      Args:
 |        c_model_ptr: Pointer to the underlying C++ CompiledModelWrapper.
 |
 |  create_input_buffer_by_name(self, signature_key: str, input_name: str) -> ai_edge_litert.tensor_buffer.TensorBuffer
 |      Creates an input TensorBuffer for the specified signature and input name.
 |
 |      Args:
 |        signature_key: Name of the signature.
 |        input_name: Name of the input tensor.
 |
 |      Returns:
 |        A TensorBuffer object for the specified input.
 |
 |  create_input_buffers(self, signature_index: int) -> List[ai_edge_litert.tensor_buffer.TensorBuffer]
 |      Creates TensorBuffers for all inputs of the specified signature.
 |
 |      Args:
 |        signature_index: Index of the signature.
 |
 |      Returns:
 |        List of TensorBuffer objects for all inputs.
 |
 |  create_output_buffer_by_name(self, signature_key: str, output_name: str) -> ai_edge_litert.tensor_buffer.TensorBuffer
 |      Creates an output TensorBuffer for the specified signature and output name.
 |
 |      Args:
 |        signature_key: Name of the signature.
 |        output_name: Name of the output tensor.
 |
 |      Returns:
 |        A TensorBuffer object for the specified output.
 |
 |  create_output_buffers(self, signature_index: int) -> List[ai_edge_litert.tensor_buffer.TensorBuffer]
 |      Creates TensorBuffers for all outputs of the specified signature.
 |
 |      Args:
 |        signature_index: Index of the signature.
 |
 |      Returns:
 |        List of TensorBuffer objects for all outputs.
 |
 |  get_input_buffer_requirements(self, input_index: int, signature_index: int = 0) -> Dict[str, Any]
 |      Returns memory requirements for an input tensor.
 |
 |      Args:
 |        input_index: Index of the input tensor.
 |        signature_index: Index of the signature. Default is 0 (first signature).
 |
 |      Returns:
 |        Dictionary with buffer requirements (size, alignment, etc.).
 |
 |  get_num_signatures(self) -> int
 |      Returns the number of signatures in the model.
 |
 |      Returns:
 |        Number of signatures.
 |
 |  get_output_buffer_requirements(self, output_index: int, signature_index: int = 0) -> Dict[str, Any]
 |      Returns memory requirements for an output tensor.
 |
 |      Args:
 |        output_index: Index of the output tensor.
 |        signature_index: Index of the signature. Default is 0 (first signature).
 |
 |      Returns:
 |        Dictionary with buffer requirements (size, alignment, etc.).
 |
 |  get_signature_by_index(self, index: int) -> Dict[str, Any]
 |      Returns signature information for the given index.
 |
 |      Args:
 |        index: Index of the signature to retrieve.
 |
 |      Returns:
 |        Dictionary containing signature information.
 |
 |  get_signature_index(self, key: str) -> int
 |      Returns the index for a signature name.
 |
 |      Args:
 |        key: Name of the signature.
 |
 |      Returns:
 |        Index of the signature, or -1 if not found.
 |
 |  get_signature_list(self) -> Dict[str, Dict[str, List[str]]]
 |      Returns a dictionary of all available model signatures.
 |
 |      Returns:
 |        Dictionary mapping signature names to their input/output specifications.
 |
 |  run_by_index(self, signature_index: int, input_buffers: List[ai_edge_litert.tensor_buffer.TensorBuffer], output_buffers: List[ai_edge_litert.tensor_buffer.TensorBuffer]) -> None
 |      Runs inference using the indexed signature and tensor lists.
 |
 |      Args:
 |        signature_index: Index of the signature to execute.
 |        input_buffers: List of input TensorBuffer objects.
 |        output_buffers: List of output TensorBuffer objects.
 |
 |  run_by_name(self, signature_key: str, input_map: Dict[str, ai_edge_litert.tensor_buffer.TensorBuffer], output_map: Dict[str, ai_edge_litert.tensor_buffer.TensorBuffer]) -> None
 |      Runs inference using the named signature and tensor maps.
 |
 |      Args:
 |        signature_key: Name of the signature to execute.
 |        input_map: Dictionary mapping input names to TensorBuffer objects.
 |        output_map: Dictionary mapping output names to TensorBuffer objects.
 |
 |  ----------------------------------------------------------------------
 |  Class methods defined here:
 |
 |  from_buffer(model_data: bytes) -> 'CompiledModel'
 |      Creates a CompiledModel from an in-memory buffer.
 |
 |      Args:
 |        model_data: Model data as bytes.
 |
 |      Returns:
 |        A new CompiledModel instance.
 |
 |  from_file(model_path: str) -> 'CompiledModel'
 |      Creates a CompiledModel from a model file.
 |
 |      Args:
 |        model_path: Path to the model file.
 |
 |      Returns:
 |        A new CompiledModel instance.
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors defined here:
 |
 |  __dict__
 |      dictionary for instance variables
 |
 |  __weakref__
 |      list of weak references to the object

