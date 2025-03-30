# models/model_loader.py
class VLLMModelWrapper:
    def __init__(
        self, model_name, tensor_parallel_size, dtype, max_num_seqs, sampling_params
    ):
        from vllm import LLM, SamplingParams

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(**sampling_params)
        # Save the original parameters for reference
        self.sampling_params_dict = sampling_params

    def generate(self, prompts):
        import time
        from vllm import SamplingParams
        
        # Start total time measurement
        total_start_time = time.time()
        
        # Prefilling phase - create a new SamplingParams with max_tokens=1
        prefill_params_dict = self.sampling_params_dict.copy()  # Copy the dictionary
        prefill_params_dict['max_tokens'] = 1
        prefill_params = SamplingParams(**prefill_params_dict)
        
        prefill_start_time = time.time()
        first_token_outputs = self.llm.generate(prompts, prefill_params)
        prefill_end_time = time.time()
        
        # Calculate prefilling time
        prefill_time = prefill_end_time - prefill_start_time
        
        # Decoding phase - generate the remaining tokens
        decode_start_time = time.time()
        full_outputs = self.llm.generate(prompts, self.sampling_params)
        decode_end_time = time.time()
        
        # Calculate decoding time (excluding prefilling)
        decode_time = decode_end_time - decode_start_time
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        
        # Calculate token counts
        generated_tokens = sum(len(out.outputs[0].token_ids) for out in full_outputs)
        first_tokens = sum(len(out.outputs[0].token_ids) for out in first_token_outputs)
        remaining_tokens = generated_tokens - first_tokens
        
        # Return the generated text and efficiency metrics
        efficiency_metrics = {
            "generated_tokens": generated_tokens,
            "total_time": total_time,
            "prefill_time": prefill_time,
            "decode_time": decode_time,
            "tokens_per_second": generated_tokens / total_time if total_time > 0 else 0,
            "first_token_time": prefill_time,  # Time to generate the first token
            "remaining_tokens_time": decode_time,  # Time to generate remaining tokens
            "first_token_count": first_tokens,
            "remaining_tokens_count": remaining_tokens
        }
        
        return [out.outputs[0].text for out in full_outputs], efficiency_metrics

    def predict(self, prompts, candidate_labels):
        # Currently not implemented for vLLM
        import time
        
        # Start time measurement
        start_time = time.time()
        
        # Just raise the error with timing information
        try:
            raise NotImplementedError("The predict method is not implemented for vLLM")
        finally:
            # Calculate time even though we're raising an error
            total_time = time.time() - start_time
            
            # Simple efficiency metrics for failed operation
            efficiency_metrics = {
                "total_samples": len(prompts),
                "total_time": total_time,
                "inference_time": 0,
                "samples_per_second": 0,
                "average_inference_time_per_sample": 0,
                "inference_only_samples_per_second": 0,
                "error": "Not implemented for vLLM"
            }
            
            # Re-raise with efficiency metrics
            raise NotImplementedError("predict method is not implemented for vLLM", efficiency_metrics)

# HF
class HFModelWrapper:
    def __init__(self, model, tokenizer, device, batch_size=8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def generate(self, prompts):
        import torch
        import time
        output_text = []
        batch_size = self.batch_size
        total_tokens = 0
        
        # Start total time measurement
        total_start_time = time.time()
        
        # Initialize prefilling and token counters
        total_prefill_time = 0
        total_remaining_tokens = 0
        total_first_tokens = 0
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenize input
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prefilling phase - generate only the first token
            prefill_start_time = time.time()
            with torch.no_grad():
                # Set to generate only one new token
                first_token_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prefill_end_time = time.time()
            prefill_time = prefill_end_time - prefill_start_time
            total_prefill_time += prefill_time
            
            # Count first tokens
            input_length = inputs["input_ids"].shape[1]
            batch_first_tokens = sum([1 for _ in range(len(batch_prompts))])
            total_first_tokens += batch_first_tokens
            
            # Decoding phase - generate remaining tokens
            decode_start_time = time.time()
            with torch.no_grad():
                full_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            decode_end_time = time.time()
            
            # Calculate token counts for the batch
            batch_total_tokens = 0
            for output in full_outputs:
                output_length = len(output) - input_length
                batch_total_tokens += max(0, output_length)  # Ensure we only count newly generated tokens
            
            # Calculate remaining tokens (excluding first token)
            batch_remaining_tokens = batch_total_tokens - batch_first_tokens
            total_remaining_tokens += batch_remaining_tokens
            total_tokens += batch_total_tokens
            
            # Decode the full outputs
            decoded = self.tokenizer.batch_decode(full_outputs, skip_special_tokens=True)
            batch_output_text = []
            for translation, prompt in zip(decoded, batch_prompts):
                batch_output_text.append(translation)
            output_text.extend(batch_output_text)
        
        total_time = time.time() - total_start_time
        decode_time = total_time - total_prefill_time
        
        efficiency_metrics = {
            "generated_tokens": total_tokens,
            "total_time": total_time,
            "prefill_time": total_prefill_time,
            "decode_time": decode_time,
            "tokens_per_second": total_tokens / total_time if total_time > 0 else 0,
            "first_token_time": total_prefill_time,  # Time to generate first tokens
            "remaining_tokens_time": decode_time,  # Time to generate remaining tokens
            "first_token_count": total_first_tokens,
            "remaining_tokens_count": total_remaining_tokens
        }
        
        return output_text, efficiency_metrics

    def predict(self, prompts, candidate_labels):
        # classification approach with efficiency metrics
        import torch
        import time
        predictions = []
        batch_size = self.batch_size

        # Start total time measurement
        total_start_time = time.time()
        total_inference_time = 0
        total_tokenize_time = 0
        total_samples = len(prompts)

        # Pre-tokenize candidate labels
        label_tokens = [
            self.tokenizer(lbl, add_special_tokens=False)["input_ids"]
            for lbl in candidate_labels
        ]

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            
            # Tokenization time is part of the prefilling process
            tokenize_start = time.time()
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model.config.max_position_embeddings,
            ).to(self.device)
            tokenize_end = time.time()
            batch_tokenize_time = tokenize_end - tokenize_start
            total_tokenize_time += batch_tokenize_time
            
            # Forward pass (prefilling) time
            inference_start = time.time()
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                logits = logits[:, -1, :]
            inference_end = time.time()
            
            batch_inference_time = inference_end - inference_start
            total_inference_time += batch_inference_time

            batch_predictions = []
            for j in range(len(batch_prompts)):
                label_scores = []
                for label_token_ids in label_tokens:
                    token_id = label_token_ids[0]  
                    score = logits[j, token_id].item()
                    label_scores.append(score)
                # pick max
                max_label = candidate_labels[label_scores.index(max(label_scores))]
                batch_predictions.append(max_label)
            
            predictions.extend(batch_predictions)
        
        total_time = time.time() - total_start_time
        
        # Create efficiency metrics for prediction
        # For classification tasks, inference_time is essentially the prefilling time
        efficiency_metrics = {
            "total_samples": total_samples,
            "total_time": total_time,
            "tokenize_time": total_tokenize_time,
            "inference_time": total_inference_time,
            "samples_per_second": total_samples / total_time if total_time > 0 else 0,
            "average_inference_time_per_sample": total_inference_time / total_samples if total_samples > 0 else 0,
            "inference_only_samples_per_second": total_samples / total_inference_time if total_inference_time > 0 else 0
        }
        
        return predictions, efficiency_metrics


def load_model(model_name, backend="hf", **kwargs):
    if backend == "vllm":
        from vllm import SamplingParams
        tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        dtype = kwargs.get("dtype", "auto")
        max_num_seqs = kwargs.get("max_num_seqs", 256)
        sampling_params = kwargs.get("sampling_params", {})
        model = VLLMModelWrapper(
            model_name=model_name,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            max_num_seqs=max_num_seqs,
            sampling_params=sampling_params,
        )
    elif backend == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        batch_size = kwargs.get("batch_size", 8)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.to(device)
        model.eval()

        model = HFModelWrapper(model, tokenizer, device, batch_size=batch_size)
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    return model