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
            gpu_memory_utilization=0.7
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.default_sampling_params_dict = sampling_params

    def generate(self, prompts, **override_params):
        import time
        from vllm import SamplingParams
        

        final_params_dict = self.default_sampling_params_dict.copy()
        final_params_dict.update(override_params)
        final_sampling_params = SamplingParams(**final_params_dict)

        total_start_time = time.time()
        
        prefill_params_dict = final_params_dict.copy()
        prefill_params_dict['max_tokens'] = 1
        prefill_params = SamplingParams(**prefill_params_dict)
        
        prefill_start_time = time.time()
        first_token_outputs = self.llm.generate(prompts, prefill_params)
        prefill_end_time = time.time()
        
        prefill_time = prefill_end_time - prefill_start_time
        
        decode_start_time = time.time()
        final_sampling_params = SamplingParams(**final_params_dict)
        full_outputs = self.llm.generate(prompts, final_sampling_params)
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
        import time
        import torch
        from vllm import SamplingParams

        predictions = []
        total_start_time = time.time()

        # Preprocess candidate labels to get token IDs
        # Note: vLLM has an internal tokenizer, but it needs to be accessed via the model.
        tokenizer = self.llm.llm_engine.tokenizer
        label_token_ids = []
        for label in candidate_labels:
            # Get the first token ID of the label
            tokens = tokenizer.encode(label)
            if tokens:
                label_token_ids.append(tokens[0])

        # Create a special SamplingParams to generate only 1 token
        # Set temperature=0 to make it deterministic
        classify_params = SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=len(candidate_labels),  # Return logprobs for all candidate labels
        )

        # Execute generation
        outputs = self.llm.generate(prompts, classify_params)

        # Process outputs
        for output in outputs:
            # Get logprobs of the first generated token
            if output.outputs[0].logprobs:
                first_token_logprobs = output.outputs[0].logprobs[0]

                # Extract scores for each candidate label token
                label_scores = []
                for token_id in label_token_ids:
                    # vLLM's logprobs is a dictionary with token_id as the key
                    if token_id in first_token_logprobs:
                        score = first_token_logprobs[token_id].logprob
                    else:
                        score = float('-inf')  # If the token is not in top-k
                    label_scores.append(score)

                # Select the label with the highest score
                max_idx = label_scores.index(max(label_scores))
                predictions.append(candidate_labels[max_idx])
            else:
                # If no logprobs are available, fall back to the generated token
                generated_token_id = output.outputs[0].token_ids[0]
                if generated_token_id in label_token_ids:
                    idx = label_token_ids.index(generated_token_id)
                    predictions.append(candidate_labels[idx])
                else:
                    # Default to the first label
                    predictions.append(candidate_labels[0])

        total_time = time.time() - total_start_time

        efficiency_metrics = {
            "total_samples": len(prompts),
            "total_time": total_time,
            "inference_time": total_time,  # For classification tasks, total time is inference time
            "samples_per_second": len(prompts) / total_time if total_time > 0 else 0,
            "average_inference_time_per_sample": total_time / len(prompts) if len(prompts) > 0 else 0,
            "inference_only_samples_per_second": len(prompts) / total_time if total_time > 0 else 0
        }

        return predictions, efficiency_metrics

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