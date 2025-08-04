import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import sys
import os
from loadmodelinitially import blip_text_decoder, blip_vision, tokenizer_config


def load_models(model_dir="."):
    """Load ONNX models and tokenizer configuration"""
    vision_path = blip_vision
    decoder_path = blip_text_decoder
    config_path = tokenizer_config
    
    # Load vision encoder
    if not os.path.exists(vision_path):
        raise FileNotFoundError(f"Vision encoder not found: {vision_path}")
    
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    # Load ONNX model and tokenizer
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    vision_session = ort.InferenceSession(vision_path,sess_options=sess_opts, providers=providers)
    
    # Load text decoder (optional)
    decoder_session = None
    if os.path.exists(decoder_path):
        try:
            decoder_session = ort.InferenceSession(decoder_path)
        except Exception as e:
            print(f"Warning: Could not load text decoder: {e}", file=sys.stderr)
    
    # Load tokenizer config
    tokenizer_cfg = None
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            tokenizer_cfg = json.load(f)
    
    return vision_session, decoder_session, tokenizer_cfg


def preprocess_image(image_input, target_size=(384, 384)):
    """Preprocess image for BLIP vision encoder"""
    # Load image
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found: {image_input}")
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        image = image_input.convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input).convert('RGB')
    else:
        raise ValueError("Unsupported image input type")
    
    # Resize image
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    image_array = (image_array - mean) / std
    
    # Convert from HWC to CHW format
    image_array = image_array.transpose(2, 0, 1)
    
    # Add batch dimension [1, 3, H, W]
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array


def encode_image(vision_session, image_input):
    """Extract visual features from image"""
    pixel_values = preprocess_image(image_input)
    ort_inputs = {'pixel_values': pixel_values}
    image_embeds = vision_session.run(None, ort_inputs)[0]
    return image_embeds


def decode_tokens(token_ids, tokenizer_config):
    """Convert token IDs back to readable text"""
    if not tokenizer_config:
        return f"Token IDs: {token_ids}"
    
    # Create reverse vocabulary
    id_to_token = {v: k for k, v in tokenizer_config['vocab'].items()}
    
    tokens = []
    for token_id in token_ids:
        if token_id in id_to_token:
            token = id_to_token[token_id]
            # Skip special tokens and subword indicators
            if token not in ['[CLS]', '[SEP]', '[PAD]'] and not token.startswith('##'):
                tokens.append(token)
    
    # Join tokens and clean up
    text = ' '.join(tokens).strip()
    
    # Basic text cleaning
    text = text.replace(' .', '.').replace(' ,', ',').replace(' !', '!')
    text = text.replace(' ?', '?').replace(' :', ':').replace(' ;', ';')
    
    return text


def generate_caption(vision_session, decoder_session, tokenizer_config, image_input, max_length=20, beam_size=1):
    """Generate image caption"""
    # Encode image
    image_embeds = encode_image(vision_session, image_input)
    
    if not decoder_session:
        return f"Image processed successfully. Features shape: {image_embeds.shape}"
    
    if not tokenizer_config:
        return "Image encoded, but tokenizer not available for text generation"
    
    # Initialize generation
    bos_id = tokenizer_config.get('bos_token_id', 30522)
    eos_id = tokenizer_config.get('eos_token_id', 102)
    
    # Start with BOS token
    input_ids = np.array([[bos_id]], dtype=np.int64)
    generated_ids = [bos_id]
    
    # Generate tokens one by one
    for step in range(max_length):
        try:
            # Create attention mask
            attention_mask = np.ones_like(input_ids, dtype=np.int64)
            
            # Run decoder
            ort_inputs = {
                'encoder_hidden_states': image_embeds,
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
            outputs = decoder_session.run(None, ort_inputs)
            logits = outputs[0]  # [batch_size, seq_len, vocab_size]
            
            # Get probabilities for next token
            next_token_logits = logits[0, -1, :]  # Last position
            
            if beam_size == 1:
                # Greedy decoding
                next_token_id = np.argmax(next_token_logits)
            else:
                # Simple beam search (top-k)
                top_k_ids = np.argsort(next_token_logits)[-beam_size:]
                next_token_id = top_k_ids[-1]  # Take best for now
            
            next_token_id = int(next_token_id)
            
            # Check for end of sequence
            if next_token_id == eos_id:
                break
            
            # Add to generated sequence
            generated_ids.append(next_token_id)
            
            # Update input for next iteration
            input_ids = np.array([generated_ids], dtype=np.int64)
            
        except Exception as e:
            print(f"Generation failed at step {step}: {e}", file=sys.stderr)
            break
    
    # Decode generated tokens (skip BOS)
    caption = decode_tokens(generated_ids[1:], tokenizer_config)
    
    return caption



def generate_caption_preprocessed(image_path):
     # Load models
    vision_session, decoder_session, tokenizer_config = load_models()
    caption = generate_caption(vision_session, decoder_session, tokenizer_config, image_path)
    return caption        

