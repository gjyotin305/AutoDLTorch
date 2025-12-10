from transformers import AutoModelForCausalLM, AutoTokenizer
from einops import rearrange
from streaming_data import StreamingITDataLoader
import torch
# from dataclasses import dataclass
from flash_attn import flash_attn_func

DEVICE_COUNT = 1
DEVICE_TYPE_TORCH = 'cuda'
get_current_device = torch.cuda.current_device

# @dataclass
# class CausalLMStore:
#     loss: float
#     logits
#     tensor


# class RotaryEmbedding:
#     def __init__(self) -> None:
#         pass

#     def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
#         device='cpu'
#         channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
#         inv_freq = 1.0 / (base ** (channel_range / head_dim))
#         # stride the time steps
#         t = torch.arange(seq_len, dtype=torch.float32, device=device)
#         # calculate the rotation frequencies at each (time, channel) pair
#         freqs = torch.outer(t, inv_freq)
#         cos, sin = freqs.cos(), freqs.sin()
#         cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
#         cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
#         return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# def apply_rotary_emb(x, cos, sin):
#     assert x.ndim == 4  # multihead attention
#     d = x.shape[3] // 2
#     cos = rearrange(cos, 'a b c d -> a c b d')
#     sin = rearrange(sin, 'a b c d -> a c b d')
#     x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
#     x1, x2 = x1.to(cos.device), x2.to(cos.device)
#     # print(x1.shape, cos.shape)
#     y1 = x1 * cos + x2 * sin # rotate pairs of dims
#     y2 = x1 * (-sin) + x2 * cos
#     out = torch.cat([y1, y2], 3) # re-assemble
#     out = out.to(x.dtype) # ensure input/output dtypes match
#     return out

# def norm(x):
#     # Purely functional rmsnorm with no learnable params
#     return torch.nn.functional.rms_norm(x, (x.size(-1),))

def repeat_kv(hidden_states, n_rep):
    bsz, n_kv_head, s_len, head_dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_head, n_rep, s_len, head_dim)
    hidden_states = rearrange(hidden_states, 'a b c d e -> a (b c) d e')
    return hidden_states


class FastModel:
    def __init__(self, model_name) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.model.config.hidden_size // self.n_heads
        self.n_groups = self.n_heads // self.n_kv_heads
        # self.rotary_emb = rotary_emb
        # cos, sin = self.rotary_emb._precompute_rotary_embeddings(seq_len=2048, head_dim=self.head_dim)
        # self.cos = cos
        # self.sin = sin
        
    def decoder_infer(self):
        pass

    def apply_qkv(self, layer, hidden_states):
        Q = layer.self_attn.q_proj(hidden_states)
        K = layer.self_attn.k_proj(hidden_states)
        V = layer.self_attn.v_proj(hidden_states)
        return Q, K, V

    def attention_fast_forward(self, layer, hidden_states, past_key_value, position_embeddings):
        layer = layer.to('cuda')
        hidden_states = hidden_states.to('cuda')
        bsz, q_len, hidden_dim = hidden_states.size()
        input_shapes = hidden_states.shape[:-1]
        hidden_shape = (*input_shapes, -1, self.head_dim)

        Q, K, V = self.apply_qkv(layer, hidden_states)
        # Shape Q, K, V : (bsz, seq_len, hidden_dim)
        
        Q = Q.view(hidden_shape).transpose(1,2)
        K = K.view(hidden_shape).transpose(1,2)
        V = V.view(hidden_shape).transpose(1,2)

        cos, sin = position_embeddings
        
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        K = K.to('cuda')
        V = V.to('cuda')

        # Flash Attention 
        Q = Q.transpose(1, 2).to('cuda', dtype=torch.bfloat16)
        K = K.transpose(1, 2).to('cuda', dtype=torch.bfloat16)
        V = V.transpose(1, 2).to('cuda', dtype=torch.bfloat16)

        A = flash_attn_func(Q, K, V, causal=True)
        attn_output = A.reshape(bsz, q_len, self.n_heads * self.head_dim)

        attn_output = layer.self_attn.o_proj(attn_output.float())
        attn_weights = None
        return attn_output, attn_weights, past_key_value

    def decoder_fast_forward(self, layer, hidden_states, past_key_value, output_attention=True, position_embedding=None): 
        layer = layer.to('cuda')
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        hidden_states, weights, present_key_value = self.attention_fast_forward(
            layer=layer,
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            position_embeddings=position_embedding
        )
        # hidden_states = hidden_states.to()
        # residual = residual.to('cuda')
        hidden_states += residual.to('cuda')

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states += residual

        layer = layer.to('cpu')
        outputs = (hidden_states,)
        if output_attention:
            outputs += (weights,)
            outputs += (present_key_value,)

        return outputs

    def forward(self, input_ids, labels):
        input_emb = self.model.model.embed_tokens(input_ids)
        _, seq_len, _ = input_emb.size()
        hidden_states = input_emb
        past_key_value = None
        
        hidden_states = hidden_states.to('cuda')

        # T0 = 0
        position_ids = torch.arange(
            0, 0 + input_emb.shape[1], device='cuda'
        ).unsqueeze(0)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        hidden_states = hidden_states.to('cuda')
        for layer in self.model.model.layers:
            outputs = self.decoder_fast_forward(
                layer=layer,
                hidden_states=hidden_states,
                past_key_value=past_key_value,
                position_embedding=position_embeddings
            )
            hidden_states = outputs[0]
            past_key_value = outputs[-1]
            
        self.model.lm_head = self.model.lm_head.to('cuda')
        self.model.model.norm = self.model.model.norm.to('cuda') 
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        loss = None
        
        # softcap=15
        if labels is not None:
            labels = labels.to('cuda')
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            # logits = softcap * torch.tanh(logits / softcap)
            logits = logits.float()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean', ignore_index=-100)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
            'past_key_value': past_key_value
        }

    def generate(
        self,
        tokens,
        temperature: float = 0.0,
        max_tokens: int = 128
    ):  
        generated_tokens = []
        rng = None
        ids = torch.tensor([tokens], dtype=torch.long, device='cuda') # add batch dim
        for _ in range(max_tokens):
            outputs = self.forward(ids, None) # (B, T, vocab_size)
            logits = outputs['logits']
            logits = logits[:, -1, :] # (B, vocab_size)
            # if top_k is not None:
            #     v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            #     logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            generated_tokens.append(token)
        
        print(self.tokenizer.decode(generated_tokens))
        return generated_tokens

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1) 

if __name__ == "__main__":
    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
    
    dataset = StreamingITDataLoader(ds_name='tatsu-lab/alpaca', tokenizer=tokenizer)
    stream_data = dataset._return_stream_ds()
    data = None
    
    for item in stream_data:
        data = dataset.collator(item)
        break
    
    print(data['input_ids'].shape, data['labels'].shape)
    

    # tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')

    fast_model = FastModel(model_name='Qwen/Qwen2.5-3B-Instruct')

    # test = "Hello, my name is"
    # tokens = tokenizer.encode(test)
    # fast_model.model.to('cuda')

    # gen_tokens = fast_model.generate(
    #     tokens=tokens,
    #     temperature=0.8,
    #     max_tokens=20
    # )
    
    # print(tokens.shape)
    # # print(model)
    # # token_emb = 
    # # print(model.mode.embed_tokens)
    # rotary_emb = RotaryEmbedding()
    final_hidden_states = fast_model.forward(input_ids=data['input_ids'], labels=data['labels'])
    # print(final_hidden_states[0].shape)
    print(final_hidden_states)

    # model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct', dtype=torch.bfloat16)
    # model = model.to('cuda')
    # # inputs = tokenizer(test, return_tensors='pt').to('cuda')
    # # outputs = model.generate(**inputs, max_new_tokens=20)
    # # print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    # outputs = model.forward(input_ids=data['input_ids'].to('cuda'), labels=data['labels'].to('cuda'))
    # print(outputs)
    # tok_emb = fast_model.model.model.embed_tokens(tokens)
    # print(tok_emb.shape)
    # print(fast_model.model)
    # for layer in fast_model.model.model.layers:
    #     print(layer)
        # check = fast_model.decoder_fast_forward(
        #     layer=layer,
        #     hidden_states=tok_emb,
        #     past_key_value=None,
        #     position_embedding=None
        # )
        # print(check)
        # print(len(check))
        # print(check[0].shape)
        # break

    # for name, param in fast_model.model.named_parameters():
    #     print(name, param.shape, param.requires_grad)

    # # fast attention 
    # # print(model.config)
    # n_heads = model.config.num_attention_heads
    # n_kv_heads = model.config.num_key_value_heads
    # head_dim = model.config.hidden_size // n_heads
    # n_groups = n_heads // n_kv_heads
    # rotary_emb = RotaryEmbedding()
    # use_cache = False

    # for layer in model.model.layers:
    #     print(layer)
    #     bsz, q_len, _ = tok_emb.size()
    #     Q, K, V = layer.self_attn.q_proj(tok_emb), layer.self_attn.k_proj(tok_emb), layer.self_attn.v_proj(tok_emb)
    #     Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1,2)
    #     K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1,2)
    #     V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1,2)
    #     print(Q.shape, K.shape, V.shape)
    #     kv_seq_len = K.shape[-2]
        # cos, sin = rotary_emb._precompute_rotary_embeddings(seq_len=kv_seq_len, head_dim=128)
    #     # rotary_emb.extend_rope_embedding(x=V, seq_len=kv_seq_len)
    #     # cos, sin = rotary_emb.get_cached(kv_seq_len, Q.device.index)
    #     # print(cos.shape)
    #     Q, K = apply_rotary_emb(Q, cos, sin), apply_rotary_emb(K, cos, sin)
    #     past_key_value = (K, V) if use_cache else None
    #     print(K.shape, Q.shape)
    #     Q = Q.transpose(1, 2)
    #     K = K.transpose(1, 2)
    #     V = V.transpose(1, 2)
    #     Q = Q.to('cuda').bfloat16()
    #     K = K.to('cuda').bfloat16()
    #     V = V.to('cuda').bfloat16()
    #     A = flash_attn_func(Q, K, V, causal = True)
    #     # print(A.shape)
    #     attn_output = A.reshape(bsz, q_len, n_heads * head_dim)
    #     print(attn_output.shape)
    #     print(attn_output.dtype)
    #     print(attn_output)
    #     # print(la)
    #     # layer.self_attn.o_proj.to('cuda')
    #     attn_output = attn_output.to('cpu').float()
    #     attn_output = layer.self_attn.o_proj(attn_output)
    
    #     print(attn_output.shape, past_key_value)
    #     break   

    # print(tok_emb, tok_emb.shape)
    # for name, module in model.named_modules():
        # print(f"Name: {name}, Module: {module}")



