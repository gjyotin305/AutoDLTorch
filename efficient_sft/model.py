from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from streaming_data import StreamingITDataLoader
import torch.utils.checkpoint as checkpoint
from flash_attn import flash_attn_func

DEVICE_COUNT = 1
DEVICE_TYPE_TORCH = 'cuda'
get_current_device = torch.cuda.current_device


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class QwenFastModel:
    def __init__(self, model_name, grad_ckpt: bool = False) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.model.to('cuda')
        self.grad_ckpt_check = grad_ckpt
        self.n_heads = self.model.config.num_attention_heads
        self.n_kv_heads = self.model.config.num_key_value_heads
        self.head_dim = self.model.config.hidden_size // self.n_heads
        self.n_groups = self.n_heads // self.n_kv_heads

    def apply_qkv(self, layer, hidden_states):
        Q = layer.self_attn.q_proj(hidden_states)
        K = layer.self_attn.k_proj(hidden_states)
        V = layer.self_attn.v_proj(hidden_states)
        return Q, K, V

    def attention_fast_forward(self, layer, hidden_states, position_embeddings):
        bsz, q_len, hidden_dim = hidden_states.size()
        input_shapes = hidden_states.shape[:-1]
        hidden_shape = (*input_shapes, -1, self.head_dim)

        Q, K, V = self.apply_qkv(layer, hidden_states)
        
        Q = Q.view(hidden_shape).transpose(1,2)
        K = K.view(hidden_shape).transpose(1,2)
        V = V.view(hidden_shape).transpose(1,2)

        cos, sin = position_embeddings
        
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        # Flash Attention 
        Q = Q.transpose(1, 2).to('cuda', dtype=torch.bfloat16)
        K = K.transpose(1, 2).to('cuda', dtype=torch.bfloat16)
        V = V.transpose(1, 2).to('cuda', dtype=torch.bfloat16)

        A = flash_attn_func(Q, K, V, causal=True)
        attn_output = A.reshape(bsz, q_len, self.n_heads * self.head_dim)

        attn_output = layer.self_attn.o_proj(attn_output.float())
        attn_weights = None
        return attn_output, attn_weights

    def decoder_fast_forward(self, hidden_states, layer, position_embedding, output_attention=False): 
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        
        hidden_states, weights = self.attention_fast_forward(
            layer=layer,
            hidden_states=hidden_states,
            position_embeddings=position_embedding
        )
        hidden_states += residual

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = layer.mlp(hidden_states)
        hidden_states += residual

        outputs = (hidden_states,)
        if output_attention:
            outputs += (weights.detach(),)

        return outputs

    def decoder_forward_ckpt(self, *args, **kwargs):
        def forward_fn(*inputs):
            return self.decoder_fast_forward(*inputs, **kwargs)
        return checkpoint.checkpoint(forward_fn, *args, use_reentrant=True, preserve_rng_state=False)

    def forward(self, input_ids, labels=None):
        input_emb = self.model.model.embed_tokens(input_ids.to('cuda'))
        _, seq_len, _ = input_emb.size()
        hidden_states = input_emb
        
        position_ids = torch.arange(
            0, 0 + seq_len, device='cuda'
        ).unsqueeze(0)

        position_embeddings = self.model.model.rotary_emb(hidden_states, position_ids)

        for layer in self.model.model.layers:
            if self.grad_ckpt_check:
                outputs = self.decoder_forward_ckpt(
                    layer=layer,
                    hidden_states=hidden_states,
                    position_embedding=position_embeddings
                )
                hidden_states = outputs[0]
            else:
                outputs = self.decoder_fast_forward(
                    layer=layer,
                    hidden_states=hidden_states,
                    position_embedding=position_embeddings
                )
                hidden_states = outputs[0]
           
            
        # Pre LM Head Layer Norm 
        hidden_states = self.model.model.norm(hidden_states)
        logits = self.model.lm_head(hidden_states)
        loss = None
        
        if labels is not None:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            logits = logits.float()
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction='mean', ignore_index=-100)
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': hidden_states,
        }

    def generate(
        self,
        tokens,
        temperature: float = 0.0,
        max_tokens: int = 128,
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
            if token == 151645:
                break
            # print(self.tokenizer.decode([token]))
            generated_tokens.append(token)
        
        print(self.tokenizer.decode(generated_tokens))
        return generated_tokens

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1) 

# if __name__ == "__main__":
# #  Example Usage: Run Forward Function to get logits(inference), loss(training)
# #  Run Generate Function to get sample generations
#     tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
#     dataset = StreamingITDataLoader(ds_name='tatsu-lab/alpaca', tokenizer=tokenizer)
#     stream_data = dataset._return_stream_ds()
#     model = QwenFastModel(model_name='Qwen/Qwen2.5-7B-Instruct', grad_ckpt=True)
#     for item in stream_data:
#         data = dataset.collator(item)
#         print(data['input_ids'].shape, data['labels'].shape)
#         input_ids = data['input_ids'].to('cuda')
#         out = model.forward(input_ids)
#         print(out)
#         break

