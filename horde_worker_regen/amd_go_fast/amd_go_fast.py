import torch

if "AMD" in torch.cuda.get_device_name() or "Radeon" in torch.cuda.get_device_name():
    try:
        from flash_attn import flash_attn_func

        sdpa = torch.nn.functional.scaled_dot_product_attention

        def sdpa_hijack(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if query.shape[3] <= 128 and attn_mask is None and query.dtype != torch.float32:
                hidden_states = flash_attn_func(
                    q=query.transpose(1, 2),
                    k=key.transpose(1, 2),
                    v=value.transpose(1, 2),
                    dropout_p=dropout_p,
                    causal=is_causal,
                    softmax_scale=scale,
                ).transpose(1, 2)
            else:
                hidden_states = sdpa(
                    query=query,
                    key=key,
                    value=value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                )
            return hidden_states

        torch.nn.functional.scaled_dot_product_attention = sdpa_hijack
        print("# # #\nAMD GO FAST\n# # #")
    except ImportError as e:
        print(f"# # #\nAMD GO SLOW\n{e}\n# # #")
else:
    print(f"# # #\nAMD GO SLOW\nCould not detect AMD GPU from:\n{torch.cuda.get_device_name()}\n# # #")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
