# --------------------------------------------------------
# Optional: load a standard MCAN checkpoint into this model (strict=False)
# --------------------------------------------------------

import torch


def load_checkpoint_into_net(net, ckpt_path, map_location='cpu'):
    pack = torch.load(ckpt_path, map_location=map_location)
    if isinstance(pack, dict) and 'state_dict' in pack:
        sd = pack['state_dict']
    else:
        sd = pack

    tgt = net.module if hasattr(net, 'module') else net

    ckpt_has_module = any(k.startswith('module.') for k in sd.keys())
    if ckpt_has_module:
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

    missing, unexpected = tgt.load_state_dict(sd, strict=False)
    print('[PRUNE_INIT_CKPT] Loaded with strict=False.')
    print('[PRUNE_INIT_CKPT] missing:', len(missing), 'unexpected:', len(unexpected))
    if len(missing) > 0:
        print('[PRUNE_INIT_CKPT] (sample missing)', list(missing)[:10])
    if len(unexpected) > 0:
        print('[PRUNE_INIT_CKPT] (sample unexpected)', list(unexpected)[:10])
