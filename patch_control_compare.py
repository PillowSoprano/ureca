# -*- coding: utf-8 -*-
"""
Auto-patcher for control_compare.py to support MamKO (no `.model`, uses `.net`)
- Fixes the line `model.model.to(device)` -> conditional
- Injects a small shim right after `model = load_model(method, args)` so that:
    * if no `.model`, set `.model = .net` when available
    * if `.model` has no `.enc`, provide a dummy enc that returns (x, None)
This way, the rest of your script can keep using `policy.model.enc(...)` etc.

Run: python patch_control_compare.py
"""

import io, re, sys
from pathlib import Path

TARGET = Path(__file__).with_name("control_compare.py")
if not TARGET.exists():
    print(f"[ERROR] {TARGET} not found. Put this patch file next to control_compare.py.")
    sys.exit(1)

src = TARGET.read_text(encoding="utf-8")

changed = False

# 1) Make `model.model.to(device)` conditional
pattern_to_device = re.compile(r"(\bmodel\s*\.\s*model\s*\.?\s*to\s*\(\s*device\s*\)\s*)")
if pattern_to_device.search(src):
    src = pattern_to_device.sub(
        # keep it simple and robust to surrounding code
        "(\n"
        "    # patched: only move if `.model` exists\n"
        "    model.model.to(device) if hasattr(model, 'model') else None\n"
        ")", src
    )
    changed = True

# 2) Inject shim right after `model = load_model(method, args)`
# We look for the line where model is created inside build_policy
pattern_load_model = re.compile(r"(model\s*=\s*load_model\s*\(\s*method\s*,\s*args\s*\)\s*)")
shim_block = (
    "\n"
    "    # === PATCH: MamKO compatibility shim ===\n"
    "    # If there is no `.model`, but `.net` exists (MamKO), alias it.\n"
    "    if not hasattr(model, 'model') and hasattr(model, 'net'):\n"
    "        model.model = model.net\n"
    "    # Provide a fallback `enc` if the underlying module has none:\n"
    "    if hasattr(model, 'model') and not hasattr(model.model, 'enc'):\n"
    "        def _compat_enc(xin):\n"
    "            # behave like an encoder: return latent and None (for aux)\n"
    "            return xin, None\n"
    "        # attach the dummy encoder\n"
    "        try:\n"
    "            model.model.enc = _compat_enc\n"
    "        except Exception:\n"
    "            # some modules may block attribute set; in that case we keep using raw state later\n"
    "            pass\n"
    "    # unify device record\n"
    "    if not hasattr(model, 'device'):\n"
    "        try:\n"
    "            import torch\n"
    "            model.device = next(iter(model.model.parameters())).device if hasattr(model,'model') else torch.device('cpu')\n"
    "        except Exception:\n"
    "            import torch\n"
    "            model.device = torch.device('cpu')\n"
    "    # === END PATCH ===\n"
)
if pattern_load_model.search(src):
    src = pattern_load_model.sub(r"\1" + shim_block, src, count=1)
    changed = True

# 3) Optional: make y-axis log scale in saved control plots (search a common plt call)
# Try to add plt.yscale('log') near 'control_*.png' save blocks if not already present.
def add_logscale_near_save(s):
    # find sections that save control_* png and add yscale before save if missing
    # This is heuristic but safe.
    s = re.sub(
        r"(\n\s*plt\.savefig\([^)]*control[^)]*\.png[^)]*\)\s*)",
        "\n    plt.yscale('log')\n\\1",
        s,
        flags=re.IGNORECASE
    )
    return s

before = src
src = add_logscale_near_save(src)
if src != before:
    changed = True

if not changed:
    print("[INFO] Nothing patched (file may already be compatible).")
else:
    backup = TARGET.with_suffix(".py.bak")
    if not backup.exists():
        backup.write_text(TARGET.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[OK] Backup written: {backup.name}")
    TARGET.write_text(src, encoding="utf-8")
    print(f"[OK] Patched: {TARGET.name}")

print("Done.")
