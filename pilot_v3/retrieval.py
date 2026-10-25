"""Stage 2 helpers: downstream same-country question answering with graph context (hypothesis H4 analogue)."""
import numpy as np, torch
from core import enc

SHOTS_GRAPH = [
    ("Graph: Eiffel Tower is located in France. Colosseum is located in Italy.\n", "Eiffel Tower", "Colosseum", " no"),
    ("Graph: Taj Mahal is located in India. Red Fort is located in India.\n", "Taj Mahal", "Red Fort", " yes"),
    ("Graph: Sydney Opera House is located in Australia.\n", "Sydney Opera House", "Machu Picchu", " unknown"),
]
SHOTS_CLOSED = [("", "Eiffel Tower", "Colosseum", " no"), ("", "Taj Mahal", "Red Fort", " yes"), ("", "Sydney Opera House", "Uluru", " yes")]


def prompt(context, x, y, closed):
    shots = SHOTS_CLOSED if closed else SHOTS_GRAPH
    s = "".join(f"{c}Question: Are {a} and {b} located in the same country?\nAnswer:{ans}\n\n" for c, a, b, ans in shots)
    return s + f"{context}Question: Are {x} and {y} located in the same country?\nAnswer:"


@torch.no_grad()
def answer(model, tok, text, options):
    ids = enc(tok, text)
    lg = model(torch.tensor([ids], device=model.device)).logits[0, -1].float().log_softmax(-1)
    first = [tok.encode(o, add_special_tokens=False)[0] for o in options]
    return options[int(np.argmax([lg[t].item() for t in first]))].strip()


def run_pair(model, tok, gx, gy, mode):
    """gx, gy: dicts with subject, pred, isr. mode in {none, all, filtered}."""
    x, y = gx["subject"], gy["subject"]
    if mode == "none":
        return answer(model, tok, prompt("", x, y, True), [" yes", " no"])
    use = [g for g in (gx, gy) if mode == "all" or g["isr"] == 1]
    ctx = "Graph: " + " ".join(f"{g['subject']} is located in {g['pred']}." for g in use) + "\n" if use else "Graph: (no facts)\n"
    return answer(model, tok, prompt(ctx, x, y, False), [" yes", " no", " unknown"])
