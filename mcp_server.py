"""
MCP Server for Synaptic Intelligence Path Integral Skill
"""

import json
import sys
from client import SynapticIntelligenceTracker

si = SynapticIntelligenceTracker()

def handle_call(name: str, args: dict) -> dict:
    if name == "start_task":
        w = args.get("initial_weights", {})
        si.start_task(w)
        return {"status": "task_started"}
    elif name == "update_step":
        w = args.get("weights", {})
        g = args.get("gradients", {})
        si.update_step(w, g)
        return {"running_integral": si.running_w}
    elif name == "finalize_task":
        w = args.get("final_weights", {})
        si.finalize_task(w)
        return {"omega": si.omega}
    elif name == "compute_si_penalty":
        w = args.get("current_weights", {})
        opt = args.get("optima", {})
        l, g = si.compute_regularization(w, opt)
        return {"si_loss": l, "si_grads": g}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        res = handle_call(req.get("method"), req.get("params", {}))
        sys.stdout.write(json.dumps(res) + "\n")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
