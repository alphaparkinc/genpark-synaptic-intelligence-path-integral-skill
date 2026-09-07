"""
Demonstration of Synaptic Intelligence Path Integral Skill
"""

from client import SynapticIntelligenceTracker

def main():
    print("=== Synaptic Intelligence (SI) Online Path Integral ===")
    si = SynapticIntelligenceTracker(c_reg=2.0)

    initial_params = {"synapse_1": 1.0, "synapse_2": 0.5}
    si.start_task(initial_params)

    # Simulate 3 optimization steps on Task 1
    step1_params = {"synapse_1": 1.1, "synapse_2": 0.52}
    step1_grads = {"synapse_1": -0.8, "synapse_2": -0.1}
    si.update_step(step1_params, step1_grads)

    step2_params = {"synapse_1": 1.25, "synapse_2": 0.55}
    step2_grads = {"synapse_1": -0.7, "synapse_2": -0.05}
    si.update_step(step2_params, step2_grads)

    si.finalize_task(step2_params)
    print("Task 1 Synaptic Importance Omega:", si.omega)

    # When training Task 2, penalize deviation from Task 1 optima
    test_weights = {"synapse_1": 1.15, "synapse_2": 0.60}
    loss, grads = si.compute_regularization(test_weights, step2_params)
    print(f"Task 2 SI Regularization Loss: {loss:.4f}")
    print(f"Task 2 SI Gradients: {grads}")

    assert loss > 0.0
    print("Synaptic Intelligence Path Integral Verification PASS!")

if __name__ == "__main__":
    main()
