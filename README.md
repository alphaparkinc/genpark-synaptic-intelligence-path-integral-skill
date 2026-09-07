# GenPark Synaptic Intelligence Path Integral Skill

Online path-integral synaptic importance estimation engine mitigating catastrophic forgetting in continual learning.

For more agent technologies, visit [GenPark](https://genpark.ai) and the [GenPark MCP Catalog](https://genpark.ai/mcp).

```mermaid
graph LR
    A[Task 1 Parameter Trajectory] -->|Path Integral Integral g_k d_theta_k| B[Calculate Omega_k]
    B --> C[Freeze Omega_k as Synaptic Memory]
    C --> D[Task 2 Optimization with Surrogate Penalty c * Omega_k * (theta_k - theta_k*)^2]
```

## Features
- Online gradient-trajectory path integral accumulation.
- Does not require storing past training samples.
- Zero external dependencies.
