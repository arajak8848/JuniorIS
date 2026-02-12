# A Software-Based Comparative Study of Error Sensitivity in Classical and Quantum Optimization Algorithms
Quantum computing uses qubits that can exist in superposition and become entangled, enabling computational approaches that differ fundamentally from classical bit-based systems. Although many quantum algorithms promise theoretical speedups for search and optimization problems, these results often rely on idealized assumptions that overlook noise, limited circuit depth, and other practical constraints of current hardware. This work presents a software-based comparison of classical and quantum optimization algorithms under realistic conditions, evaluating simulated annealing, genetic algorithms, and greedy/local search heuristics against Grover-based search, amplitude amplification, and the Quantum Approximate Optimization Algorithm (QAOA). A modular benchmarking framework with configurable noise models and systematic parameter testing is developed to ensure fair and reproducible comparisons using metrics such as solution quality, robustness, runtime, circuit depth, gate counts, qubit requirements, and sensitivity to noise. Experimental results on small problem instances show that while quantum methods retain theoretical advantages, their performance often degrades significantly in noisy environments, whereas classical heuristics demonstrate greater stability and robustness. The findings highlight key sources of error sensitivity in quantum algorithms and emphasize the importance of realistic evaluation and hybrid classical–quantum approaches for near-term quantum computing.

## Feature Calendar

| **Goal Category** | **Issue** | **Due date** |
| --------- | ------------ | -- 
|Important| [Set up](https://github.com/arajak8848/JuniorIS/issues/1) | 2/14/26 ||
|| [Problem instances](https://github.com/arajak8848/JuniorIS/issues/2) | 2/15/26 ||
|| [Implement algo 1](https://github.com/arajak8848/JuniorIS/issues/3) | 2/19/26 ||
|| [Implement algo 2](https://github.com/arajak8848/JuniorIS/issues/4) | 2/26/26 ||
|Important| [Implement algo 3](https://github.com/arajak8848/JuniorIS/issues/5) | 3/5/26 ||
|Important| [Implement algo 4](https://github.com/arajak8848/JuniorIS/issues/6) | 3/12/26 ||
|Important| [Implement simulated annealing](https://github.com/arajak8848/JuniorIS/issues/7) | 3/20/26 ||
|Important| [Implement amplitude amplification](https://github.com/arajak8848/JuniorIS/issues/8) | 3/27/26 ||
|Stretch goals| [Add noise](https://github.com/arajak8848/JuniorIS/issues/9) | 3/31/26 ||
|| [Create experiment runner](https://github.com/arajak8848/JuniorIS/issues/10) | 4/10/26 ||
|| [Store experiment results](https://github.com/arajak8848/JuniorIS/issues/11) | 4/17/26 ||
|Stretch goals| [Visualize results](https://github.com/arajak8848/JuniorIS/issues/12) | 4/24/26 ||
