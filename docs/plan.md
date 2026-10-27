# Idea

- Reduce inter-node EP communication and increase load balance of MoE with offloading with speculative expert activation.
    - Q. How to predict experts?
    1. Use router on previous layers activation.
    2. Train a MLP to predict within the same layer but before attention (overlap with attention).
    - Q. How to reduce inter-node communication?
        - At node-level, obtain a list of predicted experts that’s not already loaded.
        - Swap out non-predicted experts, in the order of LRU.
        - Q. How should we swap out the non-predicted experts?
            - We have multiple options depending on the system.
            - If we have a lot of system memory (RAM), we can do offloading.
            - Otherwise, we can do inter-node communication but in an overlapable manner.
                - Gradients will get passed back during backprop.
        - At global-level, obtain a list of all experts loaded and mark one expert to be not evictable.
            - This ensures that all experts exist somewhere and we can leverage inter-node communication > offloading.
        - Q. What if there is not enough space of experts? What if your predictions are wrong?
            - We will not load a few least popular experts if no more space.
            - For experts not existent at a node-level will incur inter-node communication.
    - Q. How to increase load balance of EP?
        - You can shuffle the experts to balance out the load.
- Azad’s Idea
    - Find clusters of activations that are frequently used.
- Assumption
    - Inter-node communication of all2all is expensive.
        - Also true for intra-node?
    - Offloading experts are fast.
    - [ ]  Q. Should we avoid redundant experts to avoid DP? Are DP comm expensive?
    - [ ]  Q. Is inter-node communication faster than offloading?
    - [ ]  Q. Are intra-node communication expensive?

# Plan

1. Run a pre-trained MoE on DeepSpeed and profile
    1. Choose a pretrained model (e.g. Qwen 3.6 35B)
2. Check the expert prediction rate
3. Check the possible throughput and memory gain based on my idea.