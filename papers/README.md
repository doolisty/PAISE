# Paper Reading Notes

## Distributed Training

<details>
<summary><b>[2022.11] Galvatron</b></summary>
<br>
* **Related Works**:
    * 自动并行（dp + tp + search algo）：OptCNN (14), FlexFlow (15, 43), Tofu (45), TensorOpt (7)
    * 自动并行（dp + pp）：PipeDream (27), DAPPLE (12)
    * 四种并行：Data Parallelism (DP), Sharded Data Parallelism (SDP), Tensor Parallelism (TP), Pipeline Parallelism (PP)
* **Motivation**:
    * transformer 都在用，很好，需要优化（说明 target）
    * 问题 1：其他工作只能支持有限的并行维度（only support limited parallelism dimensions， i.e., data parallelism and rare model parallelism dimensions）
    * 问题 2：case-by-case 针对模型和硬件设计并行策略，实际使用可能 sub-optimal（rely on strong model and hardware configurations (i.e., expert-designed parallelism strategy) and result in sub-optimal performance in practice）
    * 推论出不足：自动并行搜索的策略空间太小了
* **Methods**:
    * 把 dp 结合进许多种 mp dimension 中，构成一个比较大的搜索空间  =>  重点在 efficiently search

</details>