## vllm serve command

```
vllm serve acs/lintang14b \ 
   --served_model_name cso14b \
   --enable-auto-tool-choice \
   --tool-call-parser hermes \
   --tensor_parallel_size 2 \
   --data_parallel_size 4 \
   --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}' --max-model-len 131072
```
**add yarn config is important to allow inputs longer than 40k**
using weights from https://huggingface.co/neulab/cso-q3-14b-8x8-swe_smith-multilevel_f05_minimum-terminal-250


## run benchmarking

benchmark datasets: https://huggingface.co/datasets/Leon-Leee/acs_evals

```
uv run python -m benchmarks.agentic_code_search.run_infer \
    --dataset_file ../acs_evals/gt_location_swebench_lite_filtered.jsonl  \
    --llm-config-path .llm_config/cso14b.json \
    --system_prompt_file benchmarks/agentic_code_search/prompts/system_prompt.j2  \
    --user_prompt_file benchmarks/agentic_code_search/prompts/file_module_short.j2   \
    --tools terminal   \
    --max-iterations 15   \
    --num-workers 16   \
    --output-dir ./lintang14b-yarn-274-turn15   \
    --n-limit 300 \
    --runtime local   \
    --workspace_base_dir /tmpworkspace/testbed/   \
    --instance-timeout 60
```
note `--max-iterations 15` is the setting used by LocAgent

## Results

### 🌟 SWEBench lite (filtered), seems very close to Lintang's evaluation 
```
{ 
  "total_instances": 274,
  "successful_instances": 274,
  "error_count": 0,
  "file_loc_f1": 0.7019464720194648,
  "module_loc_f1": 0.5742092457420925,
  "entity_loc_f1": 0.43187347931873477,
  "avg_wall_time_seconds": 34.45994254446378,
  "avg_num_steps": 6.562043795620438,
  "avg_num_tool_calls": 11.459854014598541,
  "wall_time_seconds": 859.6407029628754
}
```

### SWEBench lite (unfiltered)
```
{
  "total_instances": 300,
  "successful_instances": 300,
  "error_count": 0,
  "file_loc_f1": 0.69,
  "module_loc_f1": 0.5211111111111112,
  "entity_loc_f1": 0.39355555555555555,
  "avg_num_steps": 6.386666666666667,
  "avg_num_tool_calls": 11.156666666666666,
  "avg_wall_time_seconds": 31.536650454203286,
  "wall_time_seconds": 890.1812348365784
}
```
**So the following I used the filtered subsets:**

### SWEBench Pro (filtered)
```
{
  "total_instances": 234,
  "successful_instances": 234,
  "error_count": 0,
  "file_loc_f1": 0.6151878985212318,
  "module_loc_f1": 0.4269910320348917,
  "entity_loc_f1": 0.32882356926970724,
  "avg_num_steps": 6.196581196581197,
  "avg_num_tool_calls": 11.482905982905983,
  "avg_wall_time_seconds": 31.988532804016373,
  "wall_time_seconds": 961.316802740097
}
```

### SWEBench Verified (filtered)
```
{
  "total_instances": 453,
  "successful_instances": 453,
  "error_count": 0,
  "file_loc_f1": 0.7086092715231788,
  "module_loc_f1": 0.5637285819405025,
  "entity_loc_f1": 0.46095841857431263,
  "avg_num_steps": 6.377483443708609,
  "avg_num_tool_calls": 11.097130242825607,
  "avg_wall_time_seconds": 30.253412038811522,
  "wall_time_seconds": 1289.6503579616547
}
```