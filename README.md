# TTS Benchmark Harness

A Python harness for benchmarking open-source text-to-speech (TTS) models on GPU hardware, built as part of the model benchmarking work at [360Inference](https://www.360inference.ai), an inference-engine optimisation startup.

The harness runs each model over the same text inputs, generates the audio, calculates quality and speed metrics, and logs the results so models can be compared side by side. It was developed and run on **MeluXina**, a EuroHPC petascale supercomputer in Luxembourg, using NVIDIA A100 GPU nodes scheduled through SLURM.

## Models evaluated

| Model | Size | Notes |
|---|---|---|
| [Coqui XTTS v2](https://huggingface.co/coqui/XTTS-v2) | – | Multilingual, zero-shot voice cloning |
| [Parler-TTS](https://huggingface.co/parler-tts) | Mini / Large variants | Voice controlled by a text description |
| [VibeVoice-1.5B](https://huggingface.co/microsoft/VibeVoice-1.5B) | 1.5B | Long-form, multi-speaker speech |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | 82M | Very small, fast model |

## Metrics

| Metric | What it measures | Better is |
|---|---|---|
| **RTF** (real-time factor) | Time to generate audio ÷ length of the audio. An RTF of 0.1 means 10 seconds of speech takes 1 second to generate. | Lower |
| **MCD** (mel cepstral distortion) | How far the generated audio's spectral features are from a reference recording | Lower |
| **Speaker similarity** | Cosine similarity between speaker embeddings of the generated audio and the reference voice | Higher |

## How it works

1. **Load** each model from Hugging Face, using a shared model cache so weights are downloaded once and reused across runs.
2. **Synthesise** audio for every text prompt in the evaluation set.
3. **Measure** generation time, audio length and the metrics above for every sample.
4. **Log** per-sample and per-model results to file for comparison.

Runs are designed to be reproducible and unattended: each job runs in a persistent Python/Conda environment and is submitted to the cluster as a SLURM batch job.

## Running on an HPC cluster (SLURM)

```bash
# 1. Create the environment (once)
conda create -n tts-bench python=3.10 -y
conda activate tts-bench
pip install -r requirements.txt

# 2. Point the Hugging Face cache at shared project storage (once per session)
export HF_HOME=/path/to/project/storage/hf-cache

# 3a. Interactive test on a GPU node
salloc --partition=gpu --gres=gpu:1 --time=01:00:00
python run_benchmark.py --model kokoro

# 3b. Full unattended run
sbatch jobs/benchmark_all.sbatch
```

<!-- TODO (Jonathan): change the file names above (requirements.txt, run_benchmark.py, jobs/benchmark_all.sbatch) and the --model option to match your actual scripts. Also check the partition name matches what you use on MeluXina. -->

## Results

<!-- TODO (Jonathan): paste your summary table here once runs are complete, e.g. one row per model with average RTF, MCD and speaker similarity, plus the GPU used. A short sentence on what you found (e.g. which model was fastest, which sounded most like the reference voice) is worth more to a reader than the raw numbers. -->

Results will be added here once the benchmark runs are complete.

## Practical lessons from running on HPC

- **SSH key authentication** to the cluster needs setting up carefully before anything else will work.
- **SLURM QOS and resource limits** decide how many GPUs and how much time a job can request. Jobs that ask for more than the allowance sit in the queue or are rejected.
- **Shared-storage permissions** matter when several team members use the same model cache and project directories.
- **Cache models on shared storage.** Downloading multi-gigabyte weights at the start of every job wastes GPU time.

## Related work

The wider 360Inference benchmarking programme also covers LLM serving (vLLM with Qwen2.5-7B-Instruct), speech-to-text, text-to-video and multimodal models, each run by a different team member.

## Author

**Jonathan Whale-Martin** · [LinkedIn](https://www.linkedin.com/in/jonathanwhalemartin/) · Developer, 360Inference
