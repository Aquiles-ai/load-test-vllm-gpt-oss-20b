# Load testing openai/gpt-oss-20b with vLLM and Docker

This repository contains everything necessary to replicate the load test to the vLLM server with the openai/gpt-oss-20b model, where 98.5% of successful requests were obtained (It was run on an NVIDIA H100, but it is expected that with 3 NVIDIA RTX 50590, they can have a similar performance)

Command to start the vLLM server:

```bash
docker run --gpus all \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.10.1 \
    --model openai/gpt-oss-20b \
    --api-key dummyapikey \
    --async-scheduling
```

And to run the load test:
```bash
python load_test_vllm_gpt_oss_20b.py
```
