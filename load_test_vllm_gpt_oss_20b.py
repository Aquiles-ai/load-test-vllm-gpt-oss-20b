import asyncio
import time
import statistics
from typing import List, Tuple
import httpx
from openai import AsyncOpenAI  

# CONFIG
BASE_URL = "http://127.0.0.1:8000/v1"     # tu endpoint local de vLLM
API_KEY = "dummyapikey"                 
MODEL = "openai/gpt-oss-20b"
INPUT_TEXT = "Hola como te llamas?"

# Fases (rps, dur_seconds)
PHASES: List[Tuple[int, int]] = [
    (100, 10),
    (500, 10),
    (1000, 10),
    (1500, 10),
    (2000, 10),
]

# Timeout por request
REQUEST_TIMEOUT = 30.0

# Métricas globales
all_latencies: List[float] = []
all_statuses: List[int] = []
all_errors: List[str] = []

def pctile(data: List[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[int(k)]
    d0 = s[f] * (c - k)
    d1 = s[c] * (k - f)
    return d0 + d1

async def send_with_asyncopenai(client: AsyncOpenAI, payload: dict):
    start = time.perf_counter()
    try:
        # Llamada asíncrona al Responses API
        resp = await client.responses.create(**payload)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        all_latencies.append(elapsed_ms)

        all_statuses.append(200)
        return resp
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        all_latencies.append(elapsed_ms)
        all_errors.append(repr(e))
        all_statuses.append(0)
        return None

async def run_phase(rps: int, duration_s: int):
    print(f"\n=== FASE {rps} RPS durante {duration_s}s ===")

    limits = httpx.Limits(max_keepalive_connections=1000, max_connections=max(1000, rps*2))
    httpx_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, limits=limits)
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, http_client=httpx_client, timeout=REQUEST_TIMEOUT)

    tasks = []
    payload = {
        "model": MODEL,
        "input": INPUT_TEXT,
    }

    inter = 1.0 / rps
    phase_start = time.perf_counter()
    for sec in range(duration_s):
        sec_base = time.perf_counter()
        for i in range(rps):
            # schedule task
            t = asyncio.create_task(send_with_asyncopenai(client, payload))
            tasks.append(t)
            await asyncio.sleep(inter)
        elapsed_in_sec = time.perf_counter() - sec_base
        if elapsed_in_sec < 1.0:
            await asyncio.sleep(1.0 - elapsed_in_sec)

    await asyncio.gather(*tasks, return_exceptions=True)

    await httpx_client.aclose()

    total_sent = len(tasks)
    phase_statuses = all_statuses[-total_sent:] if total_sent > 0 else []
    succeeded = sum(1 for s in phase_statuses if s and s < 400)
    failed = total_sent - succeeded

    phase_latencies = all_latencies[-total_sent:] if total_sent > 0 else []
    avg_lat = statistics.mean(phase_latencies) if phase_latencies else 0
    med = statistics.median(phase_latencies) if phase_latencies else 0
    p95 = pctile(phase_latencies, 95)
    p99 = pctile(phase_latencies, 99)

    actual_duration = time.perf_counter() - phase_start
    achieved_rps = total_sent / actual_duration if actual_duration > 0 else 0

    print(f"Enviadas: {total_sent} | Success aprox: {succeeded} | Fail: {failed}")
    print(f"Tiempo real fase: {actual_duration:.2f}s | Throughput observado: {achieved_rps:.1f} req/s")
    print(f"Latencia (ms): avg={avg_lat:.1f} med={med:.1f} p95={p95:.1f} p99={p99:.1f}")
    if all_errors:
        print("Errores (muestra hasta 5):")
        for e in all_errors[-5:]:
            print(" -", e)

async def main():
    print("Iniciando load test usando AsyncOpenAI")
    for rps, dur in PHASES:
        await run_phase(rps, dur)

    # resumen global
    total = len(all_statuses)
    successes = sum(1 for s in all_statuses if s and s < 400)
    fails = total - successes
    avg_lat = statistics.mean(all_latencies) if all_latencies else 0
    med = statistics.median(all_latencies) if all_latencies else 0
    p95 = pctile(all_latencies, 95)
    p99 = pctile(all_latencies, 99)

    print("\n=== RESUMEN GLOBAL ===")
    print(f"Total requests: {total} | Success (aprox): {successes} | Fail: {fails}")
    print(f"Latencia total (ms): avg={avg_lat:.1f} med={med:.1f} p95={p95:.1f} p99={p99:.1f}")
    if all_errors:
        print("Algunos errores (hasta 10):")
        for e in all_errors[:10]:
            print(" -", e)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrumpido por usuario")
