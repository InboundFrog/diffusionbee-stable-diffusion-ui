"""Smoke-test a bundled backend binary through its real CLI/stdio protocol (docs/backend_protocol.md).

    python packaging/smoke_test.py dist/diffusionbee_backend/diffusionbee_backend

download_model -> inspect_model -> long-running mode: wait for `sdbk mdld`, run one tiny t2im job, __stop__.
Needs network once (11 MB tiny model into the Hugging Face cache). Exits non-zero on any failure.
"""
import json, os, queue, subprocess, sys, threading, time

TINY_REPO = 'hf-internal-testing/tiny-sdxl-pipe'  # 11 MB, safetensors (same as backends/stable_diffusion/test_backend.py)
TIMEOUT = 600  # first cold start from a fresh bundle + first MPS kernel compile can be slow


def run(binary, *args):
    p = subprocess.run([binary, *args], capture_output=True, text=True, timeout=TIMEOUT)
    if p.returncode:
        sys.exit(f'FAIL {args[0]} rc={p.returncode}\n{p.stdout[-2000:]}\n{p.stderr[-3000:]}')
    return p.stdout.splitlines()


def main(binary):
    t0 = time.time()
    done = [l for l in run(binary, 'download_model', TINY_REPO) if l.startswith('done ')]
    assert done, 'download_model printed no "done <dir>" line'
    model_dir = done[-1][5:].strip()
    info = json.loads(run(binary, 'inspect_model', model_dir)[-1])
    print(f'inspect_model: {info}')
    assert info.get('type') == 'sd_model', info

    p = subprocess.Popen([binary], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, bufsize=1)
    lines = queue.Queue()
    threading.Thread(target=lambda: [lines.put(l.rstrip('\n')) for l in p.stdout] + [lines.put(None)],
                     daemon=True).start()
    err = []
    threading.Thread(target=lambda: err.extend(p.stderr), daemon=True).start()

    def wait_for(*prefixes):
        deadline = time.time() + TIMEOUT
        while time.time() < deadline:
            try:
                l = lines.get(timeout=1)
            except queue.Empty:
                continue
            if l is None:
                sys.exit(f'FAIL backend exited rc={p.wait()}\n{"".join(err)[-3000:]}')
            if l.startswith('sdbk errr'):
                sys.exit(f'FAIL {l}\n{"".join(err)[-3000:]}')
            if l.startswith(prefixes):
                return l
        p.kill()
        sys.exit(f'FAIL timeout waiting for {prefixes}\n{"".join(err)[-3000:]}')

    t_start = time.time()
    wait_for('sdbk mdld')
    print(f'sdbk mdld after {time.time() - t_start:.1f}s')
    wait_for('sdbk inrd')
    job = dict(prompt='a bee', negative_prompt='', model_path=model_dir, num_imgs=1, img_width=128,
               img_height=128, num_steps=2, guidance_scale=7.5, seed=1)
    p.stdin.write('b2py t2im ' + json.dumps(job) + '\n')
    nwim = json.loads(wait_for('sdbk nwim')[len('sdbk nwim '):])
    assert os.path.isfile(nwim['generated_img_path']), nwim
    print(f't2im ok: {nwim["generated_img_path"]}')
    p.stdin.write('__stop__\n')
    p.stdin.close()
    try:
        p.wait(timeout=60)
    except subprocess.TimeoutExpired:
        p.kill()
    print(f'smoke test passed in {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main(sys.argv[1])
