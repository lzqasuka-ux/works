"""单受试者 fMRIPrep 重跑 — sub-0040120"""
import subprocess, time, os, sys
from pathlib import Path

SUBJ = "sub-0040120"
LICENSE = r"C:\Users\zhongqing.lu\Desktop\license.txt"
BIDS    = r"D:\数据集\COBRE\COBRE_BIDS"
OUT     = r"D:\数据集\COBRE\COBRE_BIDS\derivatives"
WORK    = r"D:\数据集\COBRE\fmriprep_work"

os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

def to_docker_path(p):
    s = str(p)
    if s[1] == ':':
        s = '/' + s[0].lower() + s[2:].replace('\\', '/')
    return s.replace('\\', '/')

cmd = [
    "docker", "run", "--rm",
    "-v", f"{to_docker_path(BIDS)}:/data:ro",
    "-v", f"{to_docker_path(OUT)}:/out",
    "-v", f"{to_docker_path(WORK)}:/work",
    "-v", f"{to_docker_path(LICENSE)}:/opt/freesurfer/license.txt:ro",
    "nipreps/fmriprep:latest",
    "/data", "/out", "participant",
    "--participant-label", SUBJ,
    "--output-spaces", "MNI152NLin2009cAsym:res-2",
    "--fs-no-reconall",
    "--fs-license-file", "/opt/freesurfer/license.txt",
    "--skip-bids-validation",
    "--stop-on-first-crash",
    "--notrack",
    "--nprocs", "8",
    "--mem", "16000",
    "-w", "/work",
]

print(f"[{time.strftime('%H:%M:%S')}] 开始 {SUBJ}")
t0 = time.time()

log_path = os.path.join(WORK, f"{SUBJ}.log")
with open(log_path, "w") as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=False)
    proc.wait()

elapsed = time.time() - t0
if proc.returncode == 0:
    print(f"[{time.strftime('%H:%M:%S')}] {SUBJ} 完成 ({elapsed/60:.1f} min)")
else:
    print(f"[FAIL] {SUBJ} (exit {proc.returncode}) — 日志: {log_path}")
    sys.exit(1)