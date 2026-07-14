#!/usr/bin/env python
"""
批量 fMRIPrep 预处理脚本 — COBRE 数据集
用法（不要直接跑，等你的指令）：
    python run_fmriprep_batch.py --bids COBRE_BIDS --out COBRE_BIDS/derivatives

可选参数：
    --n-procs N          : 同时处理几个受试者（默认 1，有 GPU 建议 1–2）
    --subjects sub-01 sub-02 ... : 只处理指定受试者（默认全部 148 个）
    --fs-no-reconall     : 跳过 FreeSurfer 表面重建（加速 2–3 倍）
    --dry-run            : 只打印命令，不实际执行
    --resume             : 跳过已有 output 的受试者
"""

import os, sys, subprocess, argparse, time, json, glob
from pathlib import Path

# ---------- 默认路径 ----------
# 切换到 D 盘，C 盘已满
DEFAULT_BIDS  = r"D:\数据集\COBRE\COBRE_BIDS"
DEFAULT_OUT   = r"D:\数据集\COBRE\COBRE_BIDS\derivatives"
DEFAULT_WORK  = r"D:\数据集\COBRE\fmriprep_work"

# ---------- 参数解析 ----------
def parse_args():
    p = argparse.ArgumentParser(description="Batch fMRIPrep for COBRE BIDS dataset")
    p.add_argument("--bids", default=DEFAULT_BIDS,
                   help="BIDS 根目录 (默认: COBRE_BIDS)")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="输出目录, 生成 derivatives/ (默认: COBRE_BIDS/derivatives)")
    p.add_argument("--work", default=DEFAULT_WORK,
                   help="fMRIPrep 工作目录 (可删除) (默认: fmriprep_work)")
    p.add_argument("--n-procs", type=int, default=1,
                   help="并行受试者数量 (默认 1)")
    p.add_argument("--subjects", nargs="*", default=None,
                   help="只处理指定受试者, e.g. --subjects sub-0040000 sub-0040001")
    p.add_argument("--fs-no-reconall", action="store_true",
                   help="禁用 FreeSurfer 皮层重建 (推荐加速)")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印命令不执行")
    p.add_argument("--resume", action="store_true",
                   help="跳过已有 output 的受试者")
    p.add_argument("--fs-license", default=None,
                   help="FreeSurfer license 文件路径 (如果需要 FS)")
    return p.parse_args()

# ---------- 工具函数 ----------
def get_subjects(bids_dir, selected=None):
    """扫描 BIDS 目录获取所有 sub-* 文件夹"""
    bids = Path(bids_dir)
    if selected:
        subs = [s for s in selected if (bids / s).is_dir()]
        missing = set(selected) - set(subs)
        if missing:
            print(f"[WARNING] 未找到这些受试者: {', '.join(sorted(missing))}")
        return sorted(subs)
    return sorted([d.name for d in bids.glob("sub-*") if d.is_dir()])

def has_output(subj, out_dir):
    """检查某受试者是否已有完整输出（BOLD MNI 文件为最终标志）"""
    bold_mni = Path(out_dir) / subj / 'ses-1' / 'func' / f'{subj}_ses-1_task-rest_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'
    return bold_mni.exists()

def build_cmd(subj, args):
    """拼装 fMRIPrep docker 命令"""
    bids_dir = Path(args.bids).resolve()
    out_dir = Path(args.out).resolve()
    work_dir = Path(args.work).resolve()

    # 自动创建输出 & 工作目录
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    # 基础命令 — 参考 PowerShell 参考脚本的 Docker 命令
    # Windows 路径: C:\Users\... → /c/users/... (兼容 Git Bash / Docker WSL)
    def to_docker_path(p):
        s = str(p)
        if s[1] == ':':
            s = '/' + s[0].lower() + s[2:].replace('\\', '/')
        return s.replace('\\', '/')

    bids_docker = "/data"
    out_docker = "/out"
    work_docker = "/work"
    fs_license_docker = "/opt/freesurfer/license.txt"

    # 查找 license 文件
    if args.fs_license:
        license_src = Path(args.fs_license).resolve()
    else:
        # 默认：当前项目目录或 sibling fMRIPrep 目录
        candidates = [
            Path.cwd() / "license.txt",
            Path(__file__).resolve().parent.parent / "fMRIprep" / "license.txt",
        ]
        license_src = None
        for c in candidates:
            if c.exists():
                license_src = c
                break

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{to_docker_path(bids_dir)}:{bids_docker}:ro",
        "-v", f"{to_docker_path(out_dir)}:{out_docker}",
        "-v", f"{to_docker_path(work_dir)}:{work_docker}",
    ]

    # 挂载 FreeSurfer license（即使 --fs-no-reconall 也需要，用于脑提取 / ANTs）
    if license_src and license_src.exists():
        cmd += ["-v", f"{to_docker_path(license_src)}:{fs_license_docker}:ro"]
    else:
        print(f"  [WARNING] 未找到 FreeSurfer license 文件! 请通过 --fs-license 指定路径")

    cmd += [
        "nipreps/fmriprep:latest",
        bids_docker,
        out_docker,
        "participant",
        "--participant-label", subj,
        "--output-spaces", "MNI152NLin2009cAsym:res-2",
        "--fs-no-reconall",
        "--fs-license-file", fs_license_docker,
        "--skip-bids-validation",   # 参考脚本中的关键参数，跳过慢且易炸的 BIDS 校验
        "--stop-on-first-crash",
        "--notrack",
        "--nprocs", "8",
        "--mem", "16000",
        "-w", work_docker,
    ]

    return cmd

# ---------- 批量逻辑 ----------
def run_one(subj, args):
    """处理单个受试者，返回 True/False"""
    if args.resume and has_output(subj, args.out):
        print(f"[SKIP] {subj} 已有输出，跳过")
        return True

    cmd = build_cmd(subj, args)
    cmd_str = " ".join(cmd)

    if args.dry_run:
        print(f"[DRY-RUN] {subj}: {cmd_str}")
        return True

    t0 = time.time()

    try:
        # 不要实时打印，避免 stdout 缓冲区溢出卡住（Windows 常见）
        # 将 Docker 输出重定向到日志文件
        log_file = Path(args.work) / f"{subj}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=False,   # 以二进制模式写入，避免 Unicode 解码错误
            )
            print(f"[{time.strftime('%H:%M:%S')}] 开始 {subj} -> 日志: {log_file}")
            proc.wait()

        elapsed = time.time() - t0
        if proc.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] ✓ {subj} 完成 ({elapsed/60:.1f} min)")
            return True
        else:
            print(f"[FAIL] {subj} (退出码 {proc.returncode}) — 日志: {log_file}")
            return False
    except Exception as e:
        print(f"[ERROR] {subj} 执行异常: {e}")
        return False

def run_batch(subjects, args):
    """顺序或并发执行（简单 sequential）"""
    # 注：fMRIPrep 本身会用多线程；并发多个受试者内存消耗极大
    # 建议用 --n-procs 1，如需并发推荐 docker-compose 或 SLURM
    if args.n_procs > 1:
        print(f"[INFO] --n-procs={args.n_procs}（并发处理需注意内存，每个受试者 ~16-24 GB）")

    total = len(subjects)
    ok, fail = 0, 0

    for i, subj in enumerate(subjects, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{total}] {subj}")
        print(f"{'='*60}")

        if run_one(subj, args):
            ok += 1
        else:
            fail += 1

        # 简单并发：每次结束后等待所有子进程退出
        # （如有 n_procs > 1 需求需重构，此处保持简单 sequential 最稳定）

    print(f"\n{'='*60}")
    print(f"全部完成: OK={ok}, FAIL={fail}, TOTAL={total}")
    print(f"{'='*60}")

# ---------- 入口 ----------
if __name__ == "__main__":
    args = parse_args()

    # 1. 扫描受试者
    subjects = get_subjects(args.bids, args.subjects)
    if not subjects:
        print("[ERROR] 未找到任何受试者！请检查 --bids 路径")
        sys.exit(1)
    print(f"[INFO] 找到 {len(subjects)} 个受试者")
    if args.subjects:
        print(f"  (用户指定: {', '.join(args.subjects)})")

    # 2. 打印配置摘要
    print(f"[CONFIG]")
    print(f"  BIDS     : {args.bids}")
    print(f"  输出     : {args.out}")
    print(f"  工作目录 : {args.work}")
    print(f"  FreeSurfer : {'跳过' if args.fs_no_reconall else '启用'}")
    print(f"  Dry-run  : {'是 (不真正运行!)' if args.dry_run else '否'}")
    print(f"  续算     : {'是 (跳过已有输出)' if args.resume else '否'}")
    print()

    # 3. 批量执行
    run_batch(subjects, args)