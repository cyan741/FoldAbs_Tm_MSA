#!/usr/bin/env python3
"""Run the isolated mini-ESM subset extraction or prepare L4 epoch inputs."""
import argparse
import json
import fcntl
import time
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets')
SUBSET = Path('/NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings')
FULL = Path('/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa')

def main():
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['subset', 'l4-inputs'], default='subset')
    p.add_argument('--output-root')
    p.add_argument('--features-name', default='features_cueq')
    p.add_argument('--work-dir')
    p.add_argument('--gpu-ids', default='7')
    p.add_argument('--epochs', type=int, nargs='*', default=[])
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--offline-train-negatives', action='store_true',
                   help='Explicit opt-in only; training negatives default to online GPU inference')
    args = p.parse_args()
    profile = Path(args.output_root or (SUBSET / 'protenix_mini_esm_v0.5.0' if args.mode=='subset' else FULL)).resolve()
    profile.mkdir(parents=True, exist_ok=True)
    pipeline_lock = (profile / '.pipeline.lock').open('a')
    fcntl.flock(pipeline_lock, fcntl.LOCK_EX)
    log_dir = profile / 'logs'
    log_dir.mkdir(exist_ok=True)
    inputs_dir = profile / 'inputs'
    inputs_dir.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update(PROTENIX_ROOT_DIR=str(ASSETS), LAYERNORM_TYPE='torch',
               SSL_CERT_FILE='/etc/ssl/certs/ca-certificates.crt')
    def run(script, *argv, gpu=None):
        command = [sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str,argv)]
        child_env = env.copy()
        if gpu is not None:
            child_env['CUDA_VISIBLE_DEVICES'] = gpu
        print('Running: ' + ' '.join(command),flush=True)
        log_path = log_dir / (Path(script).stem + '_' + str(time.time_ns()) + '.log')
        print(f'Log: {log_path}', flush=True)
        with log_path.open('w') as logfile:
            result = subprocess.run(command, env=child_env, cwd=ROOT, stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode:
            print(''.join(log_path.read_text().splitlines(keepends=True)[-25:]), flush=True)
            raise subprocess.CalledProcessError(result.returncode, command)
        print(f'Finished: {script}', flush=True)
    if args.mode == 'subset':
        import pandas as pd
        sys.path.insert(0,str(ROOT))
        from scripts.make_protenix_input.make_tmp_protenix_input import build_protenix_inputs
        manifest = pd.read_csv(SUBSET / 'manifest.csv')
        source = pd.read_csv('/NAS/luyq/FoldAbS_Tm_MSA_level2/work/hitph_level2.csv', keep_default_na=False)
        selected = manifest.merge(source, left_on='sample_id', right_on='id', validate='one_to_one',suffixes=('','_source'))
        if len(selected) != len(manifest) or not selected.label.eq(selected.label_source).all() or not selected.split.eq(selected.split_source).all():
            raise ValueError('Subset IDs, labels or splits differ from delivered source')
        input_json = inputs_dir / 'subset.json'
        input_json.write_text(json.dumps(build_protenix_inputs(selected),indent=2))
        manifest.to_csv(profile / 'manifest.csv',index=False)
        input_paths = [input_json]
        notes = {'dataset':'legacy HiTPH level2 smoke test', 'records':len(manifest),
                 'split_counts':{str(k):int(v) for k,v in manifest.groupby(['split','label']).size().items()},
                 'tcra_tcrb':'CDR3 sequences from unchanged existing subset',
                 'negative_protocol':'existing fixed 1:1 subset; this is not the L4 dynamic 1:2 experiment'}
        (profile / 'dataset_notes.json').write_text(json.dumps(notes,indent=2))
    else:
        input_json = inputs_dir / 'baseline.json'
        run('make_protenix_input/make_tmp_protenix_input.py', '--hitph-splits-dir',
            '/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits', '--no-msa', '--output-json', input_json)
        input_paths = [input_json]
        for epoch in args.epochs:
            path = inputs_dir / f'train_negative_epoch_{epoch:03d}.json'
            run('make_protenix_input/make_tmp_protenix_input.py','--hitph-splits-dir',
                '/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits','--no-msa',
                '--train-negative-epoch',epoch,'--output-json',path)
            input_paths.append(path)
    if args.prepare_only:
        return
    run('prepare_mini_esm_assets.py','--checkpoint-dir', ASSETS / 'checkpoint')
    # Process exits before Protenix is loaded; no ESM model remains referenced on GPU.
    run('extract_protenix_esm_chains.py','--input-json',*input_paths,'--cache-dir',profile / 'esm_cache',
        '--checkpoint-dir',ASSETS / 'checkpoint','--seed','42','--dtype','bf16',gpu=args.gpu_ids.split(',')[0])
    for path in input_paths:
        if path.name.startswith('train_negative_') and not args.offline_train_negatives:
            continue
        run('extract_protenix_embeddings.py','--input-json',path,'--output-dir',profile / args.features_name,
            '--model-name','protenix_mini_esm_v0.5.0','--no-msa','--cycles','4','--limit','0',
            '--seed','42','--dtype','bf16','--cache-dtype','fp16','--esm-cache-dir',profile / 'esm_cache',
            '--gpu-ids',args.gpu_ids,'--resume',
            '--trimul-kernel','cuequivariance','--triatt-kernel','cuequivariance',
            *(['--work-dir', args.work_dir] if args.work_dir else []))
    # Verify every expected tensor against sequence identity and the exact run profile.
    import torch
    torch.set_num_threads(4)
    from protenix_embedding_profile import validate_joint
    spec = json.loads((profile / args.features_name / 'run_spec.json').read_text())
    count, nbytes = 0,0
    for path in input_paths:
        if path.name.startswith('train_negative_') and not args.offline_train_negatives:
            continue
        for record in json.loads(path.read_text()):
            feature = profile / args.features_name / (record['name'] + '.pt')
            payload = torch.load(feature,map_location='cpu',weights_only=True)
            validate_joint(payload, record, spec)
            count += 1
            nbytes += feature.stat().st_size
            del payload
    report = {'status':'complete','validated_records':count,'cache_bytes':nbytes,'profile':spec,
              'features_dir':str(profile / args.features_name),
              'input_paths':[str(path) for path in input_paths if args.offline_train_negatives or not path.name.startswith('train_negative_')],
              'train_negative_strategy':'offline' if args.offline_train_negatives else 'online_gpu'}
    (profile / 'completion_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)

if __name__ == '__main__':
    main()
