#!/usr/bin/env python3
"""Install an attested engine and run only the four cells that require retraining."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base, work, results, commit = sys.argv[1:]
    base, work = Path(base), Path(work)
    manifest = json.loads((base/'manifest.json').read_text())
    spec = manifest['engine']
    engine = Path(spec['root'])
    build = engine/'build_git'
    build.mkdir(parents=True, exist_ok=True)
    report = json.loads((base/'engine_build.json').read_text())
    if (report['source_commit'] != commit or spec['commit'] != commit
            or digest(base/'engine.tar.gz') != report['archive_sha256']
            or report['files']['libge2.so'] != spec['hashes']['libge2_so']
            or report['files']['gege_train'] != spec['hashes']['gege_train']):
        raise ValueError('Engine archive identity mismatch')
    expected = {'libge2.so', 'gege_train', 'gege_manual_training_update_test', 'gege_manual_backward_test'}
    if set(report['files']) != expected:
        raise ValueError('Unexpected engine artifacts')
    with tarfile.open(base/'engine.tar.gz') as archive:
        for name, sha in report['files'].items():
            target = build/name
            if not target.exists():
                member = archive.getmember(name)
                if not member.isfile():
                    raise ValueError('Engine archive member is not a regular file')
                with archive.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
                target.chmod(0o755)
            if digest(target) != sha:
                raise ValueError('Engine binary mismatch: '+name)
    repo = engine/'repo'
    if not repo.exists():
        subprocess.run(['git', 'clone', '--no-hardlinks', '/mnt/local/smansou2/pipege_engine_e498e98/repo', str(repo)], check=True)
    if subprocess.check_output(['git', '-C', str(repo), 'diff', 'HEAD']):
        raise ValueError('Refusing to overwrite engine source changes')
    subprocess.run(['git', '-C', str(repo), 'fetch', str(base/'source.bundle'), commit], check=True)
    subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', commit], check=True)
    (engine/'build_git_completed_commit.txt').write_text(commit+'\n')
    print(json.dumps(dict(status='installed_pending_correctness_gates', commit=commit, hashes=spec['hashes'])), flush=True)
    os.execv(sys.executable, [sys.executable, str(base/'scripts/supervise_arc_pipege_best.py'),
                             '--base', str(base), '--work', str(work), '--results', results,
                             '--commit', commit, '--cases', 'fb_distmult', 'fb_complex', 'wk_distmult', 'wk_complex'])


if __name__ == '__main__':
    main()
