"""Compare raw LLM samples with the final /chat response for two ablations."""
import argparse
import hashlib
import io
import json
import os
import random
import sys
from unittest import mock

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from llm import load_llm


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _read_run(directory):
    model_path = os.path.join(directory, 'llm_model.json')
    metadata_path = os.path.join(directory, 'training_metadata.json')
    with io.open(model_path, encoding='utf-8') as source:
        header = json.load(source)
    with io.open(metadata_path, encoding='utf-8') as source:
        metadata = json.load(source)

    weights_path = os.path.join(directory, header['weights_file'])
    checkpoint_path = os.path.join(directory, 'llm_ckpt.pt')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(weights_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    model = load_llm(model_path)
    if model is None:
        raise ValueError('Model could not be loaded: %s' % model_path)
    return {
        'model': model,
        'metadata': metadata,
        'files': {
            'model_json_sha256': _sha256(model_path),
            'weights_sha256': _sha256(weights_path),
            'checkpoint_sha256': _sha256(checkpoint_path),
            'training_metadata_sha256': _sha256(metadata_path),
        },
        'architecture': {
            key: header.get(key) for key in (
                'arch', 'V', 'd_model', 'num_blocks', 'num_heads',
                'ff_mult', 'max_ctx_len', 'max_seq_len', 'tok_mode',
                'tied_embeddings')
        },
    }


def _validate_runs(run_a, run_b):
    a, b = run_a['metadata'], run_b['metadata']
    if {a.get('natural'), b.get('natural')} != {0, 5}:
        raise ValueError('Expected one natural=0 run and one natural=5 run')
    if not a.get('base_data_fingerprint') or (
            a['base_data_fingerprint'] != b.get('base_data_fingerprint')):
        raise ValueError('The two runs do not have the same base data fingerprint')
    if a.get('optimizer_updates') != b.get('optimizer_updates'):
        raise ValueError('The two runs have different optimizer update counts')
    if not isinstance(a.get('optimizer_updates'), int) or (
            a['optimizer_updates'] < 1):
        raise ValueError('Optimizer update count is missing or invalid')
    shared = (
        'seed', 'limit_pairs', 'batch_size', 'gradient_accumulation',
        'd_model', 'num_blocks', 'num_heads', 'ff_mult',
        'max_context_length', 'max_sequence_length', 'dropout',
        'learning_rate', 'weight_decay', 'training_objective', 'max_steps')
    mismatches = [key for key in shared if a.get(key) != b.get(key)]
    if mismatches:
        raise ValueError('Run configuration mismatch: %s'
                         % ', '.join(mismatches))
    if a.get('gradient_accumulation') != 1 or (
            a.get('max_steps') != a.get('optimizer_updates')):
        raise ValueError('Runs are not fixed optimizer-step ablations')
    if run_a['architecture'] != run_b['architecture']:
        raise ValueError('Model architectures differ between runs')


def _load_prompts(path):
    with io.open(path, encoding='utf-8') as source:
        data = json.load(source)
    prompts = data.get('sorular', [])
    declared = data.get('soru_sayisi', len(prompts))
    if not prompts or declared != len(prompts):
        raise ValueError('Prompt list is empty or its declared count is wrong')
    return [item['soru'] for item in prompts]


def _corpus_fingerprints(base):
    names = ('corpus.jsonl', 'corpus_ids.jsonl')
    return {
        name: _sha256(os.path.join(base, name))
        for name in names
        if os.path.isfile(os.path.join(base, name))
    }


def _load_application():
    import app as application
    from corpus import Corpus

    with mock.patch.object(Corpus, 'seed_from_intents', return_value=None):
        application.load_bot()
    return application, Corpus


def main():
    parser = argparse.ArgumentParser(
        description='Paired raw-model and /chat comparison for naturalize ablation')
    parser.add_argument('natural0_dir')
    parser.add_argument('natural5_dir')
    parser.add_argument('--prompts', default=os.path.join(
        BASE, 'tools', 'soru_listesi_sohbet.json'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    run_a = _read_run(args.natural0_dir)
    run_b = _read_run(args.natural5_dir)
    _validate_runs(run_a, run_b)
    prompts = _load_prompts(args.prompts)
    prompt_hash = _sha256(args.prompts)

    corpus_before = _corpus_fingerprints(BASE)
    application, Corpus = _load_application()
    if not application.model_loaded:
        raise RuntimeError('Application classifier could not be loaded')
    application.fallback_answer = lambda _: application.DEFAULT_UNKNOWN
    client = application.app.test_client()
    generation = {
        'temperature': 0.6,
        'top_k': 6,
        'rep_penalty': 0.4,
        'max_len': 96,
        'seed_per_prompt_and_condition': args.seed,
    }
    rows = []

    write_blocked = RuntimeError(
        'Corpus writes are disabled during paired evaluation')
    with mock.patch.object(Corpus, 'append_many', side_effect=write_blocked), \
            mock.patch.object(Corpus, 'refresh', side_effect=write_blocked):
        for prompt in prompts:
            row = {'question': prompt}
            for label, run in (('natural0', run_a), ('natural5', run_b)):
                model = run['model']
                np.random.seed(args.seed)
                raw = model.sample(
                    prompt, temperature=generation['temperature'],
                    top_k=generation['top_k'],
                    rep_penalty=generation['rep_penalty'],
                    max_len=generation['max_len'])

                application.bot.llm = model
                application.bot.llm_enabled = True
                application._last_tag = None
                np.random.seed(args.seed)
                random.seed(args.seed)
                response = client.post('/chat', json={'message': prompt})
                payload = response.get_json()
                if response.status_code != 200 or not isinstance(
                        payload, dict) or not isinstance(
                            payload.get('response'), str):
                    raise RuntimeError(
                        'Unexpected /chat response for %r: HTTP %d'
                        % (prompt, response.status_code))
                final = payload['response']
                if final.startswith('Hata:'):
                    raise RuntimeError('/chat failed for %r: %s'
                                       % (prompt, final))
                row[label] = {
                    'raw_model_output': raw,
                    'application_final_response': final,
                }
            rows.append(row)

    corpus_after = _corpus_fingerprints(BASE)
    if corpus_before != corpus_after:
        raise RuntimeError('Corpus files changed during evaluation')

    report = {
        'schema_version': 1,
        'prompt_file': os.path.relpath(args.prompts, BASE),
        'prompt_file_sha256': prompt_hash,
        'prompt_count': len(prompts),
        'base_data_fingerprint': run_a['metadata']['base_data_fingerprint'],
        'application_route': '/chat',
        'application_fallback': 'network and learning disabled; unknown fallback',
        'corpus_write_guard': 'append and refresh blocked; file hashes unchanged',
        'application_classifier_sha256': _sha256(
            os.path.join(BASE, 'model', 'model.json')),
        'corpus_hashes_before_after': {
            'before': corpus_before,
            'after': corpus_after,
        },
        'generation': generation,
        'runs': {
            'natural0': {
                'metadata': run_a['metadata'],
                'architecture': run_a['architecture'],
                'file_hashes': run_a['files'],
            },
            'natural5': {
                'metadata': run_b['metadata'],
                'architecture': run_b['architecture'],
                'file_hashes': run_b['files'],
            },
        },
        'items': rows,
    }
    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with io.open(output_path, 'w', encoding='utf-8') as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
    print('Paired raw/application report written: %s' % output_path)
    print('Base data fingerprint: %s' % report['base_data_fingerprint'])
    print('Optimizer updates: %s'
          % run_a['metadata']['optimizer_updates'])
    print('Corpus hashes unchanged.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
