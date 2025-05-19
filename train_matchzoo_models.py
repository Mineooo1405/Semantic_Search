import argparse, os, sys, json, random, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import matchzoo as mz

SEED = 2025
torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

# ---------- CLI ---------- #
parser = argparse.ArgumentParser()
parser.add_argument('--train',  required=True,
                    help='TSV file of (query, positive, negative).')
parser.add_argument('--glove100', required=True,
                    help='Path to glove.6B.100d.txt')
parser.add_argument('--glove300', required=True,
                    help='Path to glove.6B.300d.txt')
parser.add_argument('--output', default='./trained_models',
                    help='Root dir to write checkpoints.')
parser.add_argument('--start',  choices=['KNRM','MatchLSTM','ArcI',
                    'MatchPyramid','DRMM','ConvKNRM'],
                    help='Skip models before this name.')
cli = parser.parse_args()

os.makedirs(cli.output, exist_ok=True)

# ---------- util ---------- #
def log(msg): print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}')

def load_triplets(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            q, pos, neg = line.rstrip('\n').split('\t')
            data.append((q, pos, 1)); data.append((q, neg, 0))
    df = pd.DataFrame(data, columns=['query', 'doc', 'label'])
    return df

def build_embedding(glove_path, term_index_map, dim):
    vecs = {}
    vocab_size = len(term_index_map)
    mat = np.zeros((vocab_size, dim), dtype='float32')

    print(f"Building embedding matrix of shape ({vocab_size}, {dim}) from {glove_path}")

    parsed_lines = 0
    nan_vectors_found = 0
    value_error_tokens = 0

    with open(glove_path, encoding='utf-8') as f:
        for i, ln in enumerate(f):
            t, *nums_str = ln.rstrip().split()
            if len(nums_str) == dim:
                try:
                    vec = np.asarray(nums_str, dtype='float32')
                    if np.isnan(vec).any():
                        nan_vectors_found += 1
                    else:
                        vecs[t] = vec
                    parsed_lines += 1
                except ValueError:
                    value_error_tokens += 1

    if nan_vectors_found > 0:
        print(f"Warning: Found {nan_vectors_found} vectors with NaN values in GloVe. They will be treated as OOV (zero vectors).")
    if value_error_tokens > 0:
        print(f"Warning: Skipped {value_error_tokens} tokens due to ValueError during number parsing in GloVe.")
    print(f"Successfully parsed {parsed_lines} lines from GloVe file.")

    found_in_glove = 0
    for term, idx in term_index_map.items():
        if idx < vocab_size: 
            if term in vecs:
                mat[idx] = vecs[term]
                found_in_glove += 1

    print(f"Filled {found_in_glove} words in embedding matrix from GloVe.")
    
    norm = np.linalg.norm(mat, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    normalized_mat = mat / norm
    
    if np.isnan(normalized_mat).any():
        print("Error: NaNs detected in the final embedding matrix. This is unexpected.")
        normalized_mat[np.isnan(normalized_mat)] = 0.0

    return normalized_mat

# ---------- global config ---------- #
BASE_CFG = dict(batch=64, epochs=10, lr=1e-3,
                patience=10, early='mean_reciprocal_rank(0.0)') # Try MRR again

MODEL_CFG = {
    'KNRM': dict(embed_dim=100, glove='100',
                 loss=mz.losses.RankHingeLoss(),
                 pre='Basic', fixed=(10,40),
                 hyper={'kernel_num':21,'sigma':0.1,'exact_sigma':0.001},
                 data_mode='point'),
    # Other models...
}

ORDER = ['KNRM','MatchLSTM','ArcI','MatchPyramid','DRMM','ConvKNRM']

# ---------- data ---------- #
log('Loading data…')
df = load_triplets(cli.train)
train_df, dev_df = \
    np.split(df.sample(frac=1, random_state=SEED),
             [int(.9*len(df))])

# Rename columns for MatchZoo pack
train_df = train_df.rename(columns={'query': 'text_left', 'doc': 'text_right'})
dev_df = dev_df.rename(columns={'query': 'text_left', 'doc': 'text_right'})

train_raw, dev_raw = mz.pack(train_df), mz.pack(dev_df)
log(f'Pairs: {len(df)} | train={len(train_raw)} | dev={len(dev_raw)}')

# ---------- training loop ---------- #
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_flag = cli.start is None
for name in ORDER:
    if not start_flag:
        if name != cli.start:
            log(f'Skip {name} (waiting for --start model)…'); continue
        start_flag = True

    cfg = {**BASE_CFG, **MODEL_CFG[name]}
    out_dir = os.path.join(cli.output, name)
    os.makedirs(out_dir, exist_ok=True)
    succ_flag = os.path.join(out_dir, '_SUCCESS')
    if os.path.exists(succ_flag):
        log(f'{name} already trained – skip'); continue

    log(f'=== {name} ===')
    fixed_L, fixed_R = cfg['fixed']
    if cfg['pre']=='Basic':
        pre = mz.preprocessors.BasicPreprocessor(
                    truncated_length_left=fixed_L,
                    truncated_length_right=fixed_R,
                    filter_low_freq=2)
    else:
        model_cls = getattr(mz.models, name)
        try:
            pre = model_cls.get_default_preprocessor(
                    truncated_length_left=fixed_L,
                    truncated_length_right=fixed_R)
        except TypeError:
            pre = model_cls.get_default_preprocessor()
    pre.fit(train_raw)
    train_pack, dev_pack = pre.transform(train_raw), pre.transform(dev_raw)

    glove_path = cli.glove100 if cfg['glove']=='100' else cli.glove300
    emb_matrix_normalized = build_embedding(glove_path,
                                     pre.context['vocab_unit'].state['term_index'],
                                     cfg['embed_dim'])
    
    embedding = emb_matrix_normalized

    callbacks = []
    if name=='DRMM':
        emb_for_drmm_hist = emb_matrix_normalized.copy()
        is_zero_row = np.all(emb_for_drmm_hist == 0, axis=1)
        num_zero_rows = np.sum(is_zero_row)
        if num_zero_rows > 0:
            emb_for_drmm_hist[is_zero_row, 0] += 1e-8 
        cb_hist = mz.dataloader.callbacks.Histogram(
                    embedding_matrix=emb_for_drmm_hist, 
                    bin_size=cfg['hyper']['hist_bin_size'],
                    hist_mode='LCH')
        callbacks.append(cb_hist)

    if name=='ConvKNRM':
        pad_cb = mz.models.ConvKNRM.get_default_padding_callback(
                    fixed_length_left=fixed_L, fixed_length_right=fixed_R,
                    with_ngram=True)
    else:
        pad_cb = getattr(mz.models, name).get_default_padding_callback(
                    fixed_length_left=fixed_L, fixed_length_right=fixed_R)
    callbacks.append(pad_cb)

    mode = cfg['data_mode']
    log(f"Creating train_set for {name} with mode: {mode}")
    train_set = mz.dataloader.Dataset(train_pack, mode=mode,
                                      callbacks=callbacks,
                                      num_dup=(2 if mode=='pair' else 1),
                                      num_neg=(1 if mode=='pair' else 0),
                                      shuffle=True, batch_size=cfg['batch'])

    log(f"Creating dev_set for {name} with mode: {mode}")
    dev_set = None
    dev_loader = None  # Set to None if no validation set

    if dev_pack is not None and not dev_pack.relation.empty:
        try:
            current_dev_set = mz.dataloader.Dataset(
                data_pack=dev_pack,
                mode=mode,
                callbacks=callbacks,
                shuffle=False, 
                batch_size=cfg['batch']
            )
            dev_loader = mz.dataloader.DataLoader(dataset=current_dev_set)
        except ValueError as e:
            log(f"Could not create dev_set: {e}")

    train_loader = mz.dataloader.DataLoader(dataset=train_set)

    model_cls = getattr(mz.models, name)
    model = model_cls()
    model.params['task'] = mz.tasks.Ranking(
        metrics=[
            mz.metrics.MeanAveragePrecision(),
            mz.metrics.MeanReciprocalRank(),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=3),
            mz.metrics.NormalizedDiscountedCumulativeGain(k=5)
        ]
    )
    model.params['task'].loss = cfg['loss']
    model.params['embedding'] = embedding
    model.params.update(cfg['hyper'])
    model.guess_and_fill_missing_params(verbose=0)
    model.build(); model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'])

    # Handle trainer setup with or without dev_loader
    if dev_loader is not None:
        trainer = mz.trainers.Trainer(model=model, optimizer=optim,
                                      trainloader=train_loader,
                                      validloader=dev_loader,
                                      epochs=cfg['epochs'],
                                      patience=cfg['patience'],
                                      key=cfg['early'],
                                      save_dir=out_dir,
                                      save_all=True, clip_norm=None)
    else:
        trainer = mz.trainers.Trainer(model=model, optimizer=optim,
                                      trainloader=train_loader,
                                      validloader=None,
                                      epochs=cfg['epochs'],
                                      patience=cfg['epochs'], # Disable early stopping
                                      key=None,  # No key if no dev_loader
                                      save_dir=out_dir,
                                      save_all=True, clip_norm=None)

    try:
        trainer.run()
    except Exception as e:
        log(f'⚠️  Train failed for {name}: {e}')
        traceback.print_exc(); continue

    pre.save(os.path.join(out_dir, 'preprocessor.dill'))
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pt'))
    with open(os.path.join(out_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump({'params': model.params.completed(),
                   'best_metrics': trainer.best_metrics}, f, indent=2)
    open(succ_flag,'w').write(datetime.now().isoformat())
    log(f'{name} ✅ done\n')
