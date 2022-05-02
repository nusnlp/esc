from os import listdir, makedirs
from os.path import basename, isdir, isfile, join, splitext
import subprocess


_IGNORE_TYPE = {"noop", "UNK", "Um"}
_EDIT_START = 0
_EDIT_END = 1
_EDIT_TYPE = 2
_EDIT_COR = 3


def parse_m2(src, cor, m2_path):
    command = "errant_parallel -orig {orig} -cor {cor} -out {out}".format(orig=src, cor=cor, out=m2_path)
    subprocess.run(command, shell=True, check=True)


def apply_edits(source, edits, offset=0):
    if isinstance(source, str):
        source = source.split(' ')
    result, offset = apply_edits_list(source, edits, offset)
    return ' '.join(result)


def apply_edits_list(source, edits, offset=0):
    for edit in edits:
        e_start = edit[_EDIT_START]
        e_end = edit[_EDIT_END]
        rep_token = edit[_EDIT_COR]

        e_cor = rep_token.split()
        len_cor = 0 if len(rep_token) == 0 else len(e_cor)
        source[e_start + offset:e_end + offset] = e_cor
        offset = offset - (e_end - e_start) + len_cor
    return source, offset


def read_m2(filepath, filter_idx=None):
    with open(filepath, encoding='utf-8') as f:
        m2_entries = f.read().strip().split('\n\n')
    
    if filter_idx is not None:
        m2_entries = [m2_entries[i] for i in filter_idx]
        # m2_entries = [m for i, m in enumerate(m2_entries) if i in filter_idx]
    parsed_data = []
    for m2_entity in m2_entries:
        m2_lines = m2_entity.split('\n')
        source = m2_lines[0][2:]
        edits = []
        for m2_line in m2_lines[1:]:
            if not m2_line.startswith("A"):
                raise ValueError("{} is not an m2 edit".format(m2_line))
            m2_line = m2_line[2:]
            features = m2_line.split("|||")
            span = features[0].split()
            start, end = int(span[0]), int(span[1])
            error_type = features[1].strip()
            if error_type in _IGNORE_TYPE:
                continue
            replace_token = features[2]
            edits.append((start, end, error_type, replace_token))
        parsed_data.append({'source': source, 'edits': edits})
    
    return parsed_data


def read_data(src_path, file_path, m2_dir, target_m2=None, filter_idx=None):
    m2_path = join(m2_dir, splitext(basename(file_path))[0] + '.m2')

    if not isfile(m2_path):
        parse_m2(src_path, file_path, m2_path)
    
    hyp_m2 = read_m2(m2_path, filter_idx)

    if target_m2 is not None:
        assert len(target_m2) == len(hyp_m2), \
            "The m2 lengths of target ({}) and hypothesis ({}) are different!"\
                .format(len(target_m2), len(hyp_m2))
        for hyp_entry, trg_entry in zip(hyp_m2, target_m2):
            assert hyp_entry['source'] == trg_entry['source']
            hyp_edits = hyp_entry['edits']
            trg_edits = set([(t[_EDIT_START], t[_EDIT_END], t[_EDIT_COR]) for t in trg_entry['edits']])
            labels = []
            for edit in hyp_edits:
                e_start, e_end, e_type, e_cor = edit
                label = 1 if (e_start, e_end, e_cor) in trg_edits else 0
                labels.append(label)
            hyp_entry['labels'] = labels
    
    return hyp_m2