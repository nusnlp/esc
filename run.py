import argparse
from collections import OrderedDict
import datetime
import json
import random
from os import listdir, makedirs
from os.path import basename, isdir, isfile, join, splitext

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from file_utils import read_data


def create_vocab(m2_dir, data_dir, source_name, target_name):
    """
    Generate a vocabulary of edit types
    """
    src_path = join(data_dir, source_name)
    target_path = join(data_dir, target_name)
    target_m2 = read_data(src_path, target_path, m2_dir)
    edit_types = set([])
    for instance in target_m2:
        edit_types |= set([e[2] for e in instance['edits']])
    edit_types = sorted(list(edit_types))
    edit_types = {e: i for i, e in enumerate(edit_types)}

    return edit_types


class M2Dataset(Dataset):
    def __init__(self, m2_dir, data_dir, source_name, target_name, vocab,
                    filter_idx=None, test=False, upsample=None):
        """
        Read all the files from data_dir, but if the files with same name
        (but with .m2 extension) exists in m2_dir, the program will read
        the one in m2_dir instead.

        This class serves as the generator for the dataset. This object
        is then used by PyTorch data generator to supply the training
        instance

        If test = false:
            The dataset is in the form of list of edits, along with the labels.
            The total length of the dataset is the number of all edits from
            all sentences from all hypotheses.
        If test = true:
            the dataset is in the form of list of edits within a sentence, from
            all hypotheses. Thus, the total length of the dataset is the same
            as the number of sentences.
        Args:
            m2_dir (string): Path to the directory containing .m2 files
            data_dir (string): Path to the text data directory
            source_name (string): the filename to be the source reference
                when generating the .m2 files
            target_name (string): the filename to be the hypothesis reference
                when generating the .m2 files to get the edit labels
            vocab (dict): the vocabulary of edit_types and hypothesis list
            filter_idx: the indexes to be used as part of the dataset. This is
                especially useful during cross-validation
            test (boolean): a flag denoting if the Dataset is a testing type.
                In testing type, the edits are grouped into
            upsample (string): a string in the format of <label 0>:<label 1>
                ratio to upsample the data
        """
        self.test = test
        if not isdir(m2_dir):
            makedirs(m2_dir)
        
        src_path = join(data_dir, source_name)
        
        if not test and target_name is not None:
            target_path = join(data_dir, target_name)
            target_m2 = read_data(src_path, target_path, m2_dir, filter_idx=filter_idx)
        else:
            target_m2 = None

        self.edit_types = vocab['edit_types']
        self.hyp_list = vocab['hyp_list']

        data = []
        for file_name in self.hyp_list:
            print('Loading {}...'.format(file_name))
            file_path = join(data_dir, file_name)
            hyp_data = read_data(src_path, file_path, m2_dir, target_m2, filter_idx)
            data.append(hyp_data)
        
        doc_lens = [len(d) for d in data]
        assert min(doc_lens) == max(doc_lens), "M2 lengths are different!"

        self.data, self.labels = self.transform(data, self.edit_types, test)
       
        if not test:
            self.label_counts()
            print('Label distribution: ', self.label_count)

            if upsample is not None:
                try:
                    ratios = [float(f) for f in upsample.split(':')]
                    scale = 1.0 / min(ratios)
                    ratios = [r * scale for r in ratios]
                except:
                    assert ValueError("Please provide the ratio in the format of class 0:class 1, e.g. 1:2")
                for class_id, ratio in enumerate(ratios):
                    label_idx = [i for i in range(len(self.labels)) \
                                    if self.labels[i] == class_id]
                    add_ratio = ratio - 1
                    if add_ratio > 0:
                        num_sample = round(add_ratio * len(label_idx))
                        print('Found {} instance of class {}, adding {} more'.format(len(label_idx), class_id, num_sample))
                        label_idx = random.sample(label_idx, num_sample)
                        add_data = [self.data[i] for i in label_idx]
                        self.data += add_data
                        add_labels = [self.labels[i] for i in label_idx]
                        self.labels += add_labels
            
            self.label_counts()
            print('New distribution: ', self.label_count)


    def label_counts(self):
        label_count = [0, 0]
        label1 = sum(self.labels)
        label0 = len(self.labels) - label1
        label_count[1] += label1
        label_count[0] += label0
        self.label_count = label_count


    def feature_size(self):
        print(self.f_size)
        return self.f_size
        

    def transform(self, data, edit_types, test=False):
        data = zip(*data)
        all_features = []
        if test:
            self.all_edits = []
        
        labels = []
        for entity in data:
            hyps = list(entity)
            assert min([hyps[0]['source'] == h['source'] for h in hyps]), "Sources are different!"

            en_edits = OrderedDict()
            for h_idx, hyp in enumerate(hyps):
                h_edits = hyp['edits']
                if 'labels' in hyp:
                    h_labels = hyp['labels']
                else:
                    h_labels = [None] * len(h_edits)
                for edit, label in zip(h_edits, h_labels):
                    e_start, e_end, e_type, e_cor = edit
                    edit_key = (e_start, e_end, e_cor)
                    if edit_key not in en_edits:
                        en_edits[edit_key] = [(h_idx, e_type, label)]
                    else:
                        en_edits[edit_key].append((h_idx, e_type, label))

            en_features = []
            en_labels = []
            for _, edits in en_edits.items():
                feature = [0] * len(edit_types) * len(hyps)
                self.f_size = len(feature)
                e_label = -999

                for edit in edits:
                    h_idx, e_type, label = edit

                    if e_type in edit_types:
                        f_idx = h_idx * len(edit_types) + edit_types[e_type]
                        feature[f_idx] = 1
                    if label is not None:
                        if e_label == -999:
                            e_label = label
                        else:
                            assert e_label == label, "Labels are different"

                en_features.append(feature)
                en_labels.append(e_label)
            
            if test:
                self.all_edits.append(
                    {'source': hyps[0]['source'], 'edits': en_edits}
                )
                all_features.append(en_features)
                labels.append(en_labels)
            else:
                all_features.extend(en_features)
                labels.extend(en_labels)


        return all_features, labels


    def __len__(self):
        """
        Return the number of instances in the data
        """
        return len(self.data)


    def __getitem__(self, idx):
        feature = torch.tensor(self.data[idx], dtype=torch.float)
        label = self.labels[idx]
        if label is not None or (isinstance(label, list) and len(label) > 0 and label[0] is not None):
            label = torch.tensor(label, dtype=torch.float)
        
        return feature, label


class Model(nn.Module):
    """
    A very simple linear model
    """
    def __init__(self, feature_length):
        super().__init__()
        self.linear = nn.Linear(feature_length, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.sigmoid(x)

        return x


def train(model, train_dataset, batch_size, lr, weight_decay, num_epoch, device,
            model_path=None, eval_dataset=None, save_last=False, verbose=False):
    """
    Train the model and save the best checkpoint
    """
    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    start = datetime.datetime.now()
    metric = 'f0.5'
    best_score = 0
    best_epoch = 0
    for epoch in range(num_epoch):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            features = data[0].to(device)
            labels = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # do forward propagation
            outputs = model(features)
            outputs = outputs.squeeze(-1)
            loss = criterion(outputs, labels)

            # do backward propagation
            loss.backward()

            # do the parameter optimization
            optimizer.step()

            # calculate running loss value for non padding
            running_loss += loss.item()

            # print loss value every 100 iterations and reset running loss
            if verbose and step % 100 == 99:
                # print('outputs: {}\nlabels: {}\n'.format(outputs, labels))
                print('[%d, %3d] loss: %.3f' %
                    (epoch + 1, step + 1, running_loss / 100))
                running_loss = 0.0
        if eval_dataset is not None:
            result = eval(model, eval_dataset, device)
            if metric in result:
                score = result[metric]
                if verbose:
                    print('[{}] Accuracy: {}, F0.5: {}'.format(epoch, result['acc'], result['f0.5']))
                if score > best_score:
                    best_score = score
                    best_epoch = epoch # 0-based index
                    if model_path is not None:
                        checkpoint = {
                            'edit_types': train_dataset.edit_types,
                            'hyp_list': train_dataset.hyp_list,
                            'model_state_dict': model.state_dict()
                        }
                        torch.save(checkpoint, model_path)
                        print('Model with {} accuracy saved on {}'.format(score, model_path))
            else:
                if verbose:
                    print('No accuracy found. No model will be saved.')
    end = datetime.datetime.now()
    if save_last:
        checkpoint = {
            'edit_types': train_dataset.edit_types,
            'hyp_list': train_dataset.hyp_list,
            'model_state_dict': model.state_dict()
        }
        torch.save(checkpoint, model_path)
    if verbose:
        print('== best checkpoint ({}) from epoch {} saved in {}'.format(best_score, best_epoch, model_path))
        print('Training finished in {} minutes.'.format((end - start).seconds / 60.0))

    return best_score, best_epoch


def eval(model, dataset, device='cpu'):
    """
    Evaluation function to get an estimated F0.5 score to save
    the best checkpoint during training.
    """
    model.eval()
    data_loader = DataLoader(dataset, batch_size=20, shuffle=False)

    with torch.no_grad():
        tp = 0
        tn = 0
        p = 0
        true_edits = 0
        total_data = 0
        result = {
            'preds': []
        }
        for data in data_loader:
            features = data[0].to(device)
            labels = data[1]
            if labels is not None:
                labels = labels.to(device)
            outputs = model(features)
            outputs = outputs.squeeze(-1)
            # print('outputs: {}\nlabels: {}\n'.format(outputs, labels))
            preds = torch.round(outputs)
            result['preds'].append(preds)
            if labels is not None:
                p += torch.sum(preds)
                true_edits += torch.sum(labels)
                tp += torch.sum((preds > 0) & (labels > 0))
                tn += torch.sum((preds == 0) & (labels == 0))
                # print('preds: {}\nlabels: {}\ntp: {}\n'.format(preds, labels, (preds == labels)))
                # print(torch.sum(preds), torch.sum(labels), torch.sum(preds == labels))
                total_data += len(labels)
                
        precision = 1 if p == 0 else float(tp) / p
        recall = 1 if true_edits == 0 else float(tp) / true_edits
        f_half = 0 if precision + recall == 0 else (1 + 0.5 * 0.5) * precision * recall / (0.5 * 0.5 * precision + recall)
        result['preds'] = torch.cat(result['preds'])
        result['acc'] = float(tp + tn) / total_data
        result['prec'] = precision
        result['rec'] = recall
        result['f0.5'] = f_half

    return result

def test(model, model_path, dataset, device, threshold=0.5):
    """
    A test function to predict the appropriate edit and apply it
    to the original sentence, resulting a corrected sentence
    """
    model_paths = model_path.split(',')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    raw_data = dataset.all_edits
    
    result = [None] * len(data_loader)
    all_outputs = []
    with torch.no_grad():
        for model_path in model_paths:
            print('Getting predictions from {}...'.format(model_path))
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            model_output = []
            for idx, data in enumerate(data_loader):
                edits = raw_data[idx]['edits']
                features = data[0].squeeze(0).to(device)
                if len(features.shape) < 2 or features.shape[1] == 0:
                    result[idx] = raw_data[idx]['source']
                    model_output.append(None)
                    continue
                outputs = model(features).squeeze(-1)
                assert len(outputs) == len(edits), \
                    "The length of outputs ({}) is different from edits ({})"\
                        .format(len(outputs), len(edits))
                model_output.append(outputs)
            all_outputs.append(model_output)
    
    all_outputs = list(zip(*all_outputs))
    all_outputs = [None if l[0] is None else torch.stack(list(l), dim=0).mean(dim=0) for l in all_outputs]

    for idx, output in enumerate(all_outputs):
        if output is None:
            continue
        source = raw_data[idx]['source'].split()
        edits = raw_data[idx]['edits']
        offset = 0
        edits_to_apply = []
        for edit, pred in zip(edits.keys(), output):
            if pred >= threshold:
                e_start, e_end, rep_token = edit
                edits_to_apply.append((e_start, e_end, rep_token, pred))

        edits_to_apply = sorted(edits_to_apply, key=lambda x: x[3], reverse=True)
        filtered_edits = []
        multiple_insertion = lambda x, y: x[0] == x[1] == y[0] == y[1]
        intersecting_range = lambda x, y: (x[0] <= y[0] < x[1] and not x[0] == y[1]) or \
                                            (y[0] <= x[0] < y[1] and not y[0] == x[1])
        for edit in edits_to_apply:
            eligible = True
            for selected_edit in filtered_edits:
                if multiple_insertion(edit, selected_edit) \
                    or intersecting_range(edit, selected_edit):
                    eligible = False
            if eligible:
                filtered_edits.append(edit)
        filtered_edits = sorted(filtered_edits)

        for edit in filtered_edits:
            e_start, e_end, rep_token, pred = edit
            e_cor = rep_token.split()
            len_cor = 0 if len(rep_token) == 0 else len(e_cor)
            source[e_start + offset:e_end + offset] = e_cor
            offset = offset - (e_end - e_start) + len_cor
        result[idx] = ' '.join(source)

    return result


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda:{}'.format(0)

    device = torch.device(device_str)
    if args.train:
        edit_types = create_vocab(args.m2_dir,
                                    args.data_dir,
                                    args.source_name,
                                    args.target_name
                                )
        hyp_list = [f for f in listdir(args.data_dir) if isfile(join(args.data_dir, f)) \
            and basename(f) not in [args.source_name, args.target_name]]
        vocab = {
            'edit_types': edit_types,
            'hyp_list': hyp_list,
        }
        with open(args.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, indent=2)
        kf = KFold(n_splits=args.val_ratio, shuffle=True, random_state=args.seed)
        dummy_file = [1 for _ in open(join(args.data_dir, args.source_name), encoding='utf-8')]

        _BATCH_SIZE = 16
        _LR = args.lr
        _EPOCH = 100

        split = kf.split(dummy_file)
        train_index, test_index = next(split)
        
        # get number of epoch
        train_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                filter_idx=train_index,
                                upsample=args.upsample,
                                )
        feature_size = train_dataset.feature_size()
        model = Model(feature_size).to(device)
        eval_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                filter_idx=test_index,
                                )

        _score, best_epoch = train(model, train_dataset, _BATCH_SIZE, _LR, args.weight_decay, _EPOCH,
                device, eval_dataset=eval_dataset)
        # full training
        torch.manual_seed(args.seed)
        print('Best checkpoint at epoch {}. Training on full dataset.'.format(best_epoch))
        model_path = join(args.model_path, 'model.pt')
        train_dataset = M2Dataset(args.m2_dir,
                                args.data_dir,
                                args.source_name,
                                args.target_name,
                                vocab,
                                upsample=args.upsample,
                                )
        feature_size = train_dataset.feature_size()
        model = Model(feature_size).to(device)
        train(model, train_dataset, _BATCH_SIZE, _LR, args.weight_decay, best_epoch,
                device, model_path, save_last=True)
        print('Finished training.')
    elif args.test:
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        test_dataset = M2Dataset(args.m2_dir,
                                 args.data_dir,
                                 args.source_name,
                                 args.target_name,
                                 vocab,
                                 test=True,
                                )
        feature_size = test_dataset.feature_size()
        model = Model(feature_size).to(device)
        sentences = test(model, args.model_path, test_dataset, device, threshold=args.threshold)
        with open(args.output_path, 'w', encoding='utf-8') as out:
            out.write('\n'.join(sentences))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='path to the data directory')
    parser.add_argument('--m2_dir', default='m2', help='path to the generated m2 files')
    parser.add_argument('--source_name', default='source.txt', help='The source filename')
    parser.add_argument('--target_name', default='target.txt', help='The target filename')
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="weight decay (L2 penalty)")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--val_ratio', type=int, default=5, help="1/val_ratio of the data is for validation")
    parser.add_argument('--threshold', type=float, default=0.5, help="probability threshold")
    parser.add_argument('--upsample', type=str, default=None, help='up-sample ratio of class 0:class 1')
    parser.add_argument('--train', default=False, action='store_true', help='train the model')
    parser.add_argument('--test', default=False, action='store_true', help='test the model')
    parser.add_argument('--target_path', help='path to the target file during training')
    parser.add_argument('--vocab_path', default='vocab.idx', help='path to the vocab file')
    parser.add_argument('--model_path', required=True, help='path to the model directory')
    parser.add_argument('--output_path', default='out.txt', help='path to the output file during testing')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_arguments()
    main(args)
