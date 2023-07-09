from cProfile import label
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import pandas as pd
import operator
import argparse
import numpy as np
import logging
import random
import pickle
import json
import os
from pathlib import Path
import gc
import sys
from datetime import datetime
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm, trange
from _config.net import KobertBiLSTMCRF, KcbertBiLSTMCRF

sys.path.append(os.path.abspath(os.path.dirname('_git_src/pytorch-bert-crf-ner/')))
from data_utils.utils import CheckpointManager, SummaryManager
from model.utils import Config
from data_utils.ner_dataset import NamedEntityRecognitionDataset, NamedEntityRecognitionFormatter
from data_utils.vocab_tokenizer import Vocabulary, Tokenizer
from data_utils.pad_sequence import keras_pad_fn
from gluonnlp.data import SentencepieceTokenizer
from kobert.utils import get_tokenizer
from sklearn.metrics import classification_report
from transformers import AutoTokenizer
# pytorch-bert-crf-ner
import gluonnlp as nlp
from kobert_tokenizer import KoBERTTokenizer

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use gpu 0, gpu 1, gpu 2, ... (gpu index)

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')
logger = logging.getLogger(__name__)


def set_seed(seed=100):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def main(parser):
    # Config
    args = parser.parse_args()
    if args.model_type == 'kcbert':
        model_dir = Path('_result/kcbert_model_with_bilstm_crf')
    else:
        model_dir = Path('_result/kobert_model_with_bilstm_crf')

    model_config = Config(json_path=model_dir / 'config.json')
    model_config.output_dir = os.path.join(model_config.output_dir, datetime.today().strftime("%y%m%d_%H%M%S"))
    # Vocab & Tokenizer
    tok_path = './_git_src/pytorch-bert-crf-ner/ptr_lm_model/tokenizer_78b3253a26.model'  # ./tokenizer_78b3253a26.model
    ptr_tokenizer = SentencepieceTokenizer(tok_path)
    # 모델 다운로드 서비스 중단으로 인해 hugging face를 통해 다운로드
    if args.model_type == 'kcbert':
        tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        vocab = AutoTokenizer.from_pretrained("beomi/kcbert-base")
        token_to_idx = vocab.vocab
        model_config.vocab_size = len(token_to_idx)
        vocab = Vocabulary(token_to_idx=token_to_idx)
    else:
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
        vocab_of_gluonnlp = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
        token_to_idx = vocab_of_gluonnlp.token_to_idx
        model_config.vocab_size = len(token_to_idx)
        vocab = Vocabulary(token_to_idx=token_to_idx)

    print("len(token_to_idx): ", len(token_to_idx))
    with open(model_dir / "token2idx_vocab.json", 'w', encoding='utf-8') as f:
        json.dump(token_to_idx, f, ensure_ascii=False, indent=4)

    # save vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'wb') as f:
        pickle.dump(vocab, f)

    # load vocab & tokenizer
    with open(model_dir / "vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)

    tokenizer = Tokenizer(vocab=vocab, split_fn=ptr_tokenizer, pad_fn=keras_pad_fn, maxlen=model_config.maxlen)
    ner_formatter = NamedEntityRecognitionFormatter(
        vocab=vocab, tokenizer=tokenizer, maxlen=model_config.maxlen, model_dir=model_dir)

    # Train & Val Datasets
    cwd = Path.cwd()
    data_in = cwd / "data_in"
    train_data_dir = cwd / "_data" / "train_set"
    tr_ds = NamedEntityRecognitionDataset(
        train_data_dir=train_data_dir, model_dir=model_dir)
    tr_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn,
                           transform_target_fn=ner_formatter.transform_target_fn)
    tr_dl = DataLoader(tr_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=0, drop_last=False)

    val_data_dir = cwd / "_data" / "validation_set"
    val_ds = NamedEntityRecognitionDataset(
        train_data_dir=val_data_dir, model_dir=model_dir)
    val_ds.set_transform_fn(transform_source_fn=ner_formatter.transform_source_fn,
                            transform_target_fn=ner_formatter.transform_target_fn)
    val_dl = DataLoader(val_ds, batch_size=model_config.batch_size, shuffle=True, num_workers=0, drop_last=False)
    gc.collect()
    torch.cuda.empty_cache()
    # Model

    if args.model_type == 'kcbert':
        # KcBertt + bi-lstm + crf
        model = KcbertBiLSTMCRF(config=model_config, num_classes=len(tr_ds.ner_to_index))
    else:
        # KoBERT + bi-lstm + crf
        model = KobertBiLSTMCRF(config=model_config, num_classes=len(tr_ds.ner_to_index), time_distribute=False)

    model.to(device)
    model.train()

    # optim
    train_examples_len = len(tr_ds)
    val_examples_len = len(val_ds)
    print("num of train: {}, num of val: {}".format(
        train_examples_len, val_examples_len))

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    # num_train_optimization_steps = int(train_examples_len / model_config.batch_size / model_config.gradient_accumulation_steps) * model_config.epochs
    t_total = len(
        tr_dl) // model_config.gradient_accumulation_steps * model_config.epochs
    optimizer = AdamW(optimizer_grouped_parameters, lr=model_config.learning_rate, eps=model_config.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=model_config.warmup_steps, t_total=t_total)

    n_gpu = torch.cuda.device_count()
    # if n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # save
    tb_writer = SummaryWriter('{}/runs/'.format(model_config.output_dir))
    checkpoint_manager = CheckpointManager(model_config.output_dir)
    summary_manager = SummaryManager(model_config.output_dir)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(tr_ds))
    logger.info("  Num Epochs = %d", model_config.epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                model_config.batch_size)
    logger.info("  Gradient Accumulation steps = %d",
                model_config.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_steps = 0
    model.zero_grad()
    set_seed()  # Added here for reproductibility (even between python 2 and 3)
    torch.cuda.empty_cache()
    # Train
    train_iterator = trange(int(model_config.epochs), desc="Epoch")
    for _epoch, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(tr_dl, desc="Iteration")
        epoch = _epoch
        for step, batch in enumerate(epoch_iterator):
            model.train()
            print(model.crf)
            x_input, token_type_ids, y_real = map(lambda elm: elm.to(device), batch)
            log_likelihood, sequence_of_tags = model(x_input, token_type_ids, y_real)

            # loss: negative log-likelihood
            loss = -1 * log_likelihood

            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if model_config.gradient_accumulation_steps > 1:
                loss = loss / model_config.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), model_config.max_grad_norm)
            tr_loss += loss.item()

            if (step + 1) % model_config.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                with torch.no_grad():
                    sequence_of_tags = torch.tensor(
                        sequence_of_tags).to(device)
                    mb_acc = (sequence_of_tags == y_real).float()[
                        y_real != vocab.PAD_ID].mean()

                tr_acc = mb_acc.item()
                tr_loss_avg = tr_loss / global_step
                tr_summary = {'loss': tr_loss_avg, 'acc': tr_acc}

                # if step % 50 == 0:
                print('epoch : {}, global_step : {}, tr_loss: {:.3f}, tr_acc: {:.2%}'.format(
                    epoch + 1, global_step, tr_summary['loss'], tr_summary['acc']))

                # training & evaluation log
                if model_config.logging_steps > 0 and global_step % model_config.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if model_config.evaluate_during_training:
                        eval_summary, list_of_y_real, list_of_pred_tags = evaluate(model, val_dl)
                        tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalars('loss', {'train': (tr_loss - logging_loss) / model_config.logging_steps, 'val': eval_summary["eval_loss"]}, global_step)
                        tb_writer.add_scalars('acc', {'train': tr_acc, 'val': eval_summary["eval_acc"]}, global_step)
                        print("eval acc: {}, loss: {}, global steps: {}".format(eval_summary['eval_acc'], eval_summary['eval_loss'], global_step))
                    print("Average loss: {} at global step: {}".format((tr_loss - logging_loss) / model_config.logging_steps, global_step))
                    logging_loss = tr_loss

                # save model
                if model_config.save_steps > 0 and global_step % model_config.save_steps == 0:
                    eval_summary, list_of_y_real, list_of_pred_tags = evaluate(
                        model, val_dl)

                    # Save model checkpoint
                    print("Saving model checkpoint to %s", model_config.output_dir)
                    if isinstance(model, nn.DataParallel):
                        state = {'global_step': global_step + 1,
                                 'model_state_dict': model.module.state_dict(),
                                 'opt_state_dict': optimizer.state_dict()}
                    else:
                        state = {'global_step': global_step + 1,
                                 'model_state_dict': model.state_dict(),
                                 'opt_state_dict': optimizer.state_dict()}
                    summary = {'train': tr_summary, 'eval': eval_summary}
                    summary_manager.update(summary)
                    print("summary: ", summary)
                    summary_manager.save('summary.json')
                    # Save
                    # acc 기준 (원래는 train_acc가 아니라 val_acc로 해야)
                    is_best = eval_summary["eval_acc"] >= best_dev_acc
                    if is_best:
                        best_dev_acc = eval_summary["eval_acc"]
                        best_dev_loss = eval_summary["eval_loss"]
                        best_steps = global_step
                        # if args.do_test:
                        # results_test = evaluate(model, test_dl, test=True)
                        # for key, value in results_test.items():
                        #     tb_writer.add_scalar('test_{}'.format(key), value, global_step)
                        # logger.info("test acc: %s, loss: %s, global steps: %s", str(eval_summary['eval_acc']), str(eval_summary['eval_loss']), str(global_step))

                        checkpoint_manager.save_checkpoint(
                            state, 'best-epoch-{}-step-{}-acc-{:.3f}-loss-{:.3f}.pth'.format(epoch + 1, global_step, best_dev_acc, best_dev_loss))
                        print("Saving model checkpoint as best-epoch-{}-step-{}-acc-{:.3f}-loss-{:.3f}.pth".format(
                            epoch + 1, global_step, best_dev_acc, best_dev_loss))

                        # print classification report and save confusion matrix
                        cr_save_path = Path(model_config.output_dir) / \
                            'best-epoch-{}-step-{}-acc-{:.3f}-loss-{:.3f}-cr.csv'.format(epoch + 1, global_step, best_dev_acc, best_dev_loss)
                        cm_save_path = Path(model_config.output_dir) / \
                            'best-epoch-{}-step-{}-acc-{:.3f}-loss-{:.3f}-cm.png'.format(epoch + 1, global_step, best_dev_acc, best_dev_loss)
                        save_cr_and_cm(val_dl, list_of_y_real, list_of_pred_tags,
                                       len_index=len(tr_ds.ner_to_index), cr_save_path=cr_save_path, cm_save_path=cm_save_path)

    tb_writer.close()
    print("global_step = {}, average loss = {}".format(global_step, tr_loss / global_step))

    return global_step, tr_loss / global_step, best_steps


def evaluate(model, val_dl, prefix="NER"):
    """ evaluate accuracy and return result """
    results = {}
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    eval_loss = 0.0
    nb_eval_steps = 0

    list_of_y_real = []
    list_of_pred_tags = []
    count_correct = 0
    total_count = 0

    for batch in tqdm(val_dl, desc="Evaluating"):
        model.train()
        x_input, token_type_ids, y_real = map(
            lambda elm: elm.to(device), batch)
        with torch.no_grad():
            inputs = {'input_ids': x_input,
                      'token_type_ids': token_type_ids,
                      'tags': y_real}
            log_likelihood, sequence_of_tags = model(**inputs)

            eval_loss += -1 * log_likelihood.float().item()
        nb_eval_steps += 1

        y_real = y_real.to('cpu')
        sequence_of_tags = torch.tensor(sequence_of_tags).to('cpu')
        # 0,1,2,3 -> [CLS], [SEP], [PAD], [MASK] index
        count_correct += (sequence_of_tags == y_real).float()[y_real != 2].sum()
        total_count += len(y_real[y_real != 2])

        for seq_elm in y_real.tolist():
            list_of_y_real += seq_elm

        for seq_elm in sequence_of_tags.tolist():
            list_of_pred_tags += seq_elm

    eval_loss = eval_loss / nb_eval_steps
    acc = (count_correct / total_count).item()  # tensor -> float
    result = {"eval_acc": acc, "eval_loss": eval_loss}
    results.update(result)

    return results, list_of_y_real, list_of_pred_tags


def save_cr_and_cm(val_dl, list_of_y_real, list_of_pred_tags, len_index, cr_save_path="classification_report.csv", cm_save_path="confusion_matrix.png"):
    """ print classification report and confusion matrix """

    # target_names = val_dl.dataset.ner_to_index.keys()
    sorted_ner_to_index = sorted(
        val_dl.dataset.ner_to_index.items(), key=operator.itemgetter(1))
    target_names = []
    for ner_tag, index in sorted_ner_to_index:
        if ner_tag in ['[CLS]', '[SEP]', '[PAD]', '[MASK]', 'O']:
            continue
        else:
            target_names.append(ner_tag)

    # ner label indice except '[CLS]', '[SEP]', '[PAD]', '[MASK]' and 'O' tag
    #label_index_to_print = list(range(5, 25))
    label_index_to_print = list(range(5, len_index))
    print(classification_report(y_true=list_of_y_real, y_pred=list_of_pred_tags,
                                target_names=target_names, labels=label_index_to_print, digits=4))
    cr_dict = classification_report(y_true=list_of_y_real, y_pred=list_of_pred_tags,
                                    target_names=target_names, labels=label_index_to_print, digits=4, output_dict=True)
    df = pd.DataFrame(cr_dict).transpose()
    df.to_csv(cr_save_path)
    np.set_printoptions(precision=2)
    plot_confusion_matrix(y_true=list_of_y_real, y_pred=list_of_pred_tags, classes=target_names,
                          labels=label_index_to_print, normalize=False, title='Confusion matrix, without normalization')
    plt.savefig(cm_save_path)
    # plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, labels,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # Only use the labels that appear in the data

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # --- plot 크기 조절 --- #
    plt.rcParams['savefig.dpi'] = 200
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['figure.figsize'] = [20, 20]  # plot 크기
    plt.rcParams.update({'font.size': 10})
    # --- plot 크기 조절 --- #

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # --- bar 크기 조절 --- #
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # --- bar 크기 조절 --- #
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes,
           yticks=np.arange(cm.shape[0]),
           yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='kobert')

    main(parser)
