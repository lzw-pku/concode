from Statistics import Statistics
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm
import torch.nn as nn
import time
import random
import numpy as np

def make_batch_elem_into_tensor(batch, entry, pad):
  seq_len = max(len(elem[entry]) for elem in batch)
  torch_batch = np.full((len(batch), seq_len), pad) #torch.LongTensor(seq_len, len(batch)).fill_(pad)
  for i in range(0, len(batch)):
    for j in range(0, len(batch[i][entry])):
      torch_batch[i][j] = batch[i][entry][j]
  return torch.LongTensor(torch_batch)
def make_batch_into_tensor(batch, vocabs, max_camel):

    d['seq2seq'] = d['question'].split()
    d['next_rules'] = d['action_rule']
    d['prev_rules'] = [-1] + d['action_rule'][:-1]
    d['parent_rules'] = getParentRule(d['logical_form_tree'], -1)  # parentRules 是 parents获取的 其中第0个的parent是<s> 是vocab加的！！！！！
    d['seq2seq_vocab'] = Vocab(d['seq2seq'], 0, 100000000, start=False, stop=False)
    d['nt'] = [grammar.get_production_rule_by_id(rule).lhs for rule in d['action_rule']]


    d['seq2seq_nums'] = nlvocab.to_num(d['seq2seq'])
    d['seq2seq_in_src_nums'] = d['seq2seq_vocab'].to_num(nlvocab.addStartOrEnd(d['seq2seq']))
    # code_in_src_nums ??????
    d['next_rules_nums'] = d['next_rules']
    d['prev_rules_nums'] = d['prev_rules']
    d['parent_rules_nums'] = d['parent_rules']
    d['nt_nums'] = ntvocab.to_num(d['nt'])

    torch_batch = {}
    # -------- for seq2seq
    torch_batch['seq2seq'] = make_batch_elem_into_tensor(batch, 'seq2seq_nums', vocabs['seq2seq'].stoi['<blank>'])
    #torch_batch['code'] = make_batch_elem_into_tensor(batch, 'code_nums', vocabs['code'].stoi['<blank>'])
    local_vocab_blank = batch[0]['seq2seq_vocab'].stoi['<blank>']
    torch_batch['seq2seq_in_src'] = make_batch_elem_into_tensor(batch, 'seq2seq_in_src_nums', local_vocab_blank)
    # src_map maps positions in the source to source vocab entries, so that we can accumulate copy scores for each vocab entry based on all
    # positions in which it appears
    torch_batch['src_map'] = expandBatchOneHot(torch_batch['seq2seq_in_src'],
                                               local_vocab_blank)  # src token mapped to vocab


    # ---------------------------------------------
    #torch_batch['code_in_src_nums'] = make_batch_elem_into_tensor(batch, 'code_in_src_nums', local_vocab_blank)
    #torch_batch['next_rules_in_src_nums'] = make_batch_elem_into_tensor(batch, 'next_rules_in_src_nums',
    #                                                                    local_vocab_blank)
    torch_batch['seq2seq_vocab'] = [b['seq2seq_vocab'] for b in batch]  # Store this for replace unk
    #torch_batch['raw_code'] = [b['code'] for b in batch]  # Store this for replace unk
    #torch_batch['raw_seq2seq'] = [b['seq2seq'] for b in batch]  # Store this for replace unk
    # -------------------------Prod Decoder
    torch_batch['nt'] = make_batch_elem_into_tensor(batch, 'nt_nums', vocabs['nt'].stoi['<blank>'])
    torch_batch['prev_rules'] = make_batch_elem_into_tensor(batch, 'prev_rules_nums',
                                                            vocabs['prev_rules'].stoi['<blank>'])
    torch_batch['parent_rules'] = make_batch_elem_into_tensor(batch, 'parent_rules_nums',
                                                              vocabs['prev_rules'].stoi['<blank>'])

    torch_batch['next_rules'] = make_batch_elem_into_tensor(batch, 'next_rules_nums',
                                                            vocabs['next_rules'].stoi['<blank>'])
    torch_batch['seq2seq_copy'] = CDDataset.stack_with_padding([torch.LongTensor(b['seq2seq_copy']) for b in batch], 0,
                                                               start_symbol=True, stop_symbol=True)
    torch_batch['children'] = [b['children'] for b in batch]  # Store this for replace unk
    # ------------------------------

    # ---- Our Encoder --------------
    torch_batch['src'] = make_batch_elem_into_tensor(batch, 'src_nums', vocabs['names_combined'].stoi['<blank>'])
    torch_batch['varTypes'] = make_batch_elem_into_tensor(batch, 'varTypes_nums', vocabs['types'].stoi['<blank>'])
    torch_batch['methodReturns'] = make_batch_elem_into_tensor(batch, 'methodReturns_nums',
                                                               vocabs['types'].stoi['<blank>'])
    torch_batch['varNames'] = make_batch_char_elem_into_tensor(batch, 'varNames_nums',
                                                               pad=vocabs['names_combined'].stoi['<blank>'],
                                                               maxl=max_camel, minl=1)
    torch_batch['methodNames'] = make_batch_char_elem_into_tensor(batch, 'methodNames_nums',
                                                                  pad=vocabs['names_combined'].stoi['<blank>'],
                                                                  maxl=max_camel, minl=1)
    torch_batch['raw_src'] = [b['src'] for b in batch]  # Store this for replace unk
    torch_batch['raw_varNames'] = [b['varNames'] for b in batch]  # Store this for replace unk
    torch_batch['raw_methodNames'] = [b['methodNames'] for b in batch]  # Store this for replace unk
    # -------------------------------------

    return torch_batch
def compute_batches(examples, batch_size, vocabs, max_camel, rank, num_gpus, decoder_type, randomize=True, trunc=-1,
                    no_filter=False):
    timer = time.process_time()

    batches = []
    curr_batch = []
    total = 0
    for i in range(rank, len(examples), num_gpus):
        #if not no_filter and len(examples[i]['next_rules']) > 200:
        #    continue
        total += 1
        curr_batch.append(examples[i])
        if len(curr_batch) == batch_size or i == (len(examples) - 1) or i == trunc:
            batches.append(make_batch_into_tensor(curr_batch, vocabs, max_camel))
            curr_batch = []
        if i == trunc:
            break
        if i % 5000 == 0: print(i)
    if randomize:
        random.shuffle(batches)
    print('Computed batched in :' + str(time.process_time() - timer) + ' secs')
    return total, batches


class Trainer:
  def __init__(self, model):
    self.model = model
    self.opt = model.module.opt if isinstance(model, nn.parallel.DistributedDataParallel) else model.opt
    self.start_epoch = self.opt.start_epoch if self.opt.start_epoch  else 1

    self.lr = self.opt.learning_rate
    self.betas = [0.9, 0.98]
    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                betas=self.betas, eps=1e-9)

    if 'prev_optim' in self.opt:
      print('Loading prev optimizer state')
      self.optimizer.load_state_dict(self.opt.prev_optim)
      for state in self.optimizer.state.values():
        for k, v in state.items():
          if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

  def save_checkpoint(self, epoch, valid_stats):
      real_model = (self.model.module
                    if isinstance(self.model, nn.parallel.DistributedDataParallel)
                    else self.model)

      model_state_dict = real_model.state_dict()
      self.opt.learning_rate = self.lr
      checkpoint = {
          'model': model_state_dict,
          'vocab': real_model.vocabs,
          'opt':   self.opt,
          'epoch': epoch,
          'optim': self.optimizer.state_dict()
        }
      torch.save(checkpoint,
                 '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                 % (self.opt.save_model + '/model', valid_stats.accuracy(),
                    valid_stats.ppl(), epoch))

  def update_learning_rate(self, valid_stats):
    if self.last_ppl is not None and valid_stats.ppl() > self.last_ppl:
        self.lr = self.lr * self.opt.learning_rate_decay
        print("Decaying learning rate to %g" % self.lr)

    self.last_ppl = valid_stats.ppl()
    self.optimizer.param_groups[0]['lr'] = self.lr

  def run_train_batched(self, train_data, valid_data, vocabs):
    print(self.model.parameters)

    total_train = train_data.compute_batches(self.opt.batch_size, vocabs, self.opt.max_camel, 0, 1, self.opt.decoder_type,  trunc=self.opt.trunc)
    total_valid = valid_data.compute_batches(10 if self.opt.decoder_type in ["prod", "concode"] else self.opt.batch_size, vocabs, self.opt.max_camel, 0, 1, self.opt.decoder_type, randomize=False, trunc=self.opt.trunc)

    print('Computed Batches. Total train={}, Total valid={}'.format(total_train, total_valid))

    report_stats = Statistics()
    self.last_ppl = None

    for epoch in range(self.start_epoch, self.opt.epochs + 1):
      self.model.train()
      
      total_stats = Statistics()
      for idx, batch in enumerate(train_data.batches):
        loss, batch_stats = self.model.forward(batch)
        batch_size = batch['code'].size(0)
        loss.div(batch_size).backward()
        report_stats.update(batch_stats)
        total_stats.update(batch_stats)
        #print(batch_stats.loss, total_stats.loss)
        clip_grad_norm(self.model.parameters(), self.opt.max_grad_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()

        if (idx + 1) % self.opt.report_every == -1 % self.opt.report_every:
          report_stats.output(epoch, idx + 1, len(train_data.batches), total_stats.start_time)
          report_stats = Statistics()
      #print(total_stats.loss, total_stats.n_words, total_stats.ppl())
      print('Train perplexity: %g' % total_stats.ppl())
      print('Train accuracy: %g' % total_stats.accuracy())

      self.model.eval()
      valid_stats = Statistics()
      for idx, batch in enumerate(valid_data.batches):
        loss, batch_stats = self.model.forward(batch)
        valid_stats.update(batch_stats)

      print('Validation perplexity: %g' % valid_stats.ppl())
      print('Validation accuracy: %g' % valid_stats.accuracy())

      self.update_learning_rate(valid_stats)
      print('Saving model')
      self.save_checkpoint(epoch, valid_stats)
      print('Model saved')
