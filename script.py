

'''
from preprocess import CDDataset
import torch
import pickle
import re
from preprocess import *
def is_terminal_rule(rule):
    return ("IdentifierNT-->" in rule and rule != 'IdentifierNT-->VarCopy' and rule != 'IdentifierNT-->MethodCopy') \
           or re.match(r"^Nt_.*_literal-->.*", rule) \
           or rule == "<unk>"


def getAnonRule(rule):
    return "Identifier_OR_Literal" if is_terminal_rule(rule) else rule

dataset = torch.load('data/d_100k_762/concode.train.pt')

examples = dataset.examples
data = []
for e in examples:
    action_rule = [getAnonRule(rule) for rule in e['next_rules']]
    d = {'question': e['src'],
         'query': ' '.join(e['origcode']),
         'action_rule': action_rule}
    data.append(d)

with open('data/dataset.pkl', 'wb') as f:
    pickle.dump(data, f)
'''
