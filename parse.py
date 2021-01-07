import re


class LogicalFormNode:
    def __init__(self, id, nt, nonterminals):
        self.id = id
        self.nt = nt
        self.nonterminals = nonterminals
        self.childList = []

    def countIdiom(self, idiomCount, depth=0):
        #print('--' * depth, self.grammar.get_production_rule_by_id(node.id), node.id)
        for child in self.childList:
            idiomCount[(self.id, child.id)] += 1
            child.countIdiom(idiomCount, depth + 1)
        assert self.nonterminals == [child.nt for child in self.childList]

    def addIdiom(self, prod1, prod2, newProd):
        if self.id == prod1.rule_id and prod2.rule_id in [child.id for child in self.childList]:
            newChildList = []
            newNonterminal = []
            flag = True
            for child in self.childList:
                if child.nt == prod2.lhs and child.id != prod2.rule_id:
                    flag = False
                    break
                if child.id == prod2.rule_id:
                    for c in child.childList:
                        newChildList.append(c)
                        newNonterminal.append(c.nt)
                else:
                    newChildList.append(child)
                    newNonterminal.append(child.nt)
            if flag:
                self.childList = newChildList
                self.nonterminals = newNonterminal
                self.id = newProd.rule_id

        for child in self.childList:
            child.addIdiom(prod1, prod2, newProd)

    def dfs(self):
        ret = [self.id]
        for child in self.childList:
            ret += child.dfs()
        return ret

    def same(self, node):
        if len(self.nonterminals) != len(node.nonterminals) or len(self.childList) != len(node.childList):
            return False
        if self.nt != node.nt or self.id != node.id:
            return False
        for nt1, nt2 in zip(self.nonterminals, node.nonterminals):
            if nt1 != nt2:
                return False
        for child1, child2 in zip(self.childList, node.childList):
            if not child1.same(child2):
                return False
        return True



class ParseTree:
    def __init__(self, grammar):
        self.grammar = grammar

    def getParseTree(self, actionList, debug=False):
        action = actionList[0]
        rule = self.grammar.get_production_rule_by_id(action)
        assert action == self.grammar.root_rule_id
        rootNode = LogicalFormNode(action, rule.lhs, rule.rhs_nonterminal)
        assert self.buildTree(rootNode, actionList[1:], debug) == []
        return rootNode

    def buildTree(self, node, actionList, debug):
        if debug:
            print(node.id, node.nt, self.grammar.get_production_rule_by_id(node.id))
        for nonterminal in node.nonterminals:
            action = actionList[0]
            rule = self.grammar.get_production_rule_by_id(action)
            if rule.lhs != nonterminal:
                print(rule, rule.lhs, nonterminal)
            assert rule.lhs == nonterminal
            childNode = LogicalFormNode(action, rule.lhs, rule.rhs_nonterminal)
            actionList = self.buildTree(childNode, actionList[1:], debug)
            node.childList.append(childNode)
        return actionList

    def debug(self, node, depth=0):
        print('---' * depth, node.id, node.nt, node.nonterminals, self.grammar.get_production_rule_by_id(node.id))
        for child in node.childList:
            self.debug(child, depth + 1)


class NLNode:
    def __init__(self, nt):
        self.nt = nt
        #self.nonterminals = nonterminals
        self.childList = []
        self.term = ''


def rewrite(s):
    s = re.sub('\\r|\\n', '', s)
    s = re.sub(' +', ' ', s)
    return s[1:-1]


def build_tree(s, depth=0):
    #print('---' * depth, s, sep='')
    for i, x in enumerate(s):
        if x == ' ':
            break
    nt = s[:i]
    node = NLNode(nt)

    s = s[i + 1:]
    if '(' not in s:
        node.term = s
        return node
    num = 0
    child = ''
    for i, x in enumerate(s):
        #print(num, x)
        if num > 0:
            child += x
        if x == '(':
            num += 1
        if x == ')':
            assert num > 0
            num -= 1
            if num == 0:
                child = child[:-1]
                node.childList.append(build_tree(child, depth + 1))
                child = ''
    return node

def rebuldTree(node):
    if node.term == '':
        del node.term
        for child in node.childList:
            rebuldTree(child)
    else:
        assert len(node.childList) == 0
        new_node = NLNode(node.term)
        node.childList.append(new_node)
        del new_node.term

def dfs(node, depth=0):
    print('---' * depth, node.nt, sep='')
    for child in node.childList:
        dfs(child, depth + 1)