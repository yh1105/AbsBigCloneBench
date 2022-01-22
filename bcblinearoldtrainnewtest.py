import os
import tensorflow as tf
import numpy as np
import networkforclonenew as network
from javalang.ast import Node
from javalang.tree import VariableDeclarator
from javalang.tree import MemberReference
from javalang.tree import MethodInvocation
import javalang
from parameters import EPOCHS, LEARN_RATE
from sklearn.metrics import precision_score, recall_score, f1_score
listVariable=[]
def dfsSearch_withid(children):
    if not isinstance(children, (str, Node, list, tuple)):
        return
    if isinstance(children, (str, Node)):
        if str(children) == '':
            return
        if str(children).startswith('"'):
            return
        if str(children).startswith("'"):
            return
        if str(children).startswith("/*"):
            return
        # ss = str(children)
        global num_nodes
        num_nodes += 1
        listt1.append(children)
        return
    for child in children:
        if isinstance(child, (str, Node, list, tuple)):
            dfsSearch_withid(child)


def _traverse_treewithid(root):
    listnode=[]
    global num_nodes
    num_nodes = 1
    queue = [root]
    root_json = {
        "node": _name(root),

        "children": []
    }
    queue_json = [root_json]
    while queue:
        current_node = queue.pop(0)
        current_node_json = queue_json.pop(0)
        global listt1
        listt1 = []
        dfsSearch_withid(current_node.children)
        children = listt1
        for child in children:
            listnode.append(str(child))
            child_json = {
                "node": str(child),
                "children": []
            }
            current_node_json['children'].append(child_json)
            if isinstance(child, (Node)):
                queue_json.append(child_json)
                queue.append(child)
    return root_json, num_nodes,listnode
def _name(node):
    return type(node).__name__

def getWordEmd(word):
    listrechar = np.array([0.0 for i in range(0, len(listchar))])
    tt = 1
    for lchar in word:
        listrechar += np.array(((len(word) - tt + 1) * 1.0 / len(word)) * np.array(dicttChar[lchar]))
        tt += 1
    return listrechar
def getNodeMatix(ll, dicttt, em):
    listnode1=[]
    listnode2=[]
    for n1 in dicttt[ll[0]]:
        listnode1.append(em[n1])
    for n2 in dicttt[ll[1]]:
        listnode2.append(em[n2])
    return [listnode1],[listnode2]
def train_model(infile, embeddings):
    # faa = open("D:\\bigclonebenchdata\\bigclonebenchdata\\74.txt", 'r', encoding="utf-8")
    # fff = faa.read()
    # tree = javalang.parse.parse_member_signature(fff)
    # sample, size = _traverse_treewithid(tree)
    # print(listVariable)
    # # print(sample)
    # s1, ss1 = _traverse_treewithid2(tree)
    # print(s1)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_feats = len(getWordEmd('ForStatement'))
    nodes_node1, nodes_node2, res = network.init_net_nofinetune_one_nohidden(num_feats,100)
    labels_node, loss_node = network.loss_layer(res)
    optimizer = tf.train.GradientDescentOptimizer(LEARN_RATE)
    train_step = optimizer.minimize(loss_node)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)  # config=tf.ConfigProto(device_count={'GPU':0}))
    sess.run(tf.global_variables_initializer())
    dictt = {}
    listrec = []
    f = open("flist8.txt", 'r')
    line = f.readline().rstrip('\t')
    l = line.split('\t')
    z = 0
    for ll in l:
        if not os.path.exists(ll):
            listrec.append(ll)
            continue
        faa = open(ll, 'r', encoding="utf-8")
        fff = faa.read()
        tree = javalang.parse.parse_member_signature(fff)
        sample, size,listnn = _traverse_treewithid(tree)
        #listnn=list(set(listnn))
        # print(ll)
        # print(listnn)
        if size > 3000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = listnn
    f.close()
    f = open("flistOld.txt", 'r')
    line = f.readline().rstrip('\t')
    l = line.split('\t')
    z = 0
    for ll in l:
        if not os.path.exists(ll):
            listrec.append(ll)
            continue
        faa = open(ll, 'r', encoding="utf-8")
        fff = faa.read()
        tree = javalang.parse.parse_member_signature(fff)
        sample, size, listnn = _traverse_treewithid(tree)
        #listnn = list(set(listnn))
        # print(ll)
        # print(listnn)
        if size > 3000 or size < 10:
            z += 1
            listrec.append(ll)
            continue
        dictt[ll] = listnn
    f.close()
    for epoch in range(1, EPOCHS + 1):
        f = open(infile, 'r')
        line = "123"
        k = 0
        while line:
            line = f.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            k += 1
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, nodes2=getNodeMatix(l,dictt,embeddings)
            batch_labels.append(l[2])
            _, err, r = sess.run(
                [train_step, loss_node, res],
                feed_dict={
                    nodes_node1: nodes1,
                    nodes_node2: nodes2,
                    labels_node: batch_labels
                }
            )
            maxnodes = max(len(nodes1[0]), len(nodes2[0]))
            if k % 1000 == 0:
                print('Epoch:', epoch,
                      'Step:', k,
                      'Loss:', err,
                      'R:', r,
                      'Max nodes:', maxnodes
                      )
            #if k>200000:
                #break
        f.close()
        correct_labels_dev = []
        predictions_dev = []
        for reci in range(0, 18):
            predictions_dev.append([])
        ff = open("recordtestdata031338.txt", 'r')
        fa=open("recordoneOldTrainNewTest"+str(epoch)+".txt",'w')
        line = "123"
        k = 0
        while line:
            line = ff.readline().rstrip('\n')
            l = line.split('\t')
            if len(l) != 3:
                break
            if (l[0] in listrec) or (l[1] in listrec):
                continue
            batch_labels=[]
            nodes1, nodes2 = getNodeMatix(l, dictt, embeddings)
            batch_labels.append(l[2])
            k += 1
            output = sess.run([res],
                              feed_dict={
                                  nodes_node1: nodes1,
                                  nodes_node2: nodes2,
                              }
                              )
            correct_labels_dev.append(int(batch_labels[0]))
            threaholder = -0.9
            fa.write(str(output[0]))
            fa.write(" ")
            for i in range(0, 18):
                if output[0] >= threaholder:
                    predictions_dev[i].append(1)
                else:
                    predictions_dev[i].append(-1)
                threaholder += 0.1
        for i in range(0, 18):
            print("testdata\n")
            print("threa:")
            print(i)
            p = precision_score(correct_labels_dev, predictions_dev[i], average='binary')
            r = recall_score(correct_labels_dev, predictions_dev[i], average='binary')
            f1score = f1_score(correct_labels_dev, predictions_dev[i], average='binary')
            print("recall_test:" + str(r))
            print("precision_test:" + str(p))
            print("f1score_test:" + str(f1score))
        ff.close()
        fa.close()


def dfsDict(root):
    global listtfinal
    listtfinal.append(str(root['node']))
    global numnodes
    numnodes+=1
    if len(root['children']):
        pass
    else:
        return
    for dictt in root['children']:
        dfsDict(dictt)
if __name__ == '__main__':
    f = open("sentenceBCB8.txt", 'r')
    line = "123"
    listword = []
    while line:
        line = f.readline().rstrip("\n")
        listt = line.split(" ")
        listword.extend(listt)
        listword = list(set(listword))
    f.close()
    f = open("sentenceBCBOld.txt", 'r')
    line = "123"
    #listword = []
    while line:
        line = f.readline().rstrip("\n")
        listt = line.split(" ")
        listword.extend(listt)
        listword = list(set(listword))
    f.close()
    listword.append("localVariables")
    #print(len(listword))
    dicttChar = {}
    def _onehot(i, total):
        return [1.0 if j == i else 0.0 for j in range(total)]
    listchar = ['7', 'I', 'E', 'D', 'u', 'C', 'Y', 'W', 'y', '|', '9', '^', 'X', 't', 'a', 'o', 'Z', 'b', 'A', 'J', 'R',
                'w', '?', 'g', '3', '$', 'B', 'l', '5', 'z', 'v', 'T', '2', 'd', '<', 'e', 'M', 'c', 'S', 'm', '4', 'K',
                'O', 'f', 'i', '=', 'Q', '+', 'x', 'N', '1', 'r', 'p', 'G', 'k', '*', 'q', 'L', 'P', '.', 'n', 'j', 'V',
                'U', '6', '/', '%', '8', 'F', 's', '!', '-', '&', '>', 'h', 'H', '0', '_']
    for i in range(0, len(listchar)):
        dicttChar[listchar[i]] = _onehot(i, len(listchar))
    dictfinalem = {}
    t = 0
    for l in listword:
        t += 1
        dictfinalem[l] = getWordEmd(l)
    train_model('recordtraindata30w.txt', dictfinalem)