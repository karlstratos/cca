import argparse
import numpy
from collections import Counter

def main(args):
    trainpairs, trainwords = read_sentpairs(args.train_sents)
    testpairs, testwords = read_sentpairs(args.test_sents)
    rep = get_rep(args.embedding_file, {word:True for word in trainwords.keys() + testwords.keys()})
    assert('<?>' in rep)
    final_acc = 0
    for _ in range(args.num_draws):
        randpairs = [ trainpairs[i] for i in numpy.random.choice(len(trainpairs), args.num_sents) ]
        acc = nearest_tagging(testpairs, randpairs, rep)
        print acc
        final_acc += acc / args.num_draws
    print 'avg acc:', final_acc
    return final_acc
        
def nearest_tagging(testpairs, trainpairs, rep):
    protos = get_protos(trainpairs)
    num_instances = 0.
    num_correct = 0.
    cache = {(proto, protos[proto]):True for proto in protos}
    for pair in testpairs:
        for j in range(len(pair[0])):
            num_instances += 1
            word = pair[0][j]
            gold_label = pair[1][j]
            nearest_proto, predicted_label = grab_nearest(word, protos, rep, cache)
            #print word, gold_label, predicted_label, '('+nearest_proto+')'
            if predicted_label == gold_label:
                num_correct += 1
    acc = num_correct / num_instances
    return acc

def grab_nearest(word, protos, rep, cache):
    if word in cache:
        return cache[word]
    x = rep[word] if word in rep else rep['<?>']
    min_dist = float('inf')
    closest_proto = False
    for proto in protos:
        if not proto in rep:
            continue
        xp = rep[proto]
        dist = numpy.linalg.norm(x - xp)
        if dist < min_dist:
            min_dist = dist
            closest_proto = proto
    cache[word] = closest_proto, protos[closest_proto]
    return cache[word]

def get_protos(trainpairs):
    label_count = {}
    word_count = {}
    for pair in trainpairs:
        for j in range(len(pair[0])):
            word = pair[0][j]
            label = pair[1][j]
            if not word in word_count:
                word_count[word] = Counter()
            if not label in label_count:
                label_count[label] = Counter()
            word_count[word][label] += 1
            label_count[label][word] += 1
    protos = {}
    for word in word_count:
        sorted_list = sorted(word_count[word].items(), key=lambda x: x[1], reverse=True)
        winner_label = sorted_list[0][0]
        protos[word] = winner_label
    return protos

def get_rep(embedding_file, words):
    rep = {}
    with open(embedding_file) as f:
        lines = f.readlines()
        for line in lines:    
            toks = line.split()
            if toks[1] in words or toks[1] == '<?>':
                rep[toks[1]] = numpy.array(map(lambda x: float(x), toks[2:]))
    return rep

def read_sentpairs(tagged_sents):
    words = {}
    sentpairs = []
    with open(tagged_sents) as f:
        lines = f.readlines()
        sentpair = [[], []]
        for line in lines:
            toks = line.split()
            if len(toks) == 0 and len(sentpair[0]) != 0:
                sentpairs.append((tuple(sentpair[0]), tuple(sentpair[1])))
                sentpair = [[], []]
            else:
                words[toks[0]] = True
                sentpair[0].append(toks[0])
                sentpair[1].append(toks[1])
        if len(toks) == 0 and len(sentpair[0]) != 0:
            sentpairs.append(sentpair)
    return sentpairs, words

if __name__=='__main__':
    argparser = argparse.ArgumentParser('Nearest-neighbor tagging')
    argparser.add_argument('train_sents', type=str, help='file of tagged sentences for training')
    argparser.add_argument('test_sents', type=str, help='file of tagged sentences for testing')
    argparser.add_argument('--embedding_file', type=str, help='file of embeddings')
    argparser.add_argument('--num_sents', type=int, default=10, help='how many sentences to draw')
    argparser.add_argument('--num_draws', type=int, default=10, help='how many times to repeat the experiment')
    argparser.add_argument('--quiet', action='store_true', help='quiet mode')
    args = argparser.parse_args()

    acc = main(args)
    
