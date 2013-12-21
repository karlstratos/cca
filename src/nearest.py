import argparse
import numpy
from io import say
from io import set_quiet
from collections import Counter

def main(args):
    if args.analyze:
        analyze_output(args.analyze)
    trainpairs, trainwords = read_sentpairs(args.train_sents)
    testpairs, testwords = read_sentpairs(args.test_sents)
    allwords = {word:True for word in trainwords.keys() + testwords.keys()}            
    print '{} training sentences, {} test sentences, {} words'.format(len(trainpairs), len(testpairs), len(allwords))
    print 'getting rep from {}'.format(args.embedding_file)
    rep = get_rep(args.embedding_file, allwords, args.top)
    assert('<?>' in rep)
    print 'has {} embeddings, {} dimensional'.format(len(rep), len(rep[rep.keys()[0]]))
    if args.longest:
        print 'drawing {} longest training sentences'.format(args.num_sents) 
        longestpairs = sorted(trainpairs, key=lambda x: len(x[0]), reverse=True)[:args.num_sents]
        acc = nearest_tagging(testpairs, longestpairs, rep, args.output)
        print 'acc:', acc
        return acc
    avg_acc = 0
    for drawnum in range(args.num_draws):
        print '{}. drawing {} random training sentences'.format(drawnum+1, args.num_sents), 
        randpairs = [ trainpairs[i] for i in numpy.random.choice(len(trainpairs), args.num_sents) ]
        acc = nearest_tagging(testpairs, randpairs, rep, args.output)
        print acc
        avg_acc += acc / args.num_draws
    print 'avg acc:', avg_acc
    return avg_acc
        
def nearest_tagging(testpairs, trainpairs, rep, output):
    protos = get_protos(trainpairs)
    if output:
        print '\n\n{} protos'.format(len(protos))
        for proto in protos:
            print proto, '\t', protos[proto]
        outf = open(output, 'wb')
    num_instances = 0.
    num_correct = 0.
    cache = {}
    proto_cache = {} 
    for pair in testpairs:
        for j in range(len(pair[0])):
            num_instances += 1
            word = pair[0][j]
            gold_label = pair[1][j]
            nearest_proto, predicted_label = grab_nearest(word, protos, rep, cache, proto_cache)
            if output:
                say('{} {} {} {}'.format(word, gold_label, predicted_label, '('+nearest_proto+')'))
                print >> outf, word, gold_label, predicted_label, '('+nearest_proto+')'
            if predicted_label == gold_label:
                num_correct += 1
        if output:
            say('')
            print >> outf

    if output:
        outf.close()            
    acc = num_correct / num_instances
    return acc

def grab_nearest(word, protos, rep, cache, proto_cache):
    if word in cache:
        return cache[word]
    x = rep[word] if word in rep else rep['<?>']

    min_dist = float('inf')
    closest_proto = False
    for proto in protos:
        if not proto in rep:
            continue
        if proto in proto_cache:
            xp = proto_cache[proto]
        else:
            xp = rep[proto]
            proto_cache[proto] = xp
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

def get_rep(embedding_file, words, top):
    rep = {}
    with open(embedding_file) as f:
        lines = f.readlines()
        for line in lines:    
            toks = line.split()
            end_ind = len(toks) if not top else top + 2
            if toks[1] in words or toks[1] == '<?>':
                rep[toks[1]] = numpy.array(map(lambda x: float(x), toks[2:end_ind]))
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
        if len(sentpair[0]) != 0:
            sentpairs.append(sentpair)
    return sentpairs, words

def analyze_output(output):
    wrong_count = Counter()
    num_words = 0
    sent = []
    sents = []
    with open(output) as f:
        for line in f:
            toks = line.split()
            if len(toks) == 0:
                sents.append(sent)
                sent = []
                continue
            num_words += 1
            word, gold, pred, closest_proto = toks
            sent.append((word, gold, pred, closest_proto))
            ordered_pair = [pred, gold] if pred < gold else [gold, pred]
            if pred != gold:
                wrong_count[tuple(ordered_pair)] += 1
                
    for sent in sents:
        for j, (word, gold, pred, closest_proto) in enumerate(sent):
            if pred != gold:
                if j > 0:
                    print sent[j-1][0], sent[j-1][1], sent[j-1][2], sent[j-1][3] 
                print word, gold, pred, closest_proto, '<--------'
                if j < len(sent) - 1:
                    print sent[j+1][0], sent[j+1][1], sent[j+1][2], sent[j+1][3]
                print  
             
    sorted_wrong = sorted(wrong_count.items(), key=lambda x: x[1], reverse=True)
    print 'top 10 most confused categories are'
    for pair, count in sorted_wrong[:10]:
        print '\t', pair[0], pair[1], count
        
    print 'acc: {}'.format(float(num_words - sum(wrong_count[pair] for pair in wrong_count)) / num_words * 100)
    exit()

if __name__=='__main__':
    argparser = argparse.ArgumentParser('Nearest-neighbor tagging')
    argparser.add_argument('--train_sents', type=str, help='file of tagged sentences for training')
    argparser.add_argument('--test_sents', type=str, help='file of tagged sentences for testing')
    argparser.add_argument('--embedding_file', type=str, help='file of embeddings')
    argparser.add_argument('--longest', action='store_true', help='use the longest sentences')
    argparser.add_argument('--num_sents', type=int, default=10, help='how many sentences to draw')
    argparser.add_argument('--num_draws', type=int, default=10, help='how many times to repeat the experiment')
    argparser.add_argument('--top', type=int, help='use only this many top dimensions')
    argparser.add_argument('--quiet', action='store_true', help='quiet mode')
    argparser.add_argument('--output', type=str, help='output predictions in this file')
    argparser.add_argument('--analyze', type=str, help='analyze this output file')
    args = argparser.parse_args()

    set_quiet(args.quiet)
    acc = main(args)
    
