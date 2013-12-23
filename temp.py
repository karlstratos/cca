import cPickle

with open('input/egw20m/egw20m.3grams.cutoff5.univ12.featurized.pickled') as f:
    sX1, sY1, iX1, iY1, massXY1, sqmassX1, sqmassY1, M1 = cPickle.load(f)

with open('input/egw20m/egw20m.cutoff5.univ12.window3.pickled') as f:
    sX2, sY2, iX2, iY2, massXY2, sqmassX2, sqmassY2, M2 = cPickle.load(f)

assert(len(sX1) == len(sX2))
assert(len(sY1) == len(sY2))
assert(M1 == M2)

print 1

for i in range(len(iX1)):
    xtok1 = iX1[i]
    assert(sqmassX1[sX1[xtok1]] == sqmassX2[sX2[xtok1]])

print 2

for i in range(len(iY1)):
    ytok1 = iY1[i]
    try:
        assert(sqmassY1[sY1[ytok1]] == sqmassY2[sY2[ytok1]])
    except:
        print ytok1
        print sqmassY1[sY1[ytok1]], sqmassY2[sY2[ytok1]]
        exit()

print 3

for i in range(len(iX1)):
    xtok1 = iX1[i]
    for j in range(len(iY1)):
        ytok1 = iY1[j]
        try:
            assert(massXY1[sX1[xtok1], sY1[ytok1]] ==  massXY2[sX2[xtok1], sY2[ytok1]])
        except:
            print xtok1, ytok1
            print massXY1[sX1[xtok1], sY1[ytok1]],  massXY2[sX2[xtok1], sY2[ytok1]]


print 4
