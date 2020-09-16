from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor
from sklearn.model_selection import train_test_split
import pandas as pd
import os, re, math, platform

SPLITSEED = 810

# Miscellaneous
_AALetter = ['A', 'C', 'D', 'E', 'F', 'G', 'H',
             'I', 'K', 'L', 'M', 'N', 'P', 'Q',
             'R', 'S', 'T', 'V', 'W', 'Y']
_ftypes = ["AAC", "DiAAC", "PHYCs"]


'''
Read fasta file to dictionary
Input: path name of fasta
Output: dataframe of Peptide Seq {ID1: Seq1, ID2: Seq2,...}
'''


def read_fasta(fname):
    with open(fname, "rU") as f:
        seq_dict = [(record.id, record.seq._data) for record in SeqIO.parse(f, "fasta")]
    seq_df = pd.DataFrame(data=seq_dict, columns=["Id", "Sequence"])
    return seq_df


"""
n_gram statistics
"""


def get_aan_corpus(n=2):
    '''
    Get AA corpus of n_gram (e.g. Di, Tri, etc.)
    Output: AA n_gram corpus ((e.g. Di:400, Tri:3000, etc.))
    '''
    n_corpus = []
    if n <= 2:
        for i in _AALetter:
            for j in _AALetter:
               n_corpus.append("{}{}".format(i, j))
        return n_corpus
    for i in get_aan_corpus(n - 1):
        for j in _AALetter:
            n_corpus.append("{}{}".format(i, j))
    return n_corpus


def get_ngram_counts(seq, n=2):
    '''
    Get n_gram statistics
    Input: peptide sequence and n
    Ouput: n_gram statistic (dictionary) {A.A Corp: Counts}
    '''
    # Get the name of ngram feature
    if n == 2:
        prefix = 'DiC'
    elif n == 3:
        prefix = 'TriC'
    else:
        prefix = '{}gram'.format(n)

    ngrams = [seq[i: i + n] for i in range(len(seq) - n + 1)]
    n_corpus = get_aan_corpus(n)
    ngram_stat = {}
    for aa_ng in n_corpus:
        ngram_stat['{}_{}'.format(prefix, aa_ng)] = ngrams.count(aa_ng) / len(ngrams) * 100
    return ngram_stat


def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen


def minSequenceLengthWithNormalAA(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(re.sub('-', '', i[1])):
            minLen = len(re.sub('-', '', i[1]))
    return minLen


"""
    input.fasta:      the input protein sequence file in fasta format.
    k_space:          the gap of two amino acids, integer, defaule: 5
    output:           the encoding file, default: 'encodings.tsv'
"""


def generateGroupPairs(groupKey):
    gPair = {}
    for key1 in groupKey:
        for key2 in groupKey:
            gPair[key1+'.'+key2] = 0
    return gPair


def cksaagp(fastas, gap = 5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap+2:
        print('Error: all the sequence length should be greater than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
        return 0

    group = {'alphaticr': 'GAVLMI',
             'aromatic': 'FYW',
             'postivecharger': 'KRH',
             'negativecharger': 'DE',
             'uncharger': 'STCPNQ'}

    AA = 'ARNDCQEGHILKMFPSTWYV'

    groupKey = group.keys()

    index = {}
    for key in groupKey:
        for aa in group[key]:
            index[aa] = key

    gPairIndex = []
    for key1 in groupKey:
        for key2 in groupKey:
            gPairIndex.append(key1+'.'+key2)

    encodings = []
    header = ['#']
    for g in range(gap + 1):
        for p in gPairIndex:
            header.append(p+'.gap'+str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        for g in range(gap + 1):
            gPair = generateGroupPairs(groupKey)
            sum = 0
            for p1 in range(len(sequence)):
                p2 = p1 + g + 1
                if p2 < len(sequence) and sequence[p1] in AA and sequence[p2] in AA:
                    gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] = gPair[index[sequence[p1]]+'.'+index[sequence[p2]]] + 1
                    sum = sum + 1

            if sum == 0:
                for gp in gPairIndex:
                    code.append(0)
            else:
                for gp in gPairIndex:
                    code.append(gPair[gp] / sum)

        encodings.append(code)

    return encodings


"""
    input.fasta:      the input protein sequence file in fasta format.
    lambda:           the lambda value, integer, defaule: 30
    output:           the encoding file, default: 'encodings.tsv'
"""

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)


def paac(fastas, lambdaValue=30, w=0.05, **kw):
    if minSequenceLengthWithNormalAA(fastas) < lambdaValue + 1:
        print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    dataFile = re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + r'\data\PAAC.txt' if platform.system() == 'Windows' else re.sub('codes$', '', os.path.split(os.path.realpath(__file__))[0]) + '/data/PAAC.txt'
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/fenmu for j in i])

    encodings = []
    header = ['#']
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)

    for i in fastas:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    return encodings


'''
Insert Iso_electric Point and net_charge(neutral) feature to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., iep, net_charge}
'''


def insert_phycs(seq_df):
    #  Function for compute Isoelectric Point or net_charge of peptide
    def get_ieq_nc(seq, is_iep=True):
        protparam = PA(seq)
        return protparam.isoelectric_point() if is_iep else protparam.charge_at_pH(7.0)

    # Calculating IsoElectricPoints and NeutralCharge
    data_size = seq_df.size
    seq_df['IEP'] = list(map(get_ieq_nc, seq_df['Sequence'], [True] * data_size))  # IsoElectricPoints
    seq_df['Net Charge'] = list(map(get_ieq_nc, seq_df['Sequence'], [False] * data_size))  # Charge(Neutral)

    # Calculating hydrophobic moment (My assume all peptides are alpha-helix)
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'eisenberg')
    descrpt.calculate_moment(window=1000, angle=100, modality='max')
    seq_df['Hydrophobic Moment'] = descrpt.descriptor.reshape(-1)

    # Calculating "Hopp-Woods" hydrophobicity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'hopp-woods')
    descrpt.calculate_global()
    seq_df['Hydrophobicity'] = descrpt.descriptor.reshape(-1)

    # Calculating Energy of Transmembrane Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'tm_tend')
    descrpt.calculate_global()
    seq_df['Transmembrane Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Levitt_alpha_helical Propensity
    descrpt = PeptideDescriptor(seq_df['Sequence'].values, 'levitt_alpha')
    descrpt.calculate_global()
    seq_df['Alpha Helical Propensity'] = descrpt.descriptor.reshape(-1)

    # Calculating Aliphatic Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.aliphatic_index()
    seq_df['Aliphatic Index'] = descrpt.descriptor.reshape(-1)

    # Calculating Boman Index
    descrpt = GlobalDescriptor(seq_df['Sequence'].values)
    descrpt.boman_index()
    seq_df['Boman Index'] = descrpt.descriptor.reshape(-1)

    return seq_df


'''
Insert Amino acid composition to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., AAC_Ax ... AAC_Yx}
'''


def insert_aac(seq_df):
    # Compute AAC for peptide in specific A.A
    def get_aac(seq, aa):
        return seq.count(aa) / len(seq) * 100

    # processing data_frame
    data_size = seq_df.size
    for ll in _AALetter:
        seq_df['AAC_{}'.format(ll)] = list(map(get_aac, seq_df['Sequence'], [ll] * data_size))
    return seq_df


'''
Insert n_grams Descriptor to the sequence data_frame
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., ngram_(1), .., ngram(20 ** n)}
'''


def insert_ngrams(seq_df, n=2):
    data_size = seq_df.size

    ngrams_df = list(map(get_ngram_counts, seq_df['Sequence'], [n] * data_size))
    ngrams_df = pd.DataFrame(ngrams_df)  # Convert ngrams features to pd.DataFrame
    seq_df = pd.concat([seq_df, ngrams_df], axis=1)
    return seq_df


"""
Insert CKSAAGP Descriptor to the sequence data_frame
(Composition of k-spaced amino acid pairs)
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., CKSAAGP(0), ..., CKSAAGP(m)}
"""


def insert_cksaagp(seq_df, gap=2):
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    encoding = cksaagp(fastas, gap=gap)
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df


"""
Insert PAAC Descriptor to the sequence data_frame
(Pseudo Amino Acid Composition)
Input: sequence data_frame {IDx: Seq_x, ...}
Output: data_frame of Peptide Seq {IDx: Seq_x, ..., CKSAAGP(0), ..., CKSAAGP(m)}
"""


def insert_paac(seq_df, lamb=3, w=0.1):
    fastas = [[idx, seq] for idx, seq in zip(seq_df['Id'], seq_df['Sequence'])]
    encoding = paac(fastas, lambdaValue=lamb, w=w)
    encoding = pd.DataFrame(encoding[1:], columns=encoding[0])
    seq_df = pd.concat([seq_df, encoding.iloc[:, 1:]], axis=1)
    return seq_df


if __name__ == "__main__":
    train_dir = "data/train_data"
    test_dir = "data/test_data"
    os.makedirs(train_dir) if not os.path.exists(train_dir) else None
    os.makedirs(test_dir) if not os.path.exists(test_dir) else None
    for prefix in ["Anti-CoV", "Anti-Virus", "non-AMP", "non-AVP"]:
        print("Transform {:s} to features dataframe...".format(prefix), end=" ")
        # Transform the entire set (delete the step for remove corresponding feature)
        df_seq = read_fasta("./data/fasta/{}.faa".format(prefix))
        df_seq = insert_aac(df_seq)
        df_seq = insert_ngrams(df_seq, n=2)
        df_seq = insert_cksaagp(df_seq, gap=2)
        df_seq = insert_paac(df_seq, lamb=4, w=0.4)
        df_seq = insert_phycs(df_seq)  # place PHYCS feature at last for analysis
        df_seq.to_csv("data/{}.csv".format(prefix), index=False)
        # Transform the train/test split
        df_train, df_test = train_test_split(df_seq, random_state=SPLITSEED, test_size=.3)
        df_train.to_csv(os.path.join(train_dir, "{:s}_train.csv".format(prefix)), index=False)
        df_test.to_csv(os.path.join(test_dir, "{:s}_test.csv".format(prefix)), index=False)
        print("Done!")
