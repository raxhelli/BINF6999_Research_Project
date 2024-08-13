import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from complexcgr import FCGR
import faulthandler

faulthandler.enable()


def filter_short_sequences_and_labels(sequences, labels, min_length=500,
                                      max_N_percentage=0.2):
    filtered_sequences = []
    filtered_labels = []
    y = np.asarray(labels)

    for sequence in sequences:
        N_count = sequence.count('N')
        N_percentage = N_count / len(sequence)
        if len(sequence) >= min_length and N_percentage <= max_N_percentage:
            filtered_sequences.append(sequence)
            filtered_labels.append(True)
        else:
            filtered_labels.append(False)

    y_filtered = y[filtered_labels]
    return filtered_sequences, y_filtered


def generate_reads(sequence, max_read_length=3000):
    sequence_length = len(sequence)
    reads = []
    rev_reads = []

    if sequence_length <= max_read_length:
        reads.append(sequence)
        rev_sequence = Seq(sequence).reverse_complement()
        rev_reads.append(str(rev_sequence))
    else:
        for i in range(0, sequence_length, max_read_length):
            tmp = sequence[i:i + max_read_length]
            reads.append(tmp)
            rev_sequence = Seq(tmp).reverse_complement()
            rev_reads.append(str(rev_sequence))

    return reads, rev_reads


def replaced_bases(seq: str):
    seq_final = seq.upper()
    replacement = list(seq_final)

    for i, base in enumerate(seq_final):
        if base not in {"A", "C", "T", "G"}:
            replacement[i] = "N"

    return "".join(replacement)


def return_reads(seqs, labels, k=6):
    y = []
    mod_seqs = [replaced_bases(seq) for seq in seqs]
    reads = []
    rev_seqs = []

    for seq in mod_seqs:
        read_list, reverse_reads = generate_reads(seq)
        reads.append(read_list)
        rev_seqs.append(reverse_reads)

    for i, entry in enumerate(reads):
        tmp = [labels[i] for _ in range(len(entry))]
        y.extend(tmp)

    fcgr = FCGR(k=k)
    X_forward = []
    X_reverse = []
    counter = 0

    for i, sub_seqs in enumerate(reads):
        for sub_seq in sub_seqs:
            X_forward.append(fcgr(sub_seq))

        counter = i + 1
        if counter % 100 == 0:
            print("%d / %d complete" % (counter, len(reads)))

    counter = 0
    for i, sub_seqs in enumerate(rev_seqs):
        for sub_seq in sub_seqs:
            X_reverse.append(fcgr(sub_seq))

        counter = i + 1
        if counter % 100 == 0:
            print("%d / %d complete" % (counter, len(reads)))

    return X_forward, X_reverse, np.asarray(y)


if __name__ == "__main__":
    K = 6
    fasta_files = ["./virus.fasta", "./eukaryote.fasta",
                   "./archaea.fasta", "./bacteria.fasta"]
    print("files loaded")
    all_sequences = []
    labels = []

    for index, fasta_file in enumerate(fasta_files):
        print(index)
        print(fasta_file)
        for record in SeqIO.parse(fasta_file, "fasta"):
            all_sequences.append(str(record.seq))
            print(len(all_sequences))

            label = 1 if index == 0 else 0
            labels.append(label)
            print(len(labels))

    print(labels)
    print("labels created")

    filtered_sequences, filtered_labels = (
        filter_short_sequences_and_labels(all_sequences, labels))
    print("filter completed")

    X_forward, X_reverse, y = return_reads(filtered_sequences,
                                           filtered_labels,
                                           k=K)
    print("FCGR created")

    np.save("X_train_forward.npy", np.asarray(X_forward))
    np.save("X_train_reverse.npy", np.asarray(X_reverse))
    np.save("y_train.npy", np.asarray(y))
