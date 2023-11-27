from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import os
import sys
import json
import copy
import collections
import logging
from typing import Optional, Tuple, Any, Dict, Iterable, List
from collections import defaultdict
import numpy as np
import argparse


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../')))
from constants import int_tokenizer as tokenizer
from constants import SPEAKER_START, SPEAKER_END, MENTION_START, MENTION_END, \
    SEP_TOKEN, COPY, SPECIAL_IDS
import utils
import conll


class DocumentState(object):
    def __init__(self, key):
        self.doc_key = key
        self.sentence_end = []
        self.token_end = []
        self.tokens = []
        self.subtokens = []
        self.info = []
        self.segments = []
        self.subtoken_map = []
        self.segment_subtoken_map = []
        self.sentence_map = []
        self.segment_sentence_map = []
        self.pronouns = []
        self.clusters = collections.defaultdict(list)
        self.word_clusters = collections.defaultdict(list)
        self.coref_stacks = collections.defaultdict(list)
        self.mention_to_seg_id = collections.defaultdict(list)
        self.word_mention_to_seg_id = collections.defaultdict(list)
        self.segment_info = []
        self.offsets = []

    def finalize(self):
        all_seg_clusters = get_seg_clusters(self.clusters,
                                            self.mention_to_seg_id,
                                            len(self.segments))
        all_word_seg_clusters = get_seg_clusters(self.word_clusters,
                                                 self.word_mention_to_seg_id,
                                                 len(self.segments))
        # print(len(self.segment_info))
        # print(len(self.segment_info[0]))
        # print(len(self.segment_info[1]))
        # print(all_seg_clusters)
        cluster_indices = get_mention_to_cid(all_seg_clusters)
        docs = []
        num_words = len(utils.flat_lists(self.segments))
        num_segments = len(self.segments)

        subtoken_map = self.segment_subtoken_map
        assert num_words == len(utils.flat_lists(self.segment_subtoken_map))

        sentence_map = self.segment_sentence_map
        assert num_words == len(utils.flat_lists(sentence_map))

        all_mentions = list(self.mention_to_seg_id.keys())
        sentences = self.segments

        # inserting <m> and </m> into target sequences for all mentions
        target_sentences = m_star_target_sequences(
            all_mentions, self.segments,
            MENTION_START, MENTION_END, SEP_TOKEN,
            self.mention_to_seg_id, cluster_indices,
            self.offsets
        )
        target_short_seqs = []
        for target_seq in target_sentences:
            target_short = trim_target_sequence(target_seq,
                                                MENTION_START, MENTION_END)
            target_short_seqs.append(target_short)
        target_maps = get_target_map(target_sentences,
                                     tokenizer.tokenize(MENTION_END)[0],
                                     tokenizer.tokenize(SEP_TOKEN)[0])
        target_tags = get_target_tags(target_sentences, target_maps,
                                      tokenizer.tokenize(MENTION_START)[0],
                                      tokenizer.tokenize(SEP_TOKEN)[0],
                                      tokenizer.tokenize(COPY)[0])
        # add gold clusters info into docs
        for i in range(num_segments):
            docs.append({
                "doc_key": f'{self.doc_key}_{i}',
                "offset": self.offsets[i],
                "sentence": sentences[i],
                "target_sentence": target_sentences[i],
                "target_action": target_tags[i],
                "target_short_sentence": target_short_seqs[i],
                'sentence_map': sentence_map[i],
                "subtoken_map": subtoken_map[i],
                "gold_clusters": self.word_clusters,
                "seg_clusters": all_word_seg_clusters[i],
                "gold_token_clusters": self.clusters
            })

        return docs


def get_seg_clusters(merged_clusters,
                     mention_to_seg_id,
                     num_segs):
    all_seg_clusters = []
    for seg_id in range(num_segs):
        seg_clusters = []
        for c in merged_clusters:
            seg_cluster = []
            for m in c:
                m_sids = mention_to_seg_id[tuple(m)]
                if seg_id in m_sids:
                    seg_cluster.append(m)
            if len(seg_cluster) >= 1:
                seg_clusters.append(seg_cluster)
        all_seg_clusters.append(seg_clusters)
    return all_seg_clusters


def get_mention_to_cid(all_seg_clusters):
    # k: old group idx  v: sorted group idx

    def get_seg_mention_to_gid(groups):
        mention_to_gid = {}
        first_mentions = [min(g, key=lambda m: (m[1], -m[0])) for g in groups]
        assert len(first_mentions) == len(groups)
        sorted_ids = sorted(list(range(len(first_mentions))),
                            key=lambda k: (first_mentions[k][1],
                                           -first_mentions[k][0]))
        gid_map = {j: i for i, j in enumerate(sorted_ids)}
        for i, g in enumerate(groups):
            gid = gid_map[i]
            for m in g:
                mention_to_gid[tuple(m)] = gid
        return mention_to_gid

    all_seg_ment2cid = []
    for seg_clusters in all_seg_clusters:
        ment2cid = get_seg_mention_to_gid(seg_clusters)
        all_seg_ment2cid.append(ment2cid)
    return all_seg_ment2cid


def m_star_target_sequences(
        mentions: List[Tuple[int, int]],
        sequences: List[List[str]],
        m_special_start: str,
        m_special_end: str,
        m_sep: str,
        mention_to_seg_id: Dict[tuple, list],
        cluster_indices: List[Dict],
        offsets: List
):
    """
        Get a sequence of target sentences with <m> and <\m> inserted.
        mentions: list of mentions, e.g. [(0, 0), (2, 3), (4, 4)] format: [start, end] (inclusive)
        sequences: list of sequences, e.g. [['I', 'have', 'a', 'cat'], ['I', 'have', 'a', 'dog']]
        m_special_start: special token for starting bracket
        m_special_end: special token for ending bracket
        mention_to_seg_id: dict, mapping mention to its segment id
    """
    m_startings, m_endings = zip(*mentions) if len(mentions) > 0 else ([], [])
    all_m_cids = []
    all_m_sids = []
    for m in mentions:
        m_sids = mention_to_seg_id[tuple(m)]
        m_seg_cids = [cluster_indices[m_sid] for m_sid in m_sids]
        m_cids = [m_seg_cid[tuple(m)] for m_seg_cid in m_seg_cids]
        all_m_cids.append(m_cids)
        all_m_sids.append(m_sids)
    # later segment comes first
    end_pos = [(m_sid, x + 1, -1,
                -m_startings[i],
                m_cid) for i, x in enumerate(
        m_endings) for m_sid, m_cid in zip(all_m_sids[i], all_m_cids[i])]
    start_pos = [(m_sid, x, 1, -m_endings[i],
                  m_cid) for i, x in enumerate(
        m_startings) for m_sid, m_cid in zip(all_m_sids[i], all_m_cids[i])]
    # insert from right to left, so that the calculated positions are not changed
    sorted_pos = sorted(end_pos + start_pos, reverse=True)
    target_sequences = copy.deepcopy(sequences)
    # offset of each segment
    # prev_loc, prev_token, prev_seg_idx = -1, None, -1
    for x in sorted_pos:
        seg_idx = x[0]
        offset = offsets[seg_idx]
        if x[2] > 0:
            # start
            assert x[2] == 1
            target_sequences[seg_idx].insert(x[1] - offset, m_special_start)
        else:
            # end
            end_inserts = tokenizer.tokenize(
                m_sep) + tokenizer.tokenize(str(x[-1])) + [m_special_end]
            for e in reversed(end_inserts):
                target_sequences[seg_idx].insert(x[1] - offset, e)
    return target_sequences


def trim_target_sequence(target_seq,
                         m_special_start,
                         m_special_end):
    out_seq = []
    ment_stack = []
    for idx, s in enumerate(target_seq):
        if s == m_special_start:
            out_seq.append(s)
            ment_stack.append(idx)
        elif len(ment_stack) > 0:
            out_seq.append(s)
            if s == m_special_end:
                ment_stack.pop()
    out_seq.append('</s>')
    return out_seq


def get_target_tags(target_sequences,
                    target_maps,
                    m_special_start,
                    m_sep, m_copy):
    # 1 for inside entity 0 for outside entity
    target_tags = []
    for target_sequence, target_map in zip(target_sequences, target_maps):
        target_tag = np.array(target_sequence)
        tgt_map = np.array(target_map, dtype=bool)
        tag_map = (target_tag != m_special_start) & (
                target_tag != m_sep) & (
                      ~tgt_map)
        target_tag[tag_map] = m_copy
        target_tags.append(target_tag.tolist())
    return target_tags


def get_target_map(target_sequences, m_special_end, m_sep):
    # 1 for inside entity 0 for outside entity
    target_maps = []
    for target_sequence in target_sequences:
        target_map = []
        status = 'o'
        for t in target_sequence:
            if status == 'o':
                target_map.append(0)
            else:
                target_map.append(1)
            if t == m_sep:
                status = 'i'
            elif t == m_special_end:
                status = 'o'
        assert len(target_map) == len(target_sequence)
        target_maps.append(target_map)
    return target_maps


def normalize_word(word, language):
    br_dict = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]"}

    if language == "arabic":
        word = word[:word.find("#")]

    if word in br_dict:
        word = br_dict[word]
        return word
    elif word == "/." or word == "/?":
        return word[1:]
    elif word == "''" or word == "``":  # <unk> otherwise
        return "\""
    elif word == "`":  # <unk> otherwise
        return "\'"
    else:
        return word


# first try to satisfy constraints1, and if not possible, constraints2.
def split_into_segments(
        document_state,
        max_segment_len,
        stride,
        constraints1,
        constraints2,
        is_train
):
    # introduce stride
    # get offset info
    current = 0
    offsets = []
    if not is_train and len(document_state.subtokens) < max_segment_len:
        stride = len(document_state.subtokens)
    all_mentions = utils.flat_lists(document_state.clusters)
    seg_idx = 0
    while current < len(document_state.subtokens):
        offsets.append(current)
        end = min(current + max_segment_len - 1 - 1,
                  len(document_state.subtokens) - 1)

        while end >= current and not constraints1[end]:
            end -= 1

        if end < current:
            end = min(current + max_segment_len - 1 - 1,
                      len(document_state.subtokens) - 1)
            while end >= current and not constraints2[end]:
                end -= 1
            if end < current:
                raise Exception("Can't find valid segment")

        document_state.segments.append(
            document_state.subtokens[current:end + 1] + [
                '</s>'])
        for m in all_mentions:
            if current <= m[0] <= m[1] <= end:
                document_state.mention_to_seg_id[tuple(m)].append(seg_idx)
                document_state.word_mention_to_seg_id[
                    (document_state.subtoken_map[m[0]],
                     document_state.subtoken_map[m[1]])].append(seg_idx)
        sent_map = document_state.sentence_map[current:end + 1]
        document_state.segment_sentence_map.append(sent_map + [sent_map[-1]])
        subtoken_map = document_state.subtoken_map[current:end + 1]
        document_state.segment_subtoken_map.append(
            subtoken_map + [subtoken_map[-1]])
        seg_idx += 1
        # current = end + 1
        next_cur = min(current + stride, len(document_state.subtokens))
        while next_cur > current and not constraints1[next_cur - 1]:
            next_cur -= 1
        if next_cur < current + 1:
            next_cur = min(current + stride, len(document_state.subtokens))
            while next_cur > current and not constraints2[next_cur - 1]:
                next_cur -= 1
            if next_cur < current + 1:
                raise Exception("Can't find valid stride")
        current = next_cur
    document_state.offsets = offsets
    return


def get_doc_sentence_map(sentence_end):
    current = 0
    sent_map = []
    for i, s in enumerate(sentence_end):
        sent_map.append(current)
        current += int(s)
    return sent_map


def get_document(
        instance, tokenizer, segment_len, stride, is_train
):
    document_state = DocumentState(instance["id"])
    doc_word_idx = -1
    sent_word_map = {}
    for sent_idx, sentence in enumerate(instance["sentences"]):
        sent_word_map[sent_idx] = {}
        for word_idx, word in enumerate(sentence):
            doc_word_idx += 1
            sent_word_map[sent_idx][word_idx] = [len(document_state.subtokens)]
            word = normalize_word(word, "english")
            if is_punctuation(word):
                subtokens = tokenizer.tokenize(word)[1:]  # skipping '_'
            else:
                subtokens = tokenizer.tokenize(word)
            document_state.tokens.append(word)
            if len(subtokens) > 0:
                document_state.token_end += [False] * (len(subtokens) - 1) + [
                    True]
            for sidx, subtoken in enumerate(subtokens):
                document_state.subtokens.append(subtoken)
                document_state.sentence_end.append(False)
                document_state.subtoken_map.append(doc_word_idx)
            sent_word_map[sent_idx][word_idx].append(len(
                document_state.subtokens))
        if len(document_state.sentence_end) > 0:
            document_state.sentence_end[-1] = True
    constraints1 = document_state.sentence_end

    assert len(document_state.sentence_end) == len(document_state.token_end)
    document_state.sentence_map = get_doc_sentence_map(
        document_state.sentence_end)
    mapped_clusters = []
    word_clusters = []
    for cluster in instance["mention_clusters"]:
        cur_cluster = []
        word_cluster = []
        for sent_idx, word_start, word_end in cluster:
            span_start = sent_word_map[sent_idx][word_start][0]
            span_end = sent_word_map[sent_idx][word_end - 1][1] - 1
            cur_cluster.append((span_start, span_end))
            word_span_start = document_state.subtoken_map[span_start]
            word_span_end = document_state.subtoken_map[span_end]
            word_cluster.append((word_span_start, word_span_end))
        mapped_clusters.append(sorted(cur_cluster, key=lambda x: x[0]))
        word_clusters.append(sorted(word_cluster, key=lambda x: x[0]))
    document_state.clusters = mapped_clusters
    document_state.word_clusters = word_clusters
    split_into_segments(
        document_state, segment_len, stride, constraints1,
        document_state.token_end, is_train
    )

    stats[f"max_seg_len"] = max(
        stats["max_seg_len"], max([len(s) for s in document_state.segments])
    )
    stats[f"max_num_seg"] = max(
        len(document_state.segments), stats[f"max_num_seg"]
    )
    document = document_state.finalize()
    return document


def is_punctuation(c):
    if (
            c in {".", ",", "?", "!", ";",
                  ":", "'s", "'m", "'ve", "n't", "'ll",
                  ")", "]", "}", "-"}
    ):
        return True
    return False


def is_special(c):
    if (
            c in {"<pad>", "</s>", "<unk>"}
    ):
        return True
    return False


def accumu(lis):
    total = 0
    for x in lis:
        yield total
        total += x


def minimize_partition(
        split, stats, tokenizer, seg_len,
        stride,
        input_dir,
        output_dir, is_train
):
    input_path = os.path.join(input_dir, f"{split}.jsonl")
    output_path = os.path.join(output_dir, f"{split}.t5-small.english."
                                           f"{seg_len}.jsonlines")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    count = 0
    logger.info("Minimizing {}".format(input_path))
    datasets, max_target_len = [], 0
    max_input_len = 0
    max_num_clusters = 0
    max_seg_clusters = 0
    with open(input_path, "r") as input_file:
        for line in input_file.readlines():
            instance = json.loads(line)
            document = get_document(instance, tokenizer, seg_len, stride,
                                    is_train)
            for doc in document:
                max_input_len = max(
                    [max_input_len] + [len(doc['sentence'])])
                max_target_len = max(
                    [max_target_len] + [len(doc['target_sentence'])])
                max_num_clusters = max(
                    [max_num_clusters] + [len(doc['gold_clusters'])])
                max_seg_clusters = max(
                    [max_num_clusters] + [len(doc['seg_clusters'])])
                datasets.append(doc)
                count += 1

    with open(output_path, 'w') as f:
        for d in datasets:
            f.write('%s\n' % json.dumps(d))
    # json.dump(datasets, open(output_path, "w"))
    # TODO: add max num clusters stats
    logger.info(
        f"Maximum input sequence length: {max_input_len}, Maximum target sequence length: {max_target_len}")
    logger.info(f'Maximum num gold clusters: {max_num_clusters}')
    logger.info(f'Maximum num segment clusters: {max_seg_clusters}')
    logger.info("Wrote {} documents to {}".format(count, output_path))


def minimize_split(stats, seg_len, stride, input_dir, output_dir):
    minimize_partition("dev", stats, tokenizer, seg_len, stride, input_dir,
                       output_dir, False)
    minimize_partition("test", stats, tokenizer, seg_len, stride, input_dir,
                       output_dir, False)
    minimize_partition("train", stats, tokenizer, seg_len, stride, input_dir,
                       output_dir, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='input directory')
    parser.add_argument('--output_dir', type=str, help='output directory')
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for seg_len in [2048, 1792, 1536, 1152]:
        stats = collections.defaultdict(int)
        stride = seg_len // 2
        minimize_split(stats, seg_len, stride, input_dir, output_dir)

        logger.info("Dataset stats:")
        for k, v in stats.items():
            logger.info("{} = {}".format(k, v))
