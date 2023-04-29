"""
Evaluate different model for symbolic translation.
"""
import os
from pathlib import Path
import argparse
import logging
from collections import defaultdict
import numpy as np
import spot
from pprint import pprint
from dataset_symbolic import load_split_dataset
from utils import load_from_file, save_to_file, name_to_prop, substitute_single_word


def evaluate_lang(true_ltls, out_ltls, true_names, out_names, out_grnds, convert_rule, all_props):
    accs = []
    for true_ltl, out_ltl, true_name, out_name, out_grnd in zip(true_ltls, out_ltls, true_names, out_names, out_grnds):
        if out_ltl == true_ltl:  # Spot cannot handle long but correct LTL formula, e.g. F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F 62_on_the_park
            is_correct = "True"
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_ltl), spot.formula(out_ltl))
                is_correct = "True" if spot_correct else "False"
            except SyntaxError:
                logging.info(f"Syntax error OR formula too long:\n{true_ltl}\n{out_ltl}")
                # breakpoint()

                if set(true_name) == set(out_grnd):
                    true_props = [name_to_prop(name, convert_rule) for name in true_name]
                    true_sub_map = {prop: sym for prop, sym in zip(true_props, all_props[:len(true_props)])}
                    true_ltl_short = substitute_single_word(true_ltl, true_sub_map)[0]

                    out_props = [name_to_prop(name, convert_rule) for name in true_name]
                    out_sub_map = {prop: sym for prop, sym in zip(out_props, all_props[:len(out_props)])}
                    out_ltl_short = substitute_single_word(out_ltl, out_sub_map)[0]

                    logging.info(f"shorten LTLs:\n{true_ltl_short}\n{out_ltl_short}\n")
                    try:  # output LTL formula may have syntax error
                        spot_correct = spot.are_equivalent(spot.formula(true_ltl_short), spot.formula(out_ltl_short))
                        is_correct = "True" if spot_correct else "False"
                    except SyntaxError:
                        logging.info(f"Syntax error:\n{true_ltl_short}\n{out_ltl_short}\n")
                        # breakpoint()

                        is_correct = "Syntax Error"
                else:
                    is_correct = "RER or Grounding Error"
        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc


def evaluate_lang_0(true_ltls, out_ltls, model_responses, validation_meta, instructions, string_match=False):
    accs = []
    # (Model output, Ground truth, LTL type and num propositions, English Instruction) for the ones model got wrong
    incorrects_eval = []

    for true_ltl, out_ltl, resp, type_prop, instruction in zip(true_ltls, out_ltls, model_responses, validation_meta, instructions):
        if true_ltl == out_ltl:  # Spot cannot handle long but correct LTL formula, e.g. F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F & 62_on_the_park U 62_on_the_park & ! 62_on_the_park U ! 62_on_the_park F 62_on_the_park
            is_correct = "True"
        elif string_match:
            is_correct = 'False'
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_ltl), spot.formula(out_ltl))
                is_correct = "True" if spot_correct else "False"
            except SyntaxError:
                is_correct = "Syntax Error"
                logging.info(f"Syntax error:\n{true_ltl}\n{out_ltl}\n")
            except TypeError:
                logging.info(f"Type error:\n{true_ltl}\n{out_ltl}\n")
                breakpoint()

        if is_correct != "True":
            incorrects_eval.append((out_ltl, true_ltl, type_prop[0], len(type_prop[1]), instruction, resp))

        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc, incorrects_eval


def evaluate_lang_new(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, out_grnds):
    accs = []
    for true_ltl, out_ltl, true_sym_ltl, out_sym_ltl, true_name, out_name, out_grnd in zip(true_ltls, out_ltls, true_sym_ltls, out_sym_ltls, true_names, out_names, out_grnds):
        if true_ltl == out_ltl:
            is_correct = "True"
        else:
            try:  # output LTL formula may have syntax error
                spot_correct = spot.are_equivalent(spot.formula(true_sym_ltl), spot.formula(out_sym_ltl))
                if spot_correct:
                    if set(true_name) == set(out_name):  # TODO: check only work if RE == lmk_name when generate grounded dataset
                        if set(true_name) == set(out_grnd):
                            is_correct = "True"
                        else:
                            is_correct = "Grounding Error"
                    else:
                        is_correct = "RER Error"
                else:
                    is_correct = "Symbolic Translation Error"
                    if set(true_name) != set(out_name):
                        is_correct += " | RER Error"
                    if set(true_name) != set(out_grnd):
                        is_correct += " | Grounding Error"
            except SyntaxError:
                logging.info(f"Syntax error: {true_sym_ltl}\n{out_sym_ltl}\n")
                is_correct = "Syntax Error"
        accs.append(is_correct)
    acc = np.mean([True if acc == "True" else False for acc in accs])
    return accs, acc


def evaluate_lang_single(model, valid_iter, valid_meta, analysis_fpath, result_log_fpath, acc_fpath, valid_iter_len):
    """
    Evaluate translation accuracy per LTL pattern type.
    """
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    result_log = [["train_or_valid", "pattern_type", "nprops", "prop_perm", "utterances", "true_ltl", "output_ltl", "is_correct"]]

    meta2accs = defaultdict(list)
    # for idx, ((utt, true_ltl), (pattern_type, prop_perm)) in enumerate(zip(valid_iter, valid_meta)):
    #     nprops = len(prop_perm)
    #     train_or_valid = "valid" if idx < valid_iter_len else "train"  # TODO: remove after having enough data
    #     out_ltl = model.translate([utt])[0].strip()
    #     try:  # output LTL formula may have syntax error
    #         is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
    #         is_correct = "True" if is_correct else "False"
    #     except SyntaxError:
    #         is_correct = "Syntax Error"
    #     logging.info(f"{idx}/{len(valid_iter)}\n{pattern_type} | {nprops} {prop_perm}\n{utt}\n{true_ltl}\n{out_ltl}\n{is_correct}\n")
    #     result_log.append([train_or_valid, pattern_type, nprops, prop_perm, utt, true_ltl, out_ltl, is_correct])
    #     if train_or_valid == "valid":
    #         meta2accs[(pattern_type, nprops)].append(is_correct)
    train_or_valid = "valid"
    nsamples, ncorrects = 0, 0
    for batch in batch(list(zip(valid_iter, valid_meta)), 100):  # batch_size = 100
        utts = [tp[0][0] for tp in batch]
        out_ltls = model.translate(utts)
        for idx, ((utt, true_ltl), (pattern_type, prop_perm, *other_meta)) in enumerate(batch):
            nsamples += 1
            nprops = len(prop_perm)
            out_ltl = out_ltls[idx].strip()
            try:  # output LTL formula may have syntax error
                is_correct = spot.are_equivalent(spot.formula(out_ltl), spot.formula(true_ltl))
                is_correct = "True" if is_correct else "False"
            except SyntaxError:
                is_correct = "Syntax Error"

            if train_or_valid == "valid":
                meta2accs[(pattern_type, nprops)].append(is_correct)
            if nsamples > valid_iter_len:
                train_or_valid = "train"
            logging.info(f"{nsamples}/{len(valid_iter)}\n{pattern_type} | {nprops} {prop_perm}\n{utt}\n{true_ltl}\n{out_ltl}\n{is_correct}\n")
            result_log.append([train_or_valid, pattern_type, nprops, prop_perm, utt, true_ltl, out_ltl, is_correct])
            if is_correct == "True":
                ncorrects += 1
            logging.info(f"partial results: {ncorrects}/{nsamples} = {ncorrects/nsamples}\n")
    save_to_file(result_log, result_log_fpath)

    meta2acc = {meta: np.mean([True if acc == "True" else False for acc in accs]) for meta, accs in meta2accs.items()}
    logging.info(meta2acc)

    analysis = load_from_file(analysis_fpath)
    acc_anaysis = [["LTL Type", "Number of Propositions", "Number of Utterances", "Accuracy"]]
    for pattern_type, nprops, nutts in analysis:
        pattern_type = "_".join(pattern_type.lower().split())
        meta = (pattern_type, int(nprops))
        if meta in meta2acc:
            acc_anaysis.append([pattern_type, nprops, nutts, meta2acc[meta]])
        else:
            acc_anaysis.append([pattern_type, nprops, nutts, "no valid data"])
    save_to_file(acc_anaysis, acc_fpath)

    total_acc = np.mean([True if acc == "True" else False for accs in meta2accs.values() for acc in accs])
    logging.info(f"total validation accuracy: {total_acc}")

    return meta2acc, total_acc


def evaluate_lang_from_file(model, split_dataset_fpath, analysis_fpath, result_log_fpath, acc_fpath):
    _, _, valid_iter, valid_meta = load_split_dataset(split_dataset_fpath)
    return evaluate_lang_single(model, valid_iter, valid_meta,
                                analysis_fpath, result_log_fpath, acc_fpath, len(valid_iter))


def aggregate_results(result_fpaths, filter_types):
    """
    Aggregate accuracy-per-formula results from K-fold cross validation or multiple random seeds.
    Assume files have same columns (LTL Type, Number of Propositions, Number of Utterances, Accuracy)
    and same values for first 3 columns.
    :param result_fpaths: paths to results file to be aggregated
    """
    total_corrects, total_samples = 0, 0
    accs = []
    meta2stats = defaultdict(list)
    for n, result_fpath in enumerate(result_fpaths):
        result = load_from_file(result_fpath, noheader=True)
        print(result_fpath)
        corrects, samples = 0, 0
        for row_idx, row in enumerate(result):
            pattern_type, nprops, nutts, acc = row
            if pattern_type not in filter_types and acc != "no valid data":
                nprops, nutts, acc = int(nprops), int(nutts), float(acc)
                meta2stats[(pattern_type, nprops)].append((nutts*acc, nutts))
                corrects += nutts * acc
                samples += nutts
        total_corrects += corrects
        total_samples += samples
        accs.append(corrects / samples)

    result_aux = load_from_file(result_fpaths[0], noheader=False)
    fields = result_aux.pop(0)
    aggregated_result = [fields]
    for row in result_aux:
        aggregated_result.append(row[:3] + [0.0])
    for row_idx, (pattern_type, nprops, nutts, _) in enumerate(aggregated_result[1:]):
        nprops, nutts = int(nprops), int(nutts)
        stats = meta2stats[(pattern_type, nprops)]
        corrects = sum([corrects_formula for corrects_formula, _ in stats])
        nutts = sum([nutts_formula for _, nutts_formula in stats])
        acc = corrects / nutts if nutts != 0 else "no valid data"
        aggregated_result[row_idx+1] = [pattern_type, nprops, nutts, acc]

    result_fnames = [os.path.splitext(result_fpath)[0] for result_fpath in result_fpaths]
    aggregated_result_fpath = f"{os.path.commonprefix(result_fnames)}_aggregated.csv"
    save_to_file(aggregated_result, aggregated_result_fpath)
    accumulated_acc = total_corrects / total_samples
    accumulated_std = np.std(accs)
    print(f"total accuracy: {accumulated_acc}")
    print(f'standard deviation: {accumulated_std}')
    return accumulated_acc, accumulated_std
