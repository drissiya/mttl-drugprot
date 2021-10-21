# coding=utf-8
# Copyright (c) Microsoft. All rights reserved.
from data_utils.task_def import DataFormat


def dump_rows(rows, out_path, data_format):
    """
    output files should have following format
    :param rows:
    :param out_path:
    :return:
    """
    with open(out_path, "w", encoding="utf-8") as out_f:
        #row0 = rows[0]
        #data_format = detect_format(row0)
        for row in rows:
            #assert data_format == detect_format(row), row
            if data_format == DataFormat.PremiseOnly:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"]))
            elif data_format == DataFormat.PremiseAndOneHypothesis:
                for col in ["uid", "label", "premise", "hypothesis"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"], row["hypothesis"]))
            elif data_format == DataFormat.PremiseAndMultiHypothesis:
                for col in ["uid", "label", "premise"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                hypothesis = row["hypothesis"]
                for one_hypo in hypothesis:
                    if "\t" in str(one_hypo):
                        import pdb; pdb.set_trace()
                hypothesis = "\t".join(hypothesis)
                out_f.write("%s\t%s\t%s\t%s\t%s\n" % (row["uid"], row["ruid"], row["label"], row["premise"], hypothesis))
            elif data_format == DataFormat.Seqence:
                for col in ["uid", "label", "premise", "start", "section", "lenn", "drug", "sentence_input", "start_sentence", "len_sentence", "token", "ade", "modifier"]:
                    if "\t" in str(row[col]):
                        import pdb; pdb.set_trace()
                out_f.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % (row["uid"], row["label"], row["premise"], row["start"], row["section"], row["lenn"], row["drug"], row["sentence_input"], row["start_sentence"], row["len_sentence"], row["token"], row["ade"], row["modifier"]))
            else:
                raise ValueError(data_format)