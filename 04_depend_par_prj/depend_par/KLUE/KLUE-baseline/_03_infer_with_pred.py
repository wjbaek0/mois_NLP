import os, json
import argparse
import pandas as pd

'''
    CMD => python infer_with_pred.py -v {str: version of pred} -dt {str: dataset type}
'''

'''
    의존관계태그 리스트
'''
def get_dep_labels():
    dep_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
        "AP_OBJ",
        "AP_CNJ",
    ]
    return dep_labels

'''
    품사태그 리스트
'''
def get_pos_labels():
    pos_labels = [
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
    ]

    return pos_labels

'''
    pred 읽어오기
'''
def load_pred(args):
    pred_path = f"./output/klue-dp/version_{args.version}/transformers/pred/pred-0.json"
    print(f"Load pred file... => {pred_path}")
    pred_df = pd.read_table(pred_path, sep=" ",names=["head_preds","type_preds","head_labels","type_labels"])
    print(pred_df[:21])
    dep_labels_list = get_dep_labels()
    pos_labels_list = get_pos_labels()

    for i in range(len(pred_df)):
        pred_df.loc[i, 'type_labels'] = dep_labels_list[pred_df.loc[i, 'type_labels']]
        pred_df.loc[i, 'type_preds'] = dep_labels_list[pred_df.loc[i, 'type_preds']]

    # print(pred_df[:6])

    return pred_df


def load_dataset(df, args):
    file_path=f"./data/klue-dp-v1.3/NXDP1902103231_{args.dataset_type}.tsv"
    infer_path=f"./output/klue-dp/version_{args.version}/infer_results.tsv"
    print(f"Load dataset file... => {file_path}")

    sent_id = -1
    token_idx = -1
    with open(file_path, "r", encoding="utf-8") as rf:

        print(f"Create infer file... => {infer_path}")
        with open(infer_path, "w", encoding="utf-8") as wf:
            for line in rf:
                line = line.strip()
                if line == "" or line == "\n" or line == "\t":
                    wf.write(line+'\n')
                    continue

                if line.startswith("#"):
                    wf.write(line+'\n')
                    parsed = line.strip().split("\t")
                    if len(parsed) != 2:  # metadata line about dataset
                        continue
                    else:
                        sent_id += 1
                        text = parsed[1].strip()
                        guid = parsed[0].replace("##", "").strip()
                else:
                    token_idx += 1
                    line = make_infer(df, line, token_idx)
                    wf.write(line+'\n')


'''
    infer 문장으로 변경 -> head, type 값을 pred된 값으로 치환    
'''
def make_infer(df, line, token_idx):
    elements_list = line.split("\t")
    elements_list[4] = str(df.loc[token_idx, 'head_preds'])
    elements_list[5] = df.loc[token_idx, 'type_preds']
    # print(elements_list)

    line = '\t'.join(elements_list)
    return line

'''
    tsv 형식으로 저장
'''
# def make_tsv(df, args):
#     df.to_csv(os.path.join("{}.tsv".format(os.path.splitext(args.json_path)[0])), sep='\t', encoding='utf-8-sig', index=False, header=False)

def run(args):
    df = load_pred(args)
    load_dataset(df, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert pred")
    parser.add_argument("-v","--version", required=False, type=str,   help="pred version", default="30")
    parser.add_argument("-dt","--dataset_type", required=False, type=str,   help="dataset type", default="test")
    args = parser.parse_args()

    run(args)