import pandas as pd
import re

def main():
    
    #########################
    ## Load and parse data ##
    #########################
    eval_names = ["1", "2", "3"]

    def r_from_text(text):
        '''Get the relation from a text'''
        return re.search("([a-zA-Z\-]*_[NS][NS])", text).__getitem__(0)

    evals = []
    for name in eval_names:

        df1 = pd.read_csv(f"human-eval/r_{name}.csv", keep_default_na = False)
        df1 = df1.drop(index = [0])

        relation_fit = [
            {
                "text": row["Unnamed: 1"],
                "relation-fit": int(row["Fit of Relation"]),
                "relation": r_from_text(row["Unnamed: 1"]),
                "id": int(row["Unnamed: 4"])
            } for _, row in df1.iterrows()]

        df2 = pd.read_csv(f"human-eval/fr_{name}.csv", keep_default_na = False)
        df2 = df2.drop(index = [0,1])

        data = [
            {
                "text": row["Unnamed: 3"],
                "fluency": int(row["Fluency Rating"]),
                "reasonableness": int(row["Reasonableness Rating"]),
                "relation": row["Unnamed: 5"],
                "id": int(row["Unnamed: 6"])
            } for _, row in df2.iterrows()]

        for r in relation_fit:
            for d in data:
                if d["id"] == r["id"] and d["relation"] == r["relation"]:
                    d["relation-fit"] = r["relation-fit"]

        evals.append(data)

        
    #################################
    ## Human Evaluation Table Data ##
    #################################
    summary = summarize_multi_human_data(evals)
    print("Data for HUMAN EVALUATION table:")
    for metric in summary:
        print("---==---")
        print(f"Averages for {metric}:")

        total = 0
        num_relations = 0
        for r in summary[metric]:
            print(f"{r}: {summary[metric][r]}")
            if r != "None":
                total += summary[metric][r]
                num_relations += 1
        print(f"All Relations: {total/num_relations}")
            
            

def summarize_human_data(data):
    info = {
        "count": dict(),
        "average-fluency": dict(),
        "average-reasonableness": dict(),
        "average-relation-fit": dict(),
    }
    
    for d in data:
        r = d["relation"]
        if r not in info["count"]:
            info["count"][r] = 1
            info["average-fluency"][r] = d["fluency"]
            info["average-reasonableness"][r] = d["reasonableness"]
            
            if r != "None":
                info["average-relation-fit"][r] = d["relation-fit"]
        else:
            info["count"][r] += 1
            info["average-fluency"][r] += d["fluency"]
            info["average-reasonableness"][r] += d["reasonableness"]
            if r != "None":
                info["average-relation-fit"][r] += d["relation-fit"]
            
    for r in info["count"]:
        info["average-fluency"][r] /= info["count"][r]
        info["average-reasonableness"][r] /= info["count"][r]
        if r != "None":
            info["average-relation-fit"][r] /= info["count"][r]
        
    return info

def summarize_multi_human_data(datasets: list):
    evals = []
    for data in datasets:
        evals.append(summarize_human_data(data))
    
    length = len(evals)
    # Get overall averages
    info = dict()
    for metric in evals[0]:
        for relation in evals[0][metric]:
            if metric not in info:
                info[metric] = dict()
            info[metric][relation] = sum(map(lambda x: x[metric][relation], evals))/length
    
   
    return info
            
if __name__ == "__main__":
    main()
