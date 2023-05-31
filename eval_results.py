from src.db_connection import get_all_completed_experiment
from src.experiment import ExperimentDB
import numpy as np 
from scipy.stats import fisher_exact, ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
def extract_score(exp):
    pred_tuple = exp.list_human_pred_test
    num_test = len(pred_tuple)
    if num_test == 0:
        return None, None
    test_index = [i[0] for i in pred_tuple]
    pred = [i[1] for i in pred_tuple]
    y = exp.y
    ground_truth = y[test_index]
    scores = []
    for i in range(num_test):
        if pred[i] == ground_truth[i]:
            scores.append(1)
        else:
            scores.append(0) 
   
    return scores, np.mean(scores)




al_scores = []
random_scores = []


al_exp = []
random_exp = []

experiments_db = get_all_completed_experiment()

for exp_db in experiments_db:
    exp = ExperimentDB(exp_db)
    print(datetime.fromtimestamp(exp.session_id))
    if datetime.fromtimestamp(1670060232) > datetime.fromtimestamp(exp.session_id):
        al_type = exp.al_type
        if al_type == 0:
            scores, avg_score =extract_score(exp)
            if scores is not None:
                random_scores.append(scores)
        elif al_type == 1:
            scores, avg_score =extract_score(exp)
            if scores is not None:
                al_scores.append(scores)
    else:
        print('to recent')
    
 

# extracting data from that awful design of mine
exp.list_human_pred_test
exp.X

means_al = [np.mean(i) for i in al_scores]
means_rd = [np.mean(i) for i in random_scores]
all_data = [means_al,means_rd]



print('AL Experiment Result...')

print("Total trials", len(means_al)+len(means_rd))
print('Mean accuracy with al :',np.mean(means_al), 'over', len(means_al))
print('Mean accuracy with random :',np.mean(means_rd), 'over',len(means_rd))


stat, p_val = ttest_ind(means_al, means_rd, equal_var=False)
print(p_val)
if p_val < 0.05:
    print('with significane')
else:
    print('without significance')



plt.boxplot(all_data, showmeans=True)
plt.show()
fig1, ax1 = plt.subplots()

#sns.boxenplot(data=all_data, ax=ax1, showfliers=False, scale="linear")
#sns.violinplot(data=all_data,cut=0, ax=ax1)
ax1.yaxis.grid(True)
ax1.set_xticks([y  for y in range(len(all_data))])

ax1.set_ylabel('Accuracy')

# add x-tick labels
# plt.setp(ax1, xticks=[y  for y in range(len(all_data))],
#          xticklabels=['with active learning', 'random baseline'])
plt.show()



