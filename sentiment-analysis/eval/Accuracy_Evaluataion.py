import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import sys

# Check if the user has provided a command-line argument for the file path
if len(sys.argv) < 2:
    print("Usage: python evaluation.py path_to_your_excel_file.xlsx")
    sys.exit(1)  # Exit the script if the file path is not provided

# Use the command-line argument for the file path
file_path = sys.argv[1]

# Load the Excel file
df = pd.read_excel(file_path)

nogold = 0
gold = 0
nopred = 0
pred = 0
correct = 0

for index,row in df.iterrows():
  row['gold_polarity_label'] = str(row['gold_polarity_label'])
  row['polarity_label'] = str(row['polarity_label'])
  if row['gold_polarity_label'] == '' or row['gold_polarity_label'] == 'nan':
    nogold = nogold + 1
  else:
    gold = gold + 1
    if row['polarity_label'] == '' or row['polarity_label'] == 'nan':
      nopred = nopred + 1
    else:
      pred = pred + 1
      if row['polarity_label'] == row['gold_polarity_label']:
        correct = correct + 1

precision = correct / pred
recall = correct / gold
fscore = 2*precision*recall/(precision+recall)

print('Total rows: ', (gold+nogold))
print('Rows with a gold label: ', gold)
print('Rows with a gold label and a prediction: ', pred)
print('Rows with a correct prediction: ', correct)
print('Rows with a wrong prediction: ', pred-correct)
print('Precision: ', precision)
print('Recall: ', recall)
print('F-score: ', fscore)


#Now calculate macro averaged metrics


# Initialize counters for macro-averaging
class_stats = {'Positive': {'tp': 0, 'fp': 0, 'fn': 0},
               'Negative': {'tp': 0, 'fp': 0, 'fn': 0},
               'Neutral': {'tp': 0, 'fp': 0, 'fn': 0}}
               	


for index,row in df.iterrows():
  row['gold_polarity_label'] = str(row['gold_polarity_label'])
  row['polarity_label'] = str(row['polarity_label'])
  gold_label = row['gold_polarity_label']
  predicted_label = row['polarity_label']
  if row['gold_polarity_label'] == '' or row['gold_polarity_label'] == 'nan':
    continue
  else:
    if row['polarity_label'] == '' or row['polarity_label'] == 'nan':
      class_stats[gold_label]['fn'] += 1
      continue
    else:
      if row['polarity_label'] == row['gold_polarity_label']:
        class_stats[gold_label]['tp'] += 1
      else:
        class_stats[gold_label]['fn'] += 1
        class_stats[predicted_label]['fp'] += 1

# Calculate macro-averaged precision, recall, and F-score
macro_precision = macro_recall = macro_fscore = 0
for label in class_stats:
  tp = class_stats[label]['tp']
  fp = class_stats[label]['fp']
  fn = class_stats[label]['fn']
  
  precision = tp / (tp + fp) if (tp + fp) else 0
  recall = tp / (tp + fn) if (tp + fn) else 0
  fscore = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
  
  macro_precision += precision
  macro_recall += recall
  macro_fscore += fscore

num_classes = len(class_stats)
macro_precision /= num_classes
macro_recall /= num_classes
macro_fscore /= num_classes

print(f'Macro-averaged Precision: {macro_precision}')
print(f'Macro-averaged Recall: {macro_recall}')
print(f'Macro-averaged F-score: {macro_fscore}')



## Filter out rows where the gold classification label is blank
#df['gold_polarity_label'] = df['gold_polarity_label'].astype(str)  # Convert gold labels to string to handle any numeric values
#df_filtered = df[df['gold_polarity_label'].notna() & (df['gold_polarity_label'] != '') & (df['gold_polarity_label'] != 'nan')]  # Filter out blanks and NaN (now converted to 'nan' as string)
##df_filtered = df[df['gold_polarity_label'].notna()]
#
## Extract gold labels and predicted labels
#gold_labels = df_filtered['gold_polarity_label']  # Assuming column 'L' contains the gold labels
#predicted_labels = df_filtered.iloc[:, -1]  # Assuming the last column contains the predicted labels
#
## Calculate precision, recall, and F-score for each class
#precision, recall, fscore, _ = precision_recall_fscore_support(gold_labels, predicted_labels, labels=['Positive', 'Negative', 'Neutral'], average=None)
#
## Calculate overall precision, recall, and F-score (weighted by support, i.e., the number of true instances for each label)
#overall_precision, overall_recall, overall_fscore, _ = precision_recall_fscore_support(gold_labels, predicted_labels, average='micro')
#
## Print the results
#print("Class-wise Metrics:")
#for label, p, r, f in zip(['Positive', 'Negative', 'Neutral'], precision, recall, fscore):
#    print(f"{label}: Precision={p:.3f}, Recall={r:.3f}, F-score={f:.3f}")
#
#print("\nOverall Metrics:")
#print(f"Precision={overall_precision:.3f}, Recall={overall_recall:.3f}, F-score={overall_fscore:.3f}")
