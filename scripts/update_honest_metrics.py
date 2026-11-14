import csv
import json

def update_table():
    with open('reports/cross_dataset/cross_table.json') as f:
        data = json.load(f)

    rows = []
    for train_ds, test_dict in data.items():
        for test_ds, model_dict in test_dict.items():
            for model_name, m in model_dict.items():
                cm = m['confusion_matrix']
                tp_fake = cm[0][0]
                fn_fake = cm[0][1]
                fp_fake = cm[1][0]
                tn_fake = cm[1][1]
                
                rec_fake = tp_fake / (tp_fake + fn_fake) if (tp_fake + fn_fake) > 0 else 0
                rec_real = tn_fake / (tn_fake + fp_fake) if (tn_fake + fp_fake) > 0 else 0
                bal_acc = (rec_fake + rec_real) / 2.0
                
                rows.append({
                    'Trained on': train_ds.upper(),
                    'Test dataset': test_ds,
                    'Model': model_name,
                    'Raw Accuracy (%)': f"{m['accuracy']*100:.2f}",
                    'Balanced Accuracy (%)': f"{bal_acc*100:.2f}",
                    'Recall Fake (%)': f"{rec_fake*100:.2f}",
                    'Recall Real (%)': f"{rec_real*100:.2f}",
                    'F1-Score (%)': f"{m['f1_score']*100:.2f}",
                    'ROC-AUC': f"{m['roc_auc']:.4f}",
                    'Brier Score': f"{m['brier_score']:.4f}"
                })

    fieldnames = [
        'Trained on', 'Test dataset', 'Model',
        'Raw Accuracy (%)', 'Balanced Accuracy (%)',
        'Recall Fake (%)', 'Recall Real (%)',
        'F1-Score (%)', 'ROC-AUC', 'Brier Score'
    ]
    with open('reports/cross_dataset_table.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Updated reports/cross_dataset_table.csv with genuine Balanced Accuracy and per-class Recall!")

if __name__ == '__main__':
    update_table()
