# Fairness / Bias Audit

- **Domain**: celebdf
- **Model**: convnext_tiny.fb_in22k_ft_in1k
- **Samples**: 1175 (750 fake, 425 real)

## Protected-Attribute Coverage

No protected-attribute labels were found in the dataset metadata, so a Fitzpatrick / age / gender fairness matrix cannot be produced from the released data. The proxy strata below are the substitutes that CAN be measured today.


## Overall (threshold 0.5)

- ROC-AUC: 0.9208
- Fake recall: 0.9960
- False-positive rate on authentic faces: 0.5624

## Proxy Strata

### blur

| Stratum | n real | n fake | Fake recall | FPR (authentic called fake) | Mean P(fake) on authentic |
|---|---:|---:|---:|---:|---:|
| Q1 | 109 | 185 | 1.0000 | 0.6972 | 0.5940 |
| Q2 | 97 | 197 | 0.9949 | 0.4536 | 0.4456 |
| Q3 | 98 | 195 | 0.9949 | 0.5204 | 0.4939 |
| Q4 | 121 | 173 | 0.9942 | 0.5620 | 0.5086 |

### brightness

| Stratum | n real | n fake | Fake recall | FPR (authentic called fake) | Mean P(fake) on authentic |
|---|---:|---:|---:|---:|---:|
| Q1 | 117 | 177 | 0.9887 | 0.4872 | 0.4547 |
| Q2 | 91 | 203 | 1.0000 | 0.5385 | 0.5096 |
| Q3 | 116 | 177 | 0.9944 | 0.5603 | 0.5184 |
| Q4 | 101 | 193 | 1.0000 | 0.6733 | 0.5761 |

### contrast

| Stratum | n real | n fake | Fake recall | FPR (authentic called fake) | Mean P(fake) on authentic |
|---|---:|---:|---:|---:|---:|
| Q1 | 122 | 172 | 0.9942 | 0.5164 | 0.4663 |
| Q2 | 101 | 193 | 1.0000 | 0.5050 | 0.4987 |
| Q3 | 92 | 201 | 1.0000 | 0.5652 | 0.4981 |
| Q4 | 110 | 184 | 0.9891 | 0.6636 | 0.5894 |

## Flagged Strata

No proxy stratum exceeded 1.5x the overall false-positive rate.

_Proxy strata, not protected attributes. A flagged stratum means the detector misclassifies authentic faces more often under those capture conditions and warrants investigation._
