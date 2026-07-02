import numpy as np
from sklearn.metrics import det_curve, auc

"""
Computes the FREUID competition metrics:
    - AuDET: Area under the Detection Error Trade-off (DET) curve (BPCER vs APCER).
    - APCER @ 1% BPCER: Attack Presentation Classification Error Rate at 1% Bona-Fide Presentation Classification Error Rate.
    - FREUID score: Harmonic mean of g_audet = 1 - AuDET and g_apcer = 1 - APCER@1%BPCER.

y_true: binary labels (0 for bona-fide, 1 for attack)
y_score: predicted probabilities or confidence scores for the attack class (1)
"""

def fxn_compute_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    
    if np.isnan(y_score).any():
        y_score = np.nan_to_num(y_score, nan=0.5) # Use: replaces nans in yscore by 0.5
        
    # Use: Check if there are at least two unique points to compute DET curve
    if len(np.unique(y_score)) < 2:
        return {
            'AuDET': 0.5,
            'APCER_at_1pct_BPCER': 1.0,
            'g_audet': 0.5,
            'g_apcer': 0.0,
            'FREUID': 1.0
        }
    
    # Use: det_curve returns: fpr, fnr, thresholds
    #      BPCER = fpr (False Positive Rate)
    #      APCER = fnr (False Negative Rate)
    bpcer, apcer, thresholds = det_curve(y_true, y_score)
    
    # Use: Area under the DET curve (AuDET)
    #      Note: sklearn's det_curve returns bpcer (fpr) in ascending order and apcer (fnr) in descending order.
    #      We can use auc(bpcer, apcer) directly because bpcer is monotonically increasing.
    if len(bpcer) < 2:
        audet = 0.5
    else:
        audet = auc(bpcer, apcer)
    
    # Use: APCER @ 1% BPCER
    #      We want to find the APCER value when BPCER is exactly 1% (0.01).
    #      Since bpcer is sorted and discrete, we can interpolate or find the closest value.
    #      If there isn't an exact match, we find the APCER corresponding to the largest BPCER <= 0.01.
    #      Thus we find the threshold where BPCER <= 0.01 and BPCER is maximized.
    target_bpcer = 0.01    
    idx = np.where(bpcer <= target_bpcer)[0]
    if len(idx) > 0:
        best_idx = idx[-1]
        apcer_at_1pct = apcer[best_idx]
    else:
        apcer_at_1pct = apcer[0]
        
    g_audet = 1.0 - audet
    g_apcer = 1.0 - apcer_at_1pct
    
    # Use: Harmonic mean formula for FREUID:
    #      FREUID = 1 - (2 * g_audet * g_apcer) / (g_audet + g_apcer)
    if (g_audet + g_apcer) > 0:
        freuid = 1.0 - (2.0 * g_audet * g_apcer) / (g_audet + g_apcer)
    else:
        freuid = 1.0
        
    return {
        'AuDET': float(audet),
        'APCER_at_1pct_BPCER': float(apcer_at_1pct),
        'g_audet': float(g_audet),
        'g_apcer': float(g_apcer),
        'FREUID': float(freuid)
    }

if __name__ == '__main__':
    np.random.seed(42)

    y_true_test = np.random.randint(0, 2, size=1000)
    y_score_test = np.random.rand(1000)
    
    # Use: Add some correlation 
    y_score_test = y_score_test + y_true_test * 0.5
    y_score_test = np.clip(y_score_test, 0, 1)
    
    metrics = fxn_compute_metrics(y_true_test, y_score_test)
    print("Test metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
