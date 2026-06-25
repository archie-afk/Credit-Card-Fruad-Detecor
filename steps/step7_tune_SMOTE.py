from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def tune_smote_ratio(X_train, y_train, X_test, y_test):
    """
    Step 7: Hyperparameter tuning for SMOTE sampling strategy.
    Locks in champion XGBoost settings and sweeps data ratios to optimize Recall.
    """
    print("--- Step 7: Tuning SMOTE Sampling Ratio ---")
    
    # 1. Pipeline locks in your step 6 champion parameters
    pipeline = Pipeline([
        ('smote', SMOTE(random_state=42)),
        ('xgb', XGBClassifier(
            learning_rate=0.2,
            max_depth=8,
            n_estimators=150,
            random_state=42,
            eval_metric='logloss'
        ))
    ])
    
    # 2. Sweep ratios to see which one crosses the 90% fraud detection mark
    param_grid = {
        'smote__sampling_strategy': [0.10, 0.15, 0.20, 0.25, 0.30]
    }
    
    # 3. Optimize for 'recall' to prioritize catching fraud
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='recall', 
        n_jobs=-1,
        verbose=2
    )
    
    print("Evaluating SMOTE strategies on raw training data splits...")
    grid_search.fit(X_train, y_train)
    
    print("\nGrid Search Complete!")
    print(f"Best SMOTE Strategy Found: {grid_search.best_params_['smote__sampling_strategy']}")
    
    # 4. Final Real-World Evaluation
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    
    print("\n" + "="*20 + " FINAL STEP 7 EVALUATION " + "="*20)
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("="*56 + "\n")
    
    return best_pipeline