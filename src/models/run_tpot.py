import pandas as pd
from tpot import TPOTClassifier
import os
import signal
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer

def signal_handler(sig, frame):
    print('\n[INTERRUPT] saving best pipeline...')
    try:
        tpot.export('best_model_at_interrupt.py')
        print('Export successful: best_model_at_interrupt.py')
    except Exception as e:
        print(f'Error during export: {e}')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Find the absolute path to the directory where this script is located
base_path = os.path.dirname(os.path.abspath(__file__))

# Build paths relative to the script location: 
# Move up to 'src', then down to 'data/preprocessed'
train_path = os.path.abspath(os.path.join(base_path, "..", "..", "data", "processed", "selected_features_train.csv"))
test_path = os.path.abspath(os.path.join(base_path, "..", "..", "data", "processed", "selected_features_test.csv"))

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

f1_scorer = make_scorer(f1_score, pos_label=1)
my_cv = TimeSeriesSplit(n_splits=5)

try:
    print("Starting AutoML Session...")
    tpot = TPOTClassifier(generations=1000, 
                        population_size=100,
                        cv=my_cv,
                        scoring=f1_scorer, 
                        verbosity=2, 
                        random_state=42, 
                        n_jobs=-1,
                        periodic_checkpoint_folder='tpot_checkpoints',
                        #early_stop=10
                        )

    X_train=train.drop(columns=['rain_tomorrow'])
    X_test=test.drop(columns=['rain_tomorrow'])
    y_train=train['rain_tomorrow']
    y_test=test['rain_tomorrow']
    
    tpot.fit(X_train, y_train) 
    
    print(tpot.score(X_test, y_test))

    tpot.export('best_tpot_pipeline.py')
    print("Optimization finished successfully!")
    
except Exception as e:
    with open("error_log.txt", "w") as f:
        f.write(str(e))
    print("An error occurred. Check error_log.txt")


