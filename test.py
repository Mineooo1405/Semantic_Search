import matchzoo
print(matchzoo.__version__) # To confirm the version
# Try to find Evaluator in common places
try:
    from matchzoo.trainers import Evaluator
    print("Found Evaluator in matchzoo.trainers")
except ImportError:
    print("Not in matchzoo.trainers")
try:
    from matchzoo.evaluation import Evaluator
    print("Found Evaluator in matchzoo.evaluation")
except ImportError:
    print("Not in matchzoo.evaluation")
try:
    from matchzoo.metrics import Evaluator
    print("Found Evaluator in matchzoo.metrics")
except ImportError:
    print("Not in matchzoo.metrics")
# If you have an idea of other possible modules, add them here.