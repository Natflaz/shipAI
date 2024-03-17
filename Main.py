from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.evaluate import evaluate_multi_output_model, print_evaluation_results
from src.init import X_train, X_test, y_train, y_test
from src.model import VotingEnsembleRegressor

print("compressor decay")

Voting = VotingEnsembleRegressor(weights=[3, 1])

Voting.fit(X_train, y_train[['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']])
predictions = Voting.predict(X_test)

results = evaluate_multi_output_model(predictions, y_test[
    ['GT Compressor decay state coefficient', 'GT Turbine decay state coefficient']].to_numpy())
print_evaluation_results(results)
