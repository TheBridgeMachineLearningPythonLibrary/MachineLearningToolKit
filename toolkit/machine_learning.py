from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import pandas as pd

def balance_binary_target(df, strategy='smote', minority_ratio=None, visualize=False):
    """
    This function balances a target binary variable of a dataframe using different oversampling strategies.

    Args:
    - df: dataframe with the target variable to balance.
    - strategy: oversampling strategy to use (default='smote'). The options are: 'smote', 'adasyn' or 'random'.
    - minority_ratio: proportion of the minority class after oversampling (default=None).
    - visualize: if True, visualize the balanced data (default=False).

    Returns:
    - DataFrame: dataframe with the balanced target variable.
    """

    # Automatically detect the target variable column.
    target_col = df.select_dtypes(include=['bool', 'int', 'float']).columns[0]

    # Separate target variable and predictor variables
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Oversampling the minority class using the selected strategy
    if strategy == 'smote':
        sampler = SMOTE(random_state=42)
    elif strategy == 'adasyn':
        sampler = ADASYN(random_state=42)
    elif strategy == 'random':
        sampler = RandomOverSampler(random_state=42)
    else:
        raise ValueError("Estrategia de sobremuestreo inválida. Las opciones son: 'smote', 'adasyn' o 'random'.")

    X_resampled, y_resampled = sampler.fit_resample(X, y)

    # Adjust the minority class ratio if specified
    if minority_ratio is not None:
        target_counts = y_resampled.value_counts()
        minority_class = target_counts.idxmin()
        majority_class = target_counts.idxmax()

        minority_count = target_counts[minority_class]
        majority_count = target_counts[majority_class]

        desired_minority_count = int(minority_ratio * (minority_count + majority_count))

        if desired_minority_count < minority_count:
            drop_indices = y_resampled[y_resampled == minority_class].index[:minority_count - desired_minority_count]
            X_resampled = X_resampled.drop(drop_indices)
            y_resampled = y_resampled.drop(drop_indices)
        elif desired_minority_count > minority_count:
            extra_count = desired_minority_count - minority_count
            extra_X, extra_y = sampler.fit_resample(X_resampled[y_resampled == minority_class], y_resampled[y_resampled == minority_class])
            X_resampled = pd.concat([X_resampled, extra_X], axis=0)
            y_resampled = pd.concat([y_resampled, extra_y], axis=0)

    # Display the balanced data if specified
    if visualize:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title('Distribución de la variable objetivo balanceada')
        y_resampled.value_counts().plot(kind='bar', ax=ax)
        ax.set_xlabel(target_col)
        ax.set_ylabel('Frecuencia')

    # Combine the predictor variables and the balanced target variable in a new dataframe.
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

    return df_resampled