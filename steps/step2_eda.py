# steps/step2_eda.py
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df):

    # ── Output folder ──────────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'outputs')
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # Creates folder if it doesn't exist

    # ── Correlations with Class ────────────────────────────────
    correlations = df.corr()['Class'].sort_values()
    print(correlations)

    # ── Chart 1-4: EDA Overview ────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Credit Card Fraud - EDA', fontsize=16)

    # Chart 1: Class imbalance
    axes[0, 0].bar(['Legitimate', 'Fraud'],
                   df['Class'].value_counts(),
                   color=['steelblue', 'crimson'])
    axes[0, 0].set_title('Class Distribution')
    axes[0, 0].set_ylabel('Number of Transactions')

    # Chart 2: Transaction amount distribution
    df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6,
        label='Legitimate', ax=axes[0, 1], color='steelblue')
    df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6,
        label='Fraud', ax=axes[0, 1], color='crimson')
    axes[0, 1].set_xlim(0, 2500)
    axes[0, 1].set_title('Transaction Amount Distribution')
    axes[0, 1].set_xlabel('Amount (£)')
    axes[0, 1].legend()

    # Chart 3: Fraud over time
    df[df['Class'] == 0].plot.scatter(x='Time', y='Amount',
        alpha=0.1, ax=axes[1, 0], color='steelblue', label='Legitimate', s=1)
    df[df['Class'] == 1].plot.scatter(x='Time', y='Amount',
        alpha=0.5, ax=axes[1, 0], color='crimson', label='Fraud', s=10)
    axes[1, 0].set_title('Fraud vs Time and Amount')
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Amount (£)')

    # Chart 4: Key feature means by class
    feature_means = df.groupby('Class')[['V14', 'V17', 'V12', 'V10']].mean()
    feature_means.T.plot(kind='bar', ax=axes[1, 1], color=['steelblue', 'crimson'])
    axes[1, 1].set_title('Key Feature Means by Class')
    axes[1, 1].set_xlabel('Feature')
    axes[1, 1].set_ylabel('Mean Value')
    axes[1, 1].legend(['Legitimate', 'Fraud'])
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'eda_overview.png'), dpi=150)
    plt.close()
    print("Saved: outputs/eda_overview.png")

    # ── Correlation Matrix ─────────────────────────────────────
    columns_of_interest = ['Class', 'Amount', 'Time', 'V10', 'V12', 'V14', 'V17', 'V2', 'V4', 'V11']
    corr_matrix = df[columns_of_interest].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        vmin=-1, vmax=1,
        linewidths=0.5
    )
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print("Saved: outputs/correlation_matrix.png")