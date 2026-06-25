import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create an outputs directory if it doesn't exist yet
os.makedirs('outputs', exist_ok=True)


# %% [2] Function: Run EDA Dashboard
def run_eda(df):
    """
    Generates a 2x2 EDA dashboard and saves it directly to the outputs folder.
    """
    print("\nClass distribution:")
    counts = df['Class'].value_counts()
    percentages = df['Class'].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Count': counts, 'Percentage (%)': percentages}))
    print("="*40 + "\n")
    
    print("Generating EDA Dashboard charts...")
    
    # Initialize a 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Credit Card Fraud - Comprehensive EDA Dashboard', fontsize=18, fontweight='bold')

    # ── Chart 1 (Top-Left): Correlation Matrix Heatmap
    columns_of_interest = ['Class', 'Amount', 'Time', 'V10', 'V12', 'V14', 'V17', 'V2', 'V4', 'V11']
    corr_matrix = df[columns_of_interest].corr()

    sns.heatmap(
        corr_matrix, 
        annot=True,             
        cmap='coolwarm',        
        fmt=".2f",              
        vmin=-1, vmax=1,        
        linewidths=0.5,
        ax=axes[0, 0]          
    )
    axes[0, 0].set_title('Feature Correlation Matrix', fontsize=14, pad=10)

    # ── Chart 2 (Top-Right): Average value of key features by class
    feature_means = df.groupby('Class')[['V14', 'V17', 'V12', 'V10']].mean()
    feature_means.T.plot(kind='bar', ax=axes[0, 1], color=['steelblue', 'crimson'])
    axes[0, 1].set_title('Key Feature Means by Class', fontsize=14, pad=10)
    axes[0, 1].set_xlabel('Feature')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].legend(['Legitimate', 'Fraud'])
    axes[0, 1].tick_params(axis='x', rotation=0)

    # ── Chart 3 (Bottom-Left): Fraud over time
    df[df['Class'] == 0].plot.scatter(x='Time', y='Amount',
        alpha=0.1, ax=axes[1, 0], color='steelblue', label='Legitimate', s=1)
    df[df['Class'] == 1].plot.scatter(x='Time', y='Amount',
        alpha=0.5, ax=axes[1, 0], color='crimson', label='Fraud', s=10)
    axes[1, 0].set_title('Fraud vs Time and Amount', fontsize=14, pad=10)
    axes[1, 0].set_xlabel('Time (seconds)')
    axes[1, 0].set_ylabel('Amount (£)')

    # ── Chart 4 (Bottom-Right): Transaction amount by class (zoomed in)
    df[df['Class'] == 0]['Amount'].hist(bins=50, alpha=0.6,
        label='Legitimate', ax=axes[1, 1], color='steelblue')
    df[df['Class'] == 1]['Amount'].hist(bins=50, alpha=0.6,
        label='Fraud', ax=axes[1, 1], color='crimson')
    axes[1, 1].set_xlim(0, 2500)   
    axes[1, 1].set_title('Transaction Amount Distribution', fontsize=14, pad=10)
    axes[1, 1].set_xlabel('Amount (£)')
    axes[1, 1].legend()

    # Final layout adjustments 
    plt.tight_layout()
    
    # Save the entire dashboard directly to your outputs directory
    output_path = 'outputs/eda_dashboard.png'
    plt.savefig(output_path, dpi=300)
    print(f"Dashboard saved successfully to: {output_path}")
    
    plt.show()