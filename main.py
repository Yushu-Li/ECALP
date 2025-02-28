import argparse
import torch
from utils.iterative_label_propagation import iterative_label_propagation
import os

def main(args):
    """Main function to run the label propagation with various configurations."""

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    datasets = args.datasets.split('/')
    dataset_results = {}  # Store results for each dataset

    for dataset_name in datasets:
        # Load pre-calculated features
        try:
            pre_calculated = torch.load(os.path.join(args.features_path, args.task, args.clip_model, f'{dataset_name}.pt'))
            image_features_load, text_features_load, gts = pre_calculated['image_features'], pre_calculated['text_features'], pre_calculated['labels']
            text_features = text_features_load.to(device)  # Move to the GPU
        except FileNotFoundError:
            print(f"Error: Feature file not found for dataset {dataset_name}")
            continue # Skip to the next dataset

        # Preparing few-shot data if needed
        few_shot_image_features = None
        few_shot_labels = None

        if args.mode == 'FS':
            try:
                few_shot_pre_calculate = torch.load(os.path.join(args.features_path, args.task, args.clip_model, 'few_shots', f'{dataset_name}.pt'))
                few_shot_image_features, few_shot_labels = few_shot_pre_calculate['image_features'], few_shot_pre_calculate['labels']
                # Use specified number of shots and flatten
                few_shot_image_features = few_shot_image_features[:, :args.num_shots, :].reshape(-1, few_shot_image_features.shape[-1]).to(device)
                few_shot_labels = few_shot_labels[:, :args.num_shots, :].reshape(-1, few_shot_labels.shape[-1]).to(device)
            except FileNotFoundError:
                print(f"Error: Few-shot feature file not found for dataset {dataset_name}")
                continue # Skip to the next datatset

        # Evaluation setup
        correct_top1_lp = 0 # Top-1 accuracy
        num_samples = image_features_load.shape[0]

        # Disable gradient calculation
        with torch.no_grad():
            for i in range(num_samples):

                # Stack features and labels
                image_features = torch.cat((image_features_load[:i+1], image_features_load[i].unsqueeze(0)), dim=0)

                # ----------------- Label Propagation -----------------
                predictions_lp = iterative_label_propagation(
                    image_features, text_features, args.k_text, args.k_image, args.k_fewshot,
                    args.gamma, args.alpha, fewshot_image_features=few_shot_image_features,
                    fewshot_labels=few_shot_labels, max_iter=args.num_iterations)

                # ----------------- Accuracy Calculation -----------------
                gt = gts[i]

                predictions_lp = predictions_lp.argmax(dim=1)

                correct_top1_lp += (predictions_lp == gt).sum().item()


        # Print Results
        accuracy_lp = (correct_top1_lp / num_samples) * 100
        print(f"Dataset: {dataset_name}, Mode: {args.mode}")
        print(f"Top-1 Accuracy(LP):     {accuracy_lp:.2f}%")
        print("-" * 40)

        # Store results
        dataset_results[dataset_name] = accuracy_lp

        # Save results to file if specified
        if args.output_file:
            with open(args.output_file, 'a') as f:
                f.write(f"Dataset: {dataset_name}, Mode: {args.mode}\n")
                f.write(f"Top-1 Accuracy(LP): {(correct_top1_lp / num_samples) * 100:.2f}%\n")
                f.write("\n")

    # Summarize results
    print("\n--- Summary of Results ---")
    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write("\n--- Summary of Results ---\n")
    for dataset, accuracy in dataset_results.items():
        print(f"{dataset}: {accuracy:.2f}%")
        if args.output_file:
            with open(args.output_file, 'a') as f:
                f.write(f"{dataset}: {accuracy:.2f}%\n")
    print("--------------------------")
    if args.output_file:
        with open(args.output_file, 'a') as f:
            f.write("--------------------------\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label Propagation with CLIP Features')

    # Data and model paths
    parser.add_argument('--task', type=str, choices=['fine_grained', 'style_transfer', 'out_of_distribution'], default="fine_grained", help='Task to process')
    parser.add_argument('--datasets', type=str, default="DTD", help='Datasets to process (separated by "/")')
    parser.add_argument('--features_path', type=str, default='./features/', help='Path to the pre-calculated features')
    parser.add_argument('--clip_model', type=str, choices=['RN', 'VIT'], default='RN', help='Name of the CLIP model')
    parser.add_argument('--output_file', type=str, help='Path to save the results in txt')

    # Label propagation hyperparameters
    parser.add_argument('--k_text', type=int, default=3, help='Number of nearest text neighbors')
    parser.add_argument('--k_image', type=int, default=8, help='Number of nearest image neighbors')
    parser.add_argument('--k_fewshot', type=int, default=8, help='Number of few-shot examples for KNN')
    parser.add_argument('--gamma', type=float, default=10.0, help='Scaling factor for image-image similarity')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for text-based predictions')
    parser.add_argument('--num_iterations', type=int, default=3, help='Number of label propagation iterations')

    # Zero-shot or Few-shot
    parser.add_argument('--mode', type=str, choices=['ZS', 'FS'], default='ZS', help='Zero-shot (ZS) or Few-shot (FS) mode')
    parser.add_argument('--num_shots', type=int, default=16, help='Number of few-shot examples per class')

    args = parser.parse_args()

    main(args)