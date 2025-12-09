import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Callable
import torch
from torchvision.ops import box_iou
import csv, json

# ==============================================================================
#                               TYPE DEFINITIONS
# ==============================================================================
# Dictionary where keys are class names and values are dictionaries with the metric results 
# (e.g. { 'objectX' : {'Precision': 0.9876543210, 'Recall': ..., ...})
MetricsData = Dict[str, Dict[str, float | int]]

# List of dictionaries, where each dictionary stores info about a bounding box (image_name, class, coordinates, and 
# score (only for predictions))
LoadedBoxesData = List[Dict[str, Any]]

# Tuple that stores info about matched predictions (true positives, false positives, scores, total GT instances)
MatchedPredsData = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]

# Tuple of two elements:
# 1) Dictionary that contains dictionaries with info about the GT BBoxes of a target class for every image ->
# {'boxes': List[List[float]], 'matches': List[bool]}
# 2) Total number of instances for the target class across all images.
GTsPerImageData = Tuple[Dict[str, Dict[str, List]], int]

# ==============================================================================
#                               HELPER FUNCTIONS
# ==============================================================================
def _group_ground_truths_by_image(
    ground_truths: LoadedBoxesData, 
    target_class: str
) -> GTsPerImageData:
    """Helper function that filters ground truths by class and organizes them by image name for fast lookup.
    
    Args:
        ground_truths: List of all ground truths.
        target_class: Class name to filter the ground truths.

    Returns:
        A tuple containing the dictionary with the instances of the target class per image, and the total number of the 
        instances of the class.
    """

    gts_per_image = defaultdict(lambda: {'boxes': [], 'matches': []})
    for gt in ground_truths:
        if gt['class'] != target_class:
            continue

        image_name = gt['image_name']
        gts_per_image[image_name]['boxes'].append(gt['box'])
        gts_per_image[image_name]['matches'].append(False)
    
    instances_number = sum(len(image['boxes']) for image in gts_per_image.values())

    return gts_per_image, instances_number

def _filter_predictions_and_sort(
    predictions: LoadedBoxesData, 
    target_class: str
) -> LoadedBoxesData:
    """Helper function that filters predictions by class and sorts them by confidence score."""

    predictions_of_class = [p for p in predictions if p['class'] == target_class]
    predictions_of_class.sort(key=lambda d: d['score'], reverse=True)
        
    return predictions_of_class

# ==============================================================================
#                        CORE FUNCTIONS FOR EVALUATION
# ==============================================================================
def find_matching_predictions(
    predictions: LoadedBoxesData, 
    ground_truths: LoadedBoxesData, 
    target_class: str =None, 
    iou_thresh: float =.5
) -> MatchedPredsData:
    
    """Matches the predictions with the ground truths of a specific object class
    
    Args
        predictions: List of predictions.
        ground_truths: List of ground truths.
        target_class: Class name to filter the predictions.
        iou_thresh: Intersection-over-Union threshold to determine matches.
    
    Returns:
        A tuple containing True Positives, False Positives, scores, and the number of GT instances 
        for the target class.
    """

    gts_per_image, instances_number = _group_ground_truths_by_image(ground_truths, target_class)
    sorted_predictions = _filter_predictions_and_sort(predictions, target_class)

    preds_num = len(sorted_predictions)

    # Initialize vectors "tp" and "fp"
    tp = torch.zeros(preds_num)
    fp = torch.zeros(preds_num)

    for i, info in enumerate(sorted_predictions):
        image_name = info['image_name']
        pred_box = info['box']

        # If the image name is not in gts_per_image keys, or has no boxes (not possible in practice), then the prediction is FP
        if image_name not in gts_per_image or len(gts_per_image[image_name]['boxes']) == 0:
            fp[i] = 1
            continue
        
        gt_boxes = gts_per_image[image_name]['boxes']
        matches = gts_per_image[image_name]['matches']
        
        ious = box_iou(pred_box.unsqueeze(0), torch.stack(gt_boxes)) # returns a tensor 1xN
        best_iou, best_id = torch.max(ious, dim=1)

        if best_iou >= iou_thresh and matches[best_id] == False:
            tp[i] = 1
            matches[best_id] = True
        else:
            fp[i] = 1
            
    scores = torch.Tensor([p['score'] for p in sorted_predictions])
    return tp, fp, scores, instances_number

def calculate_precision_recall_f1(
    tp: torch.Tensor, 
    fp: torch.Tensor, 
    scores: torch.Tensor, 
    instances_number: int,
    score_threshold: float =.25
) -> Tuple[float, float, float]:
    """
    Filters the predictions by a score threshold and calculates Precision, recall, and F1-score 

    Args:
        tp: Binary vector of size=predictions_number, which indicates the true positives.
        fp: Binary vector of size=predictions_number, which indicates the false positives.
        scores: Vector with the confidence scores corresponding to the tp/fp tensors.
        instances_number: The number of the ground truth boxes.
        score_threshold: Minimum confidence score to filter the predictions.
    Returns:
        A tuple of(Precision, Recall, and F1-score).
    """
    if instances_number == 0 or len(scores) == 0:
        return (0.0, 0.0, 0.0)
    
    tp = torch.as_tensor(tp, dtype=torch.float32)
    fp = torch.as_tensor(fp, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32)   

    # Filter the predictions according to the confidence score threshold, in order to calculate P, R, F1
    keep_indices = torch.where(scores>score_threshold)
    tp = tp[keep_indices]
    fp = fp[keep_indices]
    total_tp = torch.sum(tp)
    total_fp = torch.sum(fp)

    epsilon = 1e-16
    
    # Precision
    precision = total_tp / (total_tp + total_fp + epsilon)

    # Recall
    recall = total_tp / (instances_number + epsilon)

    # F1-score
    f1 = 2 * precision * recall / (precision + recall + epsilon)

    return precision.item(), recall.item(), f1.item()

def calculate_average_precision(
    tp: torch.Tensor, 
    fp: torch.Tensor, 
    scores: torch.Tensor, 
    instances_number: int
) -> float:
    """
    Calculates the average precision using the all-points summation method.

    Args:
        tp: Binary vector of size=predictions_number, which indicates the true positives.
        fp: Binary vector of size=predictions_number, which indicates the false positives.
        scores: Vector with the confidence scores of the predictions, to be used for sorting.
        instances_number: The number of the ground truth boxes.
    Returns:
        The Average Precision.
    """

    if instances_number == 0 or len(scores) == 0:
        return 0.0
    
    tp = torch.as_tensor(tp, dtype=torch.float32)
    fp = torch.as_tensor(fp, dtype=torch.float32)
    scores = torch.as_tensor(scores, dtype=torch.float32)

    assert len(tp) == len(fp) == len(scores), 'The vectors "TP", "FP", and "scores" must have the same length!'

    # Sort the predictions in case they are not sorted already (as expected)
    if len(scores) > 1 and not (scores[:-1] >= scores[1:]).all():
        sorted_indices = torch.argsort(scores, descending=True)
        tp = tp[sorted_indices]
        fp = fp[sorted_indices]
    
    cum_tp = torch.cumsum(tp, dim=0)
    cum_fp = torch.cumsum(fp, dim=0)
    
    cum_precision = cum_tp / (cum_tp + cum_fp + 1e-16)

    # Smooth the Precision-Recall curve to ensure precision is monotonically decreasing. At each recall level, the precision value is replaced by 
    # the maximum precision value between that recall level or any higher recall level [PASCAL VOC, Everingham et al. (2010)]
    for i in range(len(cum_precision) - 2, -1, -1): # iterate the cumulative precision vector backwards
        cum_precision[i] = max(cum_precision[i], cum_precision[i + 1])

    cum_recall = cum_tp / instances_number
    
    shifted_cum_recall = torch.cat([torch.zeros(1), cum_recall[:-1]])
    df_recall = cum_recall - shifted_cum_recall # Recall[i] - Recall[i-1]
    
    average_precision = torch.sum(cum_precision * df_recall)

    return average_precision.item()

def evaluate_predictions(
    total_preds: LoadedBoxesData, 
    total_gts: LoadedBoxesData, 
    classes: List[Any], 
    score_threshold: float = 0.25
) -> MetricsData:
    
    """Orchestrates the evaluation pipeline and calculates metrics.
    
    Args:
        total_preds: List of predicted boxes.
        total_gts: List of ground truth boxes.
        classes: List of class names.
        score_threshold: Minimum confidence score to filter the predictions (used for P, R, F1).
    
    Returns:
        A dictionary containing the results of the metrics.
    """

    all_metrics = {} # Dictionary to store the overall results for the evaluation
    total_matches, total_instances = [], 0 # They will be used to calculate micro averages
    total_prec, total_rec, total_f1, total_ap = 0, 0, 0, 0 # They will be used to calculate macro averages
    
    for cls in classes:
        # Find matches
        tp, fp, scores, inst_num = find_matching_predictions(total_preds, total_gts, target_class=cls)
        
        # Calculate metrics
        prec, rec, f1 = calculate_precision_recall_f1(tp, fp, scores, inst_num, score_threshold=score_threshold)
        avg_prec = calculate_average_precision(tp, fp, scores, inst_num)
    
        total_matches.extend(zip(tp.tolist(), fp.tolist(), scores.tolist()))
        total_instances += inst_num
        
        total_prec += prec
        total_rec += rec
        total_f1 += f1
        total_ap += avg_prec
    
        all_metrics[cls] = {
            'Precision': prec,
            'Recall': rec,
            'f1': f1,
            'AP_50': avg_prec,
            'Boxes num.': int(inst_num)
        }
    
    if len(total_matches) > 0:
        total_matches.sort(key=lambda x: x[-1], reverse=True)
        all_tp, all_fp, all_scores = zip(*total_matches)
        
        micro_prec, micro_rec, micro_f1 = calculate_precision_recall_f1(all_tp, all_fp, all_scores, total_instances, score_threshold=score_threshold)
        micro_avg_prec = calculate_average_precision(all_tp, all_fp, all_scores, total_instances)
    
    else:
        micro_prec, micro_rec, micro_f1, micro_avg_prec = 0.0, 0.0, 0.0, 0.0
    
    # Micro averages
    all_metrics['micro avg'] = {
        'Precision': micro_prec,
        'Recall': micro_rec,
        'f1': micro_f1,
        'AP_50': micro_avg_prec,
        'Boxes num.': int(total_instances)
    }
    
    # Macro averages
    cls_num = len(classes) if len(classes) > 0 else 1 # In practice, len(classes) will never be 0, but the code needs to "catch" this case
    
    all_metrics['macro avg'] = {
        'Precision': total_prec / cls_num,
        'Recall': total_rec / cls_num,
        'f1': total_f1 / cls_num,
        'AP_50': total_ap / cls_num,
        'Boxes num.': None
    }

    return all_metrics

# ==============================================================================
#                             LOAD & EXPORT DATA
# ==============================================================================
def load_boxes(
    filepath: Path, 
    classes: List[str], 
    with_conf: bool=False
) -> LoadedBoxesData:
    """Loads bounding boxes from a .txt file. Each line in the .txt contains the following information:
    - For Ground Truth boxes: [object_class, X1, Y1, X2, Y2]
    - For predicted boxes: [object_class, X1, Y1, X2, Y2, confidence_score]

    Args:
        filepath: Path where the .txt file is.
        classes: List of class names to map class IDs to names.
        with_conf: Boolean flag to indicate whether the file contains GT or predictions.

    Returns:
        A list of dictionaries with the following keys:
            'image_name': The corresponding filename of the box.
            'class': The object class of the box.
            'box': The coordinates [x1, y1, x2, y2] of the box.
            'score': Confidence score (only if with_conf=True).
    """

    filepath = Path(filepath)
    image_name = filepath.with_suffix('.png').name 
    boxes = []
    if not filepath.exists():
        return boxes

    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for l in lines:
        entries = l.strip().split()
        
        if len(entries) == 0:
            continue

        # The file is expected to contain class indices that correspond to the classes list
        try:
            cls_idx = int(float(entries[0]))
            class_name = classes[cls_idx]
        except (ValueError, IndexError):
            print(f'Invalid class ID {entries[0]} in {filepath}. Skipping.')
            continue        
        
        x1, y1, x2, y2 = map(float, entries[1:5])
        box = torch.tensor([x1, y1, x2, y2])

        entry = {
            'image_name': image_name,
            'class': class_name,
            'box': box
        }            
        
        if with_conf:
            entry['score'] = float(entries[5])
        
        boxes.append(entry)

    return boxes

def _is_valid_file(filepath: Path) -> bool:
    """Helper function to filter out any hidden .txt files, or files that are in hidden folders 
    (e.g. ".ipynb_checkpoints", ".git", etc.)"""

    check = lambda part: not part.startswith('.') or part in {'.', '..'}
    return all(map(check, filepath.parts))

def load_predictions_and_gts(
    predictions_dir: str,
    ground_truths_dir: str,
    classes: List[str],
) -> Tuple[LoadedBoxesData, LoadedBoxesData]:
    """
    Loads prediction and ground truth files from their respective directories. Ensures strict 1-to-1 filename matching 
    before loading.
    """
    predictions_dir = Path(predictions_dir)
    ground_truths_dir = Path(ground_truths_dir)
       
    pred_files = list(filter(_is_valid_file, predictions_dir.rglob('*.txt')))
    gt_files = list(filter(_is_valid_file, ground_truths_dir.rglob('*.txt')))
    
    if len(pred_files) != len(gt_files):
            raise ValueError(f'File count mismatch! Found {len(pred_files)} predictions and {len(gt_files)} ground '\
                             f'truths. The filenames must match 1-to-1.')    

    # Sort according to the filenames
    sorting_key = lambda path: path.name
    pred_files = sorted(pred_files, key=sorting_key)
    gt_files = sorted(gt_files, key=sorting_key)
    
    total_preds = []
    total_gts = []
    
    for g, p in zip(gt_files, pred_files):
    
        assert g.name == p.name, 'The files must match 1-by-1!'
        
        total_gts.extend(load_boxes(g, classes, with_conf=False))
        total_preds.extend(load_boxes(p, classes, with_conf=True))

    return total_preds, total_gts

def export_to_csv(
    metrics: MetricsData, 
    output_path: Path
) -> None:
    """ Saves metrics to a CSV file."""

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Precision', 'Recall', 'f1', 'AP_50', 'Boxes num.'])
        for cls, cls_metrics in metrics.items():
            row_values = list(cls_metrics.values())
            writer.writerow([cls, *row_values])

def export_to_json(
    metrics: MetricsData, 
    output_path: Path
) -> None:
    """Saves metrics to a JSON file."""

    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

def export_metrics(
    metrics: MetricsData, 
    output_dir: Path, 
    base_filename: str = 'report', 
    output_format: str = 'csv'
) -> None:
    """Base function to save the metrics to a file. It calls the respective export function, according to the preferred 
    format."""

    output_format = output_format.lower()
    # Exporters, to map the given "output_format" to the respective function
    exporters = {
        'csv': (export_to_csv, '.csv'),
        'json': (export_to_json, '.json')
    }
    
    if output_format not in exporters:
        raise ValueError(f"Unknown format: {output_format}")

    export_function, extension = exporters[output_format]

    # Set up the paths for the output
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / f'{base_filename}{extension}'

    # Save the metrics
    export_function(metrics, output_path)
    print(f'Evaluation results were saved to:\n{output_path.absolute()}')

# ==============================================================================
#                            CONFIGURATION
# ==============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(description='Object Detection Evaluator')
    
    parser.add_argument('--pred_dir', type=str, required=True, help='Path to prediction txt files')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to ground truth txt files')
    parser.add_argument('--classes_file', type=str, required=True, help='Path to the txt with the classes(one class per line)')
    
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory for the evaluation report file')
    parser.add_argument('--file_basename', type=str, default='report', help='Output filename')
    parser.add_argument('--output_format', type=str, default='csv', help='Output format')
    parser.add_argument('--score_thresh', type=float, default=0.25, help='Score threshold for Precission, Recall, and F1')
    
    return parser.parse_args()

def load_classes_from_txt(txt_path: Path) -> List[str]:
    """Loads the class names from a .txt. Expects one class per line"""

    if not Path(txt_path).exists():
        raise FileNotFoundError(f"Classes file not found: {txt_path}")
    with open(txt_path) as txt:
        return [line.strip() for line in txt if line.strip()]
    
def main():
    """Main function to run the script from the terminal."""

    args = parse_arguments()
    try:
        classes = load_classes_from_txt(args.classes_file)
        preds, gts = load_predictions_and_gts(args.pred_dir, args.gt_dir, classes)
        results = evaluate_predictions(preds, gts, classes, args.score_thresh)
        export_metrics(results, args.output_dir, args.file_basename, args.output_format)
        
        # Print the results to the terminal
        import pandas as pd
        df = pd.DataFrame.from_dict(results, orient='index')
        if 'Boxes num.' in df.columns:
            df['Boxes num.'] = df['Boxes num.'].astype('Int64')
        print('\nEvaluation report:')
        print(df)
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")

if __name__ == '__main__':
    main()
